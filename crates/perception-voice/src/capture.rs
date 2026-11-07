//! Supervised microphone capture.
//!
//! The Python version learned these the hard way, so they are requirements
//! here rather than refinements:
//!
//! - **A failed open is not terminal.** Retry forever with backoff. The old
//!   code logged once and let the thread die, leaving systemd reporting a
//!   healthy unit with a dead microphone.
//! - **A silent stream is a failed stream.** A capture stream can open and
//!   then stop delivering callbacks — a Bluetooth HFP transport dropping, or
//!   the default source changing underneath. Treat a gap in callbacks as a
//!   failure and reopen.
//! - **Re-resolve the device every attempt.** Never cache a device handle
//!   across failures; the node we had may no longer exist.
//!
//! Audio is handed upward in fixed-size chunks (`buffer_size`) of mono f32 at
//! the configured rate, so the VAD and encoder see exactly what they expect
//! regardless of what the hardware offered.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

pub const RETRY_MIN: Duration = Duration::from_millis(500);
pub const RETRY_MAX: Duration = Duration::from_secs(10);
/// No callbacks for this long means the stream has stopped carrying audio.
pub const STALL_TIMEOUT: Duration = Duration::from_secs(3);

pub struct CaptureOptions {
    pub sample_rate: usize,
    pub buffer_size: usize,
    /// Substring matched against device names; `None` takes the default input.
    pub device: Option<String>,
}

/// Runs capture until `stop` is set, reopening the stream as needed.
///
/// Chunks are published on `tx`. A channel rather than a callback keeps the
/// realtime thread's only job a non-blocking send, and lets each reopened
/// stream own a fresh chunker without the closure needing to be reusable.
pub fn run(
    options: CaptureOptions,
    stop: Arc<AtomicBool>,
    tx: crossbeam_channel::Sender<Vec<f32>>,
) -> Result<()> {
    let last_callback = Arc::new(AtomicU64::new(now_millis()));
    let mut backoff = RETRY_MIN;
    let mut attempt = 0u32;
    let mut announced = false;

    while !stop.load(Ordering::Relaxed) {
        attempt += 1;
        last_callback.store(now_millis(), Ordering::Relaxed);

        match open_stream(&options, &last_callback, &tx) {
            Ok((stream, name, native_rate, channels)) => {
                if let Err(e) = stream.play() {
                    tracing::error!("could not start capture stream (attempt {attempt}): {e}");
                } else {
                    if announced {
                        tracing::warn!(
                            "audio stream reopened on {name} ({native_rate} Hz, {channels} ch, attempt {attempt})"
                        );
                    } else {
                        tracing::info!(
                            "audio stream started on {name} ({native_rate} Hz, {channels} ch)"
                        );
                        announced = true;
                    }
                    backoff = RETRY_MIN;

                    // Supervise until stalled or asked to stop.
                    while !stop.load(Ordering::Relaxed) {
                        std::thread::sleep(Duration::from_millis(100));
                        let idle = now_millis().saturating_sub(last_callback.load(Ordering::Relaxed));
                        if Duration::from_millis(idle) > STALL_TIMEOUT {
                            tracing::error!(
                                "no audio callbacks for {:.1}s on {name} -- reopening the stream",
                                idle as f64 / 1000.0
                            );
                            break;
                        }
                    }
                }
                drop(stream);
            }
            Err(e) => {
                tracing::error!("could not open capture stream (attempt {attempt}): {e}");
            }
        }

        if stop.load(Ordering::Relaxed) {
            break;
        }
        sleep_interruptible(backoff, &stop);
        backoff = (backoff * 2).min(RETRY_MAX);
    }

    tracing::info!("capture supervisor stopped");
    Ok(())
}

type OpenedStream = (cpal::Stream, String, cpal::SampleRate, cpal::ChannelCount);

fn open_stream(
    options: &CaptureOptions,
    last_callback: &Arc<AtomicU64>,
    tx: &crossbeam_channel::Sender<Vec<f32>>,
) -> Result<OpenedStream> {
    let host = cpal::default_host();

    // Re-resolve every attempt; a cached handle may name a destroyed node.
    let device = match options.device.as_deref() {
        Some(wanted) => host
            .input_devices()?
            .find(|d| {
                d.description()
                    .map(|desc| desc.name().contains(wanted))
                    .unwrap_or(false)
            })
            .ok_or_else(|| anyhow!("no input device matching {wanted:?}"))?,
        None => host
            .default_input_device()
            .ok_or_else(|| anyhow!("no default input device"))?,
    };
    let name = device
        .description()
        .map(|d| d.name().to_string())
        .unwrap_or_else(|_| "<unnamed>".into());

    // Prefer the exact rate we want so no resampling is needed; fall back to
    // whatever the device offers. PipeWire's ALSA `default` accepts 16 kHz
    // directly, so the fallback is for raw hardware devices.
    let wanted_rate: cpal::SampleRate = options.sample_rate as cpal::SampleRate;
    let supports_wanted = device
        .supported_input_configs()
        .map(|mut it| {
            it.any(|c| c.min_sample_rate() <= wanted_rate && wanted_rate <= c.max_sample_rate())
        })
        .unwrap_or(false);

    let default_config = device.default_input_config()?;
    let channels = default_config.channels();
    let config = if supports_wanted {
        cpal::StreamConfig {
            channels,
            sample_rate: wanted_rate,
            buffer_size: cpal::BufferSize::Default,
        }
    } else {
        tracing::warn!(
            "{name} does not offer {} Hz; capturing at {} Hz and resampling",
            options.sample_rate,
            default_config.sample_rate()
        );
        default_config.config()
    };

    let native_rate = config.sample_rate;

    // Fresh per stream: a partial chunk belongs to the stream that produced
    // it, and a different device may have a different rate.
    let mut chunker = Chunker::new(options.buffer_size);
    chunker.configure(native_rate as usize, channels as usize, options.sample_rate);

    let cb_last = last_callback.clone();
    let cb_tx = tx.clone();
    let stream = device.build_input_stream(
        config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            cb_last.store(now_millis(), Ordering::Relaxed);
            chunker.push(data, |chunk| {
                // Never block the realtime thread. A full queue means the
                // consumer is wedged, and dropping is the correct response --
                // blocking here would stall capture for every other consumer
                // too.
                if cb_tx.try_send(chunk.to_vec()).is_err() {
                    tracing::warn!("audio queue full; dropping a chunk");
                }
            });
        },
        |err| tracing::error!("capture stream error: {err}"),
        None,
    )?;

    Ok((stream, name, native_rate, channels))
}

/// Downmixes to mono, resamples to the target rate, and emits fixed-size chunks.
pub struct Chunker {
    target_chunk: usize,
    native_rate: usize,
    channels: usize,
    target_rate: usize,
    pending: Vec<f32>,
    /// Fractional read position for non-integer rate ratios.
    phase: f64,
}

impl Chunker {
    pub fn new(target_chunk: usize) -> Self {
        Self {
            target_chunk,
            native_rate: 0,
            channels: 1,
            target_rate: 0,
            pending: Vec::new(),
            phase: 0.0,
        }
    }

    pub fn configure(&mut self, native_rate: usize, channels: usize, target_rate: usize) {
        self.native_rate = native_rate;
        self.channels = channels.max(1);
        self.target_rate = target_rate;
        self.reset();
    }

    pub fn reset(&mut self) {
        self.pending.clear();
        self.phase = 0.0;
    }

    /// Feed interleaved native-rate samples; `emit` is called per full chunk.
    pub fn push<F: FnMut(&[f32])>(&mut self, data: &[f32], mut emit: F) {
        // interleaved -> mono
        let frames = data.len() / self.channels;
        let mono: Vec<f32> = (0..frames)
            .map(|f| {
                let base = f * self.channels;
                data[base..base + self.channels].iter().sum::<f32>() / self.channels as f32
            })
            .collect();

        if self.native_rate == self.target_rate || self.native_rate == 0 {
            self.pending.extend_from_slice(&mono);
        } else {
            self.resample_into_pending(&mono);
        }

        while self.pending.len() >= self.target_chunk {
            let chunk: Vec<f32> = self.pending.drain(..self.target_chunk).collect();
            emit(&chunk);
        }
    }

    /// Box-filtered decimation for integer ratios, linear otherwise.
    ///
    /// A bare linear resample from 48 kHz aliases speech energy above 8 kHz
    /// back down into the band the VAD and encoder care about; averaging over
    /// the decimation window is a crude but real low-pass that avoids it.
    fn resample_into_pending(&mut self, mono: &[f32]) {
        let ratio = self.native_rate as f64 / self.target_rate as f64;
        if (ratio - ratio.round()).abs() < 1e-9 && ratio >= 1.0 {
            let step = ratio.round() as usize;
            let mut i = 0usize;
            while i + step <= mono.len() {
                let sum: f32 = mono[i..i + step].iter().sum();
                self.pending.push(sum / step as f32);
                i += step;
            }
            // carry the remainder so no samples are lost across callbacks
            if i < mono.len() {
                let sum: f32 = mono[i..].iter().sum();
                self.pending.push(sum / (mono.len() - i) as f32);
            }
        } else {
            let mut pos = self.phase;
            while (pos as usize) + 1 < mono.len() {
                let i = pos as usize;
                let frac = (pos - i as f64) as f32;
                self.pending.push(mono[i] * (1.0 - frac) + mono[i + 1] * frac);
                pos += ratio;
            }
            self.phase = pos - mono.len() as f64;
            if self.phase < 0.0 {
                self.phase = 0.0;
            }
        }
    }
}

/// Milliseconds since first use, from a monotonic clock. Wall time would let
/// an NTP step masquerade as a stalled stream.
fn now_millis() -> u64 {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_millis() as u64
}

fn sleep_interruptible(total: Duration, stop: &Arc<AtomicBool>) {
    let deadline = Instant::now() + total;
    while Instant::now() < deadline && !stop.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(50));
    }
}
