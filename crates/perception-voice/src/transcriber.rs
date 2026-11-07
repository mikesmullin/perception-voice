//! Capture -> VAD segmentation -> transcription, as three stages on their own
//! threads joined by channels.
//!
//! The split is deliberate. The engine loads on the transcription thread, so
//! the microphone, VAD and IPC socket are live long before the model is — the
//! Python build used to queue all of that behind ~4s of model loading, and
//! fixing it was most of an 8.2s -> 1.8s startup. Audio captured during the
//! load is still segmented and queued; only the decode waits.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use audio8_asr::Engine;
use chrono::{DateTime, Local};
use crossbeam_channel::{bounded, Receiver, Sender};

use crate::buffer::TranscriptionBuffer;
use crate::capture::{self, CaptureOptions};
use crate::config::Config;
use crate::streams::StreamHub;
use crate::vad::VoiceActivityDetector;

/// Digital silence / all-zero audio.
pub const LEVEL_FLOOR_DB: f32 = -96.0;

/// ~8s of audio at 512-sample chunks: enough to ride out a decode, small
/// enough that a wedged consumer is noticed rather than hidden.
const CHUNK_QUEUE: usize = 256;
/// Utterances waiting on the engine. Deep enough to cover a model still
/// loading at startup.
const UTTERANCE_QUEUE: usize = 32;

/// VAD construction parameters, separated out because the detector itself
/// cannot cross a thread boundary.
struct VadSettings {
    webrtc_sensitivity: u8,
    silero_sensitivity: f32,
    silero_model: Option<std::path::PathBuf>,
    sample_rate: usize,
}

impl VadSettings {
    /// Catch misconfiguration on the caller's thread.
    fn validate(&self) -> Result<()> {
        if self.sample_rate != 16_000 {
            anyhow::bail!(
                "VAD supports 16 kHz only; config asks for {} Hz",
                self.sample_rate
            );
        }
        Ok(())
    }

    fn build(&self) -> Result<VoiceActivityDetector> {
        VoiceActivityDetector::new(
            self.webrtc_sensitivity,
            self.silero_sensitivity,
            self.silero_model.as_deref(),
            self.sample_rate,
        )
    }
}

struct Utterance {
    audio: Vec<f32>,
    start: DateTime<Local>,
    end: DateTime<Local>,
}

pub struct Transcriber {
    stop: Arc<AtomicBool>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

impl Transcriber {
    /// Start all three stages. Returns once the threads are spawned; the
    /// microphone is live within milliseconds, the engine shortly after.
    pub fn start(
        config: &Config,
        hub: StreamHub,
        buffer: Arc<Mutex<TranscriptionBuffer>>,
    ) -> Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let (chunk_tx, chunk_rx) = bounded::<Vec<f32>>(CHUNK_QUEUE);
        let (utt_tx, utt_rx) = bounded::<Utterance>(UTTERANCE_QUEUE);

        let mut threads = Vec::new();

        // --- capture ------------------------------------------------------
        let capture_opts = CaptureOptions {
            sample_rate: config.audio.sample_rate,
            buffer_size: config.audio.buffer_size,
            device: config.audio.mic_device.clone(),
        };
        let capture_stop = stop.clone();
        threads.push(
            std::thread::Builder::new()
                .name("capture".into())
                .spawn(move || {
                    if let Err(e) = capture::run(capture_opts, capture_stop, chunk_tx) {
                        tracing::error!("capture supervisor exited: {e}");
                    }
                })?,
        );

        // --- VAD segmentation --------------------------------------------
        // webrtc_vad::Vad holds a raw pointer and is not Send, so it has to be
        // built on the thread that uses it. Validate first, so a bad config
        // fails at startup rather than inside a detached thread.
        let vad_settings = VadSettings {
            webrtc_sensitivity: config.vad.webrtc_sensitivity,
            silero_sensitivity: config.vad.silero_sensitivity,
            silero_model: config
                .vad
                .silero_enabled
                .then(|| config.silero_model_path()),
            sample_rate: config.audio.sample_rate,
        };
        vad_settings.validate()?;

        let seg = Segmenter::new(config);
        let seg_stop = stop.clone();
        let seg_hub = hub.clone();
        threads.push(
            std::thread::Builder::new()
                .name("vad-segmenter".into())
                .spawn(move || {
                    let vad = match vad_settings.build() {
                        Ok(v) => v,
                        Err(e) => {
                            tracing::error!("VAD failed to initialize: {e:#}");
                            return;
                        }
                    };
                    seg.run(vad, chunk_rx, utt_tx, seg_hub, seg_stop)
                })?,
        );

        // --- transcription ------------------------------------------------
        let bundle = config.bundle_dir();
        let threads_n = if config.audio8.threads == 0 {
            std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8)
        } else {
            config.audio8.threads
        };
        let max_new_tokens = config.audio8.max_new_tokens;
        let sample_rate = config.audio.sample_rate;
        let tr_hub = hub.clone();
        threads.push(
            std::thread::Builder::new()
                .name("transcribe".into())
                .spawn(move || {
                    let started = std::time::Instant::now();
                    let engine = match Engine::load(&bundle, threads_n) {
                        Ok(e) => {
                            tracing::info!(
                                "STT engine ready in {:.2}s ({} threads)",
                                started.elapsed().as_secs_f64(),
                                threads_n
                            );
                            e
                        }
                        Err(e) => {
                            tracing::error!("STT engine failed to load: {e:#}");
                            return;
                        }
                    };
                    run_transcription(
                        engine,
                        utt_rx,
                        buffer,
                        tr_hub,
                        max_new_tokens,
                        sample_rate,
                    );
                })?,
        );

        Ok(Self { stop, threads })
    }

    pub fn stop(self) {
        self.stop.store(true, Ordering::Relaxed);
        for t in self.threads {
            // Threads are cooperative and poll `stop` at 100ms; a join that
            // takes longer than that means something is wedged, and we would
            // rather exit than hang a `systemctl restart`.
            let _ = t.join();
        }
    }
}

/// Turns a chunk stream into utterances using VAD plus a trailing-silence rule.
struct Segmenter {
    buffer_size: usize,
    pre_chunks: usize,
    max_silence_chunks: usize,
    min_utterance_samples: usize,
    sample_rate: usize,
}

impl Segmenter {
    fn new(config: &Config) -> Self {
        let chunks_per_second = config.audio.sample_rate as f32 / config.audio.buffer_size as f32;
        Self {
            buffer_size: config.audio.buffer_size,
            pre_chunks: (chunks_per_second * config.audio.pre_recording_buffer_duration) as usize,
            max_silence_chunks: (chunks_per_second
                * config.audio.post_speech_silence_duration) as usize,
            min_utterance_samples: (config.audio.sample_rate as f32
                * config.audio.min_utterance_duration) as usize,
            sample_rate: config.audio.sample_rate,
        }
    }

    fn run(
        self,
        mut vad: VoiceActivityDetector,
        chunks: Receiver<Vec<f32>>,
        utterances: Sender<Utterance>,
        hub: StreamHub,
        stop: Arc<AtomicBool>,
    ) {
        // Rolling pre-roll so an utterance includes the audio just before VAD
        // fired; without it every recording clips its own first syllable.
        let mut pre_roll: std::collections::VecDeque<Vec<f32>> =
            std::collections::VecDeque::with_capacity(self.pre_chunks + 1);
        let mut frames: Vec<f32> = Vec::new();
        let mut recording = false;
        let mut silence_chunks = 0usize;
        let mut started_at: Option<DateTime<Local>> = None;

        while !stop.load(Ordering::Relaxed) {
            let chunk = match chunks.recv_timeout(Duration::from_millis(200)) {
                Ok(c) => c,
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
            };

            let (is_speech, _confidence) = vad.is_speech(&chunk);
            let was_recording = recording;

            if is_speech {
                if !recording {
                    recording = true;
                    started_at = Some(Local::now());
                    frames.clear();
                    for pre in &pre_roll {
                        frames.extend_from_slice(pre);
                    }
                }
                frames.extend_from_slice(&chunk);
                silence_chunks = 0;
            } else if recording {
                frames.extend_from_slice(&chunk);
                silence_chunks += 1;
                if silence_chunks >= self.max_silence_chunks {
                    recording = false;
                    let audio = std::mem::take(&mut frames);
                    let start = started_at.take().unwrap_or_else(Local::now);
                    if audio.len() >= self.min_utterance_samples {
                        if utterances
                            .try_send(Utterance { audio, start, end: Local::now() })
                            .is_err()
                        {
                            tracing::warn!(
                                "transcription queue full; dropped an utterance"
                            );
                        }
                    } else {
                        tracing::debug!(
                            "utterance too short ({:.2}s); discarding",
                            audio.len() as f32 / self.sample_rate as f32
                        );
                    }
                }
            }

            hub.on_audio_chunk(
                &chunk,
                is_speech,
                !was_recording && recording,
                was_recording && !recording,
            );

            pre_roll.push_back(chunk);
            while pre_roll.len() > self.pre_chunks {
                pre_roll.pop_front();
            }
            debug_assert!(self.buffer_size > 0);
        }
        tracing::info!("vad segmenter stopped");
    }
}

fn run_transcription(
    mut engine: Engine,
    utterances: Receiver<Utterance>,
    buffer: Arc<Mutex<TranscriptionBuffer>>,
    hub: StreamHub,
    max_new_tokens: usize,
    sample_rate: usize,
) {
    while let Ok(utterance) = utterances.recv() {
        let level_db = utterance_level_db(&utterance.audio, sample_rate);
        let result = match engine.transcribe(&utterance.audio, max_new_tokens) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("transcription failed: {e:#}");
                continue;
            }
        };
        let text = result.text.trim().to_string();
        if text.is_empty() {
            continue;
        }

        let stored = buffer
            .lock()
            .expect("buffer")
            .add(&text, utterance.start, level_db);
        tracing::info!(
            "transcribed {} words{}{}",
            text.split_whitespace().count(),
            level_db.map(|d| format!(" (P95 {d:.0}dB)")).unwrap_or_default(),
            if stored { "" } else { " [discarded]" }
        );

        // Push to `subscribe words` the moment the utterance is final, so
        // downstream consumers never have to poll.
        hub.publish_words(serde_json::json!({
            "ev": "utterance",
            "ts": chrono::Local::now().timestamp_millis() as f64 / 1000.0,
            "text": text,
            "t_start": utterance.start.timestamp_millis() as f64 / 1000.0,
            "t_end": utterance.end.timestamp_millis() as f64 / 1000.0,
        }));
    }
    tracing::info!("transcription worker stopped");
}

/// P95 of per-window RMS, in dBFS.
///
/// The 95th percentile rather than the peak so one clipped sample cannot
/// dominate the reading. A rough speaker cue (nearer voices score higher),
/// not a calibrated measurement.
pub fn utterance_level_db(audio: &[f32], sample_rate: usize) -> Option<f32> {
    if audio.is_empty() {
        return None;
    }
    let window = (sample_rate / 32).max(1);
    let windows = audio.len() / window;
    if windows == 0 {
        let mean_sq = audio.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>()
            / audio.len() as f64;
        let rms = mean_sq.sqrt();
        return Some(if rms <= 0.0 {
            LEVEL_FLOOR_DB
        } else {
            (20.0 * rms.log10()).max(LEVEL_FLOOR_DB as f64) as f32
        });
    }

    let mut rms: Vec<f64> = (0..windows)
        .map(|w| {
            let slice = &audio[w * window..(w + 1) * window];
            let mean_sq =
                slice.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / window as f64;
            mean_sq.sqrt()
        })
        .collect();
    rms.sort_by(|a, b| a.partial_cmp(b).expect("finite"));

    // numpy's default linear-interpolation percentile
    let pos = 0.95 * (rms.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let p95 = rms[lo] + (rms[hi] - rms[lo]) * (pos - lo as f64);

    Some(if p95 <= 0.0 {
        LEVEL_FLOOR_DB
    } else {
        (20.0 * p95.log10()).max(LEVEL_FLOOR_DB as f64) as f32
    })
}
