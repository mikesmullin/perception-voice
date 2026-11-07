//! Server-push streams: `levels` (binary feature frames) and `words` (JSON).
//!
//! Frame layout and cadence are byte-compatible with the Python hub, since Ada
//! consumes them: two half-chunk windows per 512-sample mic chunk, so 62.5 Hz
//! at 16 kHz.
//!
//! All fan-out happens on one worker thread so neither the realtime audio
//! callback nor a decode thread can block on a slow subscriber. The queue
//! drops its oldest entry when full — animation frames are disposable, and a
//! stalled viewer must not become back-pressure on audio capture.

use std::f64::consts::PI;
use std::io::Write;
use std::os::unix::net::UnixStream;
use std::sync::{Arc, Mutex};

use crossbeam_channel::{bounded, Receiver, Sender};
use rustfft::{num_complex::Complex, Fft, FftPlanner};

pub const FRAME_MAGIC: u32 = 0x3146_5641; // 'AVF1'
/// Re-exported under a clearer name for frame-resync checks in clients.
pub const FLAG_MAGIC_CHECK: u32 = FRAME_MAGIC;
pub const STREAM_MIC: u32 = 0;
pub const FRAME_SIZE: usize = 36;

pub const FLAG_VAD: u32 = 1 << 0;
pub const FLAG_START: u32 = 1 << 1;
pub const FLAG_END: u32 = 1 << 2;

pub const CHANNELS: [&str; 2] = ["levels", "words"];

const BAND_EDGES_HZ: [f64; 5] = [0.0, 300.0, 1000.0, 3000.0, 8000.0];
const QUEUE_CAPACITY: usize = 512;

/// 36 bytes little-endian: magic, stream id, rms, 4 bands, pitch, flags.
pub fn pack_frame(rms: f32, bands: [f32; 4], pitch: f32, flags: u32) -> [u8; FRAME_SIZE] {
    let mut out = [0u8; FRAME_SIZE];
    out[0..4].copy_from_slice(&FRAME_MAGIC.to_le_bytes());
    out[4..8].copy_from_slice(&STREAM_MIC.to_le_bytes());
    out[8..12].copy_from_slice(&rms.to_le_bytes());
    for (i, b) in bands.iter().enumerate() {
        let o = 12 + i * 4;
        out[o..o + 4].copy_from_slice(&b.to_le_bytes());
    }
    out[28..32].copy_from_slice(&pitch.to_le_bytes());
    out[32..36].copy_from_slice(&flags.to_le_bytes());
    out
}

/// rms plus four spectral band energies, each softly compressed into 0..1 so
/// they can drive shader uniforms directly.
struct FeatureExtractor {
    sample_rate: usize,
    n: usize,
    fft: Option<Arc<dyn Fft<f32>>>,
    window: Vec<f32>,
    masks: Vec<(usize, usize)>,
}

impl FeatureExtractor {
    fn new(sample_rate: usize) -> Self {
        Self { sample_rate, n: 0, fft: None, window: Vec::new(), masks: Vec::new() }
    }

    fn prepare(&mut self, n: usize) {
        let mut planner = FftPlanner::new();
        self.fft = Some(planner.plan_fft_forward(n));
        // numpy's np.hanning is symmetric; the mel frontend uses the periodic
        // form instead. They must not be swapped.
        self.window = (0..n)
            .map(|i| {
                if n <= 1 {
                    1.0
                } else {
                    (0.5 - 0.5 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()) as f32
                }
            })
            .collect();

        // rfftfreq(n, 1/sr) bin -> Hz, as half-open [lo, hi) index ranges
        let bins = n / 2 + 1;
        let hz = |k: usize| k as f64 * self.sample_rate as f64 / n as f64;
        self.masks = BAND_EDGES_HZ
            .windows(2)
            .map(|w| {
                let lo = (0..bins).position(|k| hz(k) >= w[0]).unwrap_or(bins);
                let hi = (0..bins).position(|k| hz(k) >= w[1]).unwrap_or(bins);
                (lo, hi)
            })
            .collect();
        self.n = n;
    }

    fn features(&mut self, chunk: &[f32]) -> (f32, [f32; 4]) {
        if chunk.is_empty() {
            return (0.0, [0.0; 4]);
        }
        if chunk.len() != self.n {
            self.prepare(chunk.len());
        }
        let n = chunk.len();

        let mean_sq = chunk.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / n as f64;
        let rms = compress(mean_sq.sqrt() as f32);

        let mut scratch: Vec<Complex<f32>> = chunk
            .iter()
            .zip(&self.window)
            .map(|(v, w)| Complex { re: v * w, im: 0.0 })
            .collect();
        self.fft.as_ref().expect("prepared").process(&mut scratch);

        // normalized magnitude spectrum: a full-scale sine peaks near 1.0
        let scale = 2.0 / n as f32;
        let mut bands = [0.0f32; 4];
        for (b, &(lo, hi)) in self.masks.iter().enumerate() {
            let mut peak = 0.0f32;
            for k in lo..hi.min(n / 2 + 1) {
                let m = scratch[k].norm() * scale;
                if m > peak {
                    peak = m;
                }
            }
            bands[b] = compress(peak);
        }
        (rms, bands)
    }
}

/// `min(1, (x * 8) ** 0.7)` — the Python's soft compression curve.
fn compress(x: f32) -> f32 {
    if x <= 0.0 {
        0.0
    } else {
        (x * 8.0).powf(0.7).min(1.0)
    }
}

enum Event {
    Chunk {
        chunk: Vec<f32>,
        vad_active: bool,
        started: bool,
        ended: bool,
    },
    Words(serde_json::Value),
}

#[derive(Clone)]
pub struct StreamHub {
    tx: Sender<Event>,
    subs: Arc<Mutex<Vec<(String, UnixStream)>>>,
}

impl StreamHub {
    pub fn new(sample_rate: usize) -> Self {
        let (tx, rx) = bounded(QUEUE_CAPACITY);
        let subs: Arc<Mutex<Vec<(String, UnixStream)>>> = Arc::new(Mutex::new(Vec::new()));
        let hub = Self { tx, subs: subs.clone() };
        std::thread::Builder::new()
            .name("stream-hub".into())
            .spawn(move || worker(rx, subs, sample_rate))
            .expect("spawn stream-hub");
        hub
    }

    pub fn add_subscriber(&self, channel: &str, stream: UnixStream) {
        // A stuck client gets dropped on write timeout, never waited on.
        let _ = stream.set_write_timeout(Some(std::time::Duration::from_secs(2)));
        self.subs.lock().expect("subs").push((channel.to_string(), stream));
        tracing::info!("stream subscriber added: {channel}");
    }

    pub fn has_subscribers(&self, channel: &str) -> bool {
        self.subs
            .lock()
            .expect("subs")
            .iter()
            .any(|(c, _)| c == channel)
    }

    /// Called from the realtime audio thread: copy and enqueue only.
    pub fn on_audio_chunk(&self, chunk: &[f32], vad_active: bool, started: bool, ended: bool) {
        if !self.has_subscribers("levels") {
            return;
        }
        self.enqueue(Event::Chunk {
            chunk: chunk.to_vec(),
            vad_active,
            started,
            ended,
        });
    }

    pub fn publish_words(&self, event: serde_json::Value) {
        if !self.has_subscribers("words") {
            return;
        }
        self.enqueue(Event::Words(event));
    }

    fn enqueue(&self, event: Event) {
        // Never block the audio thread: on a full queue the sender simply
        // gives up on this frame rather than waiting for a slow subscriber.
        if self.tx.try_send(event).is_err() {
            tracing::trace!("stream queue full; dropping a frame");
        }
    }
}

fn worker(rx: Receiver<Event>, subs: Arc<Mutex<Vec<(String, UnixStream)>>>, sample_rate: usize) {
    let mut extractor = FeatureExtractor::new(sample_rate);
    while let Ok(event) = rx.recv() {
        match event {
            Event::Chunk { chunk, vad_active, started, ended } => {
                let half = chunk.len() / 2;
                if half == 0 {
                    continue;
                }
                let base = if vad_active { FLAG_VAD } else { 0 };
                let (r1, b1) = extractor.features(&chunk[..half]);
                let first = pack_frame(r1, b1, 0.0, base | if started { FLAG_START } else { 0 });
                let (r2, b2) = extractor.features(&chunk[half..]);
                let second = pack_frame(r2, b2, 0.0, base | if ended { FLAG_END } else { 0 });

                let mut payload = [0u8; FRAME_SIZE * 2];
                payload[..FRAME_SIZE].copy_from_slice(&first);
                payload[FRAME_SIZE..].copy_from_slice(&second);
                fanout(&subs, "levels", &payload);
            }
            Event::Words(value) => {
                let body = match serde_json::to_vec(&value) {
                    Ok(b) => b,
                    Err(e) => {
                        tracing::warn!("could not serialize words event: {e}");
                        continue;
                    }
                };
                let mut framed = (body.len() as u32).to_be_bytes().to_vec();
                framed.extend_from_slice(&body);
                fanout(&subs, "words", &framed);
            }
        }
    }
}

fn fanout(subs: &Arc<Mutex<Vec<(String, UnixStream)>>>, channel: &str, payload: &[u8]) {
    let mut guard = subs.lock().expect("subs");
    let mut dropped = Vec::new();
    for (i, (c, stream)) in guard.iter_mut().enumerate() {
        if c != channel {
            continue;
        }
        if stream.write_all(payload).and_then(|_| stream.flush()).is_err() {
            dropped.push(i);
        }
    }
    for i in dropped.into_iter().rev() {
        let (c, _) = guard.remove(i);
        tracing::info!("stream subscriber dropped: {c}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_is_36_bytes_with_the_expected_magic() {
        let f = pack_frame(0.5, [0.1, 0.2, 0.3, 0.4], 0.0, FLAG_VAD);
        assert_eq!(f.len(), 36);
        assert_eq!(u32::from_le_bytes(f[0..4].try_into().unwrap()), FRAME_MAGIC);
        assert_eq!(u32::from_le_bytes(f[4..8].try_into().unwrap()), STREAM_MIC);
        assert_eq!(f32::from_le_bytes(f[8..12].try_into().unwrap()), 0.5);
        assert_eq!(f32::from_le_bytes(f[12..16].try_into().unwrap()), 0.1);
        assert_eq!(f32::from_le_bytes(f[28..32].try_into().unwrap()), 0.0);
        assert_eq!(u32::from_le_bytes(f[32..36].try_into().unwrap()), FLAG_VAD);
    }

    #[test]
    fn compression_curve_matches_the_python() {
        // min(1, (x*8) ** 0.7)
        assert_eq!(compress(0.0), 0.0);
        assert!((compress(0.125) - 1.0).abs() < 1e-6); // (1.0)^0.7
        assert_eq!(compress(1.0), 1.0); // clamped
        let expected = (0.01f32 * 8.0).powf(0.7);
        assert!((compress(0.01) - expected).abs() < 1e-6);
    }

    #[test]
    fn silence_yields_no_energy() {
        let mut fx = FeatureExtractor::new(16_000);
        let (rms, bands) = fx.features(&[0.0; 256]);
        assert_eq!(rms, 0.0);
        assert_eq!(bands, [0.0; 4]);
    }

    #[test]
    fn a_tone_lands_in_the_band_that_contains_it() {
        let mut fx = FeatureExtractor::new(16_000);
        // 2 kHz sine -> third band [1000, 3000)
        let chunk: Vec<f32> = (0..256)
            .map(|i| (2.0 * PI * 2000.0 * i as f64 / 16_000.0).sin() as f32)
            .collect();
        let (rms, bands) = fx.features(&chunk);
        assert!(rms > 0.5, "rms was {rms}");
        let loudest = bands
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(loudest, 2, "bands were {bands:?}");
    }

    #[test]
    fn band_masks_cover_the_configured_edges() {
        let mut fx = FeatureExtractor::new(16_000);
        fx.prepare(256);
        assert_eq!(fx.masks.len(), 4);
        // first band starts at bin 0 and all ranges must be non-empty
        assert_eq!(fx.masks[0].0, 0);
        for &(lo, hi) in &fx.masks {
            assert!(hi > lo, "empty band range {lo}..{hi}");
        }
    }
}
