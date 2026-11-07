//! Whisper-style 128-bin log-mel spectrogram.
//!
//! Must match `transformers`' `WhisperFeatureExtractor` numerically, because
//! the audio encoder was traced against its output. Every choice here mirrors
//! a specific decision in `transformers.audio_utils`:
//!
//! - periodic Hann window (`np.hanning(n_fft + 1)[:-1]`), not symmetric
//! - `center=True` with reflect padding of `n_fft / 2` on both ends
//! - power spectrum (magnitude squared), one-sided
//! - Slaney mel scale *and* Slaney filter normalization
//! - log10 with a 1e-10 floor, then clamp to `max - 8`, then `(x + 4) / 4`
//!
//! The final frame is dropped, as the Whisper extractor does.

use std::f64::consts::PI;
use std::sync::Arc;

use rustfft::{num_complex::Complex, Fft, FftPlanner};

pub const SAMPLE_RATE: usize = 16_000;
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const N_MELS: usize = 128;

/// Slaney-scale Hz -> mel.
fn hz_to_mel(freq: f64) -> f64 {
    const F_MIN: f64 = 0.0;
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    let min_log_mel = (MIN_LOG_HZ - F_MIN) / F_SP;
    let logstep = (6.4f64).ln() / 27.0;
    if freq >= MIN_LOG_HZ {
        min_log_mel + (freq / MIN_LOG_HZ).ln() / logstep
    } else {
        (freq - F_MIN) / F_SP
    }
}

/// Slaney-scale mel -> Hz.
fn mel_to_hz(mel: f64) -> f64 {
    const F_MIN: f64 = 0.0;
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    let min_log_mel = (MIN_LOG_HZ - F_MIN) / F_SP;
    let logstep = (6.4f64).ln() / 27.0;
    if mel >= min_log_mel {
        MIN_LOG_HZ * ((mel - min_log_mel) * logstep).exp()
    } else {
        F_MIN + F_SP * mel
    }
}

/// Triangular mel filterbank, `[n_mels][n_freq_bins]`, Slaney-normalized.
fn mel_filter_bank(n_freq_bins: usize, n_mels: usize, sample_rate: usize) -> Vec<Vec<f32>> {
    let fmin = 0.0f64;
    let fmax = sample_rate as f64 / 2.0;

    // n_mels + 2 band edges in mel space, linearly spaced
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);
    let filter_freqs: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64))
        .collect();

    // FFT bin centre frequencies
    let fft_freqs: Vec<f64> = (0..n_freq_bins)
        .map(|i| i as f64 * fmax / (n_freq_bins - 1) as f64)
        .collect();

    let mut filters = vec![vec![0.0f32; n_freq_bins]; n_mels];
    for m in 0..n_mels {
        let (lo, ctr, hi) = (filter_freqs[m], filter_freqs[m + 1], filter_freqs[m + 2]);
        // Slaney normalization: unit *area* rather than unit peak.
        let enorm = 2.0 / (hi - lo);
        for (k, &f) in fft_freqs.iter().enumerate() {
            let up = if ctr > lo { (f - lo) / (ctr - lo) } else { 0.0 };
            let down = if hi > ctr { (hi - f) / (hi - ctr) } else { 0.0 };
            let w = up.min(down).max(0.0);
            filters[m][k] = (w * enorm) as f32;
        }
    }
    filters
}

pub struct MelExtractor {
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    filters: Vec<Vec<f32>>,
}

impl Default for MelExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl MelExtractor {
    pub fn new() -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(N_FFT);
        // Periodic Hann: np.hanning(N+1)[:-1]
        let window = (0..N_FFT)
            .map(|n| (0.5 - 0.5 * (2.0 * PI * n as f64 / N_FFT as f64).cos()) as f32)
            .collect();
        let filters = mel_filter_bank(N_FFT / 2 + 1, N_MELS, SAMPLE_RATE);
        Self { fft, window, filters }
    }

    /// Log-mel spectrogram of `audio`, returned as `[n_mels][n_frames]`.
    pub fn log_mel(&self, audio: &[f32]) -> Vec<Vec<f32>> {
        let pad = N_FFT / 2;
        let padded = reflect_pad(audio, pad);

        if padded.len() < N_FFT {
            return vec![vec![0.0; 0]; N_MELS];
        }
        // transformers: num_frames = 1 + floor((len - frame_length) / hop)
        let n_frames = 1 + (padded.len() - N_FFT) / HOP_LENGTH;

        let n_bins = N_FFT / 2 + 1;
        let mut power = vec![0.0f32; n_frames * n_bins];
        let mut scratch: Vec<Complex<f32>> = vec![Complex { re: 0.0, im: 0.0 }; N_FFT];

        for t in 0..n_frames {
            let base = t * HOP_LENGTH;
            for i in 0..N_FFT {
                scratch[i] = Complex {
                    re: padded[base + i] * self.window[i],
                    im: 0.0,
                };
            }
            self.fft.process(&mut scratch);
            for k in 0..n_bins {
                let c = scratch[k];
                power[t * n_bins + k] = c.re * c.re + c.im * c.im;
            }
        }

        // mel projection, then log10 with a floor
        let mut mel = vec![vec![0.0f32; n_frames]; N_MELS];
        for (m, filt) in self.filters.iter().enumerate() {
            for t in 0..n_frames {
                let row = &power[t * n_bins..(t + 1) * n_bins];
                let mut acc = 0.0f32;
                for k in 0..n_bins {
                    acc += filt[k] * row[k];
                }
                mel[m][t] = acc.max(1e-10).log10();
            }
        }

        // Whisper drops the trailing frame.
        if n_frames > 1 {
            for row in mel.iter_mut() {
                row.pop();
            }
        }

        // clamp to max-8, then scale into roughly [-1, 1]
        let mut max = f32::NEG_INFINITY;
        for row in &mel {
            for &v in row {
                if v > max {
                    max = v;
                }
            }
        }
        let floor = max - 8.0;
        for row in mel.iter_mut() {
            for v in row.iter_mut() {
                *v = (v.max(floor) + 4.0) / 4.0;
            }
        }
        mel
    }
}

/// `np.pad(x, pad, mode="reflect")` — mirrors without repeating the edge:
/// `[1,2,3,4]` padded by 2 becomes `[3,2,1,2,3,4,3,2]`.
///
/// numpy reflects repeatedly when `pad >= len`; this does the same by walking
/// a triangle-wave index, so it is correct for any input length. In practice
/// utterances are far longer than the 200-sample pad (the VAD will not emit
/// anything under `min_utterance_duration`), but a short clip must not panic.
fn reflect_pad(x: &[f32], pad: usize) -> Vec<f32> {
    if x.is_empty() {
        return vec![0.0; 2 * pad];
    }
    if x.len() == 1 {
        return vec![x[0]; 2 * pad + 1];
    }
    let mut out = Vec::with_capacity(x.len() + 2 * pad);
    for i in (1..=pad).rev() {
        out.push(x[reflect_index(i as isize, x.len())]);
    }
    out.extend_from_slice(x);
    for i in 1..=pad {
        out.push(x[reflect_index(x.len() as isize - 1 - i as isize, x.len())]);
    }
    out
}

/// Map a possibly out-of-range index onto `0..len` by reflecting at both ends,
/// excluding the endpoints — the triangle wave numpy's "reflect" mode follows.
fn reflect_index(i: isize, len: usize) -> usize {
    debug_assert!(len >= 2);
    let period = 2 * (len as isize - 1);
    let mut k = i.rem_euclid(period);
    if k >= len as isize {
        k = period - k;
    }
    k as usize
}
