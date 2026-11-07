//! Dual voice-activity detection: WebRTC as a cheap gate, Silero to confirm.
//!
//! WebRTC costs ~1us and rejects most silence, so Silero (~1ms of ONNX) only
//! runs on candidates. The ordering matters for behaviour as well as cost:
//! WebRTC at its most aggressive mode also rejects *quiet speech*, and because
//! it short-circuits, Silero never gets a say on those chunks. That is a real
//! trade-off, not an implementation detail — a quiet microphone can go
//! undetected — so `vad.webrtc_sensitivity` is worth lowering before blaming
//! the model when speech is being missed.

use std::path::Path;

use anyhow::{anyhow, Result};
use ort::session::Session;
use ort::value::Tensor;
use webrtc_vad::{SampleRate, Vad, VadMode};

/// WebRTC needs a 10, 20 or 30 ms frame; 30 ms at 16 kHz is 480 samples.
const WEBRTC_FRAME: usize = 480;
/// Silero v5 at 16 kHz expects exactly this many new samples per call.
const SILERO_FRAME: usize = 512;
/// ...prefixed by this much of the *previous* frame. The model was exported
/// expecting 576 samples; feeding it a bare 512 is accepted and returns
/// near-zero probabilities for everything, which is indistinguishable from a
/// microphone that hears nothing.
const SILERO_CONTEXT: usize = 64;
const SILERO_STATE: usize = 128;

pub struct VoiceActivityDetector {
    webrtc: Vad,
    silero: Option<Silero>,
    silero_sensitivity: f32,
    sample_rate: usize,
    pub webrtc_checks: u64,
    pub silero_checks: u64,
    pub speech_detected: u64,
}

struct Silero {
    session: Session,
    /// `[2, 1, 128]`, carried across calls — Silero is stateful, and resetting
    /// it per chunk measurably degrades detection at utterance starts.
    state: Vec<f32>,
    /// Tail of the previous frame, prepended to the next one.
    context: Vec<f32>,
}

impl VoiceActivityDetector {
    pub fn new(
        webrtc_sensitivity: u8,
        silero_sensitivity: f32,
        silero_model: Option<&Path>,
        sample_rate: usize,
    ) -> Result<Self> {
        if sample_rate != 16_000 {
            return Err(anyhow!(
                "VAD supports 16 kHz only; config asks for {sample_rate} Hz"
            ));
        }
        let mode = match webrtc_sensitivity {
            0 => VadMode::Quality,
            1 => VadMode::LowBitrate,
            2 => VadMode::Aggressive,
            _ => VadMode::VeryAggressive,
        };
        tracing::info!("initializing WebRTC VAD (sensitivity={webrtc_sensitivity})");
        let webrtc = Vad::new_with_rate_and_mode(SampleRate::Rate16kHz, mode);

        let silero = match silero_model {
            Some(path) if path.exists() => {
                tracing::info!("initializing Silero VAD (sensitivity={silero_sensitivity})");
                Some(Silero::load(path)?)
            }
            Some(path) => {
                tracing::warn!(
                    "Silero model not found at {}; running with WebRTC only, \
                     which is noticeably more trigger-happy on noise",
                    path.display()
                );
                None
            }
            None => None,
        };

        Ok(Self {
            webrtc,
            silero,
            silero_sensitivity,
            sample_rate,
            webrtc_checks: 0,
            silero_checks: 0,
            speech_detected: 0,
        })
    }

    /// `(is_speech, confidence)`. Confidence is Silero's probability, or 1.0
    /// when only WebRTC is available.
    pub fn is_speech(&mut self, chunk: &[f32]) -> (bool, f32) {
        self.webrtc_checks += 1;

        let mut frame = [0i16; WEBRTC_FRAME];
        for (i, slot) in frame.iter_mut().enumerate() {
            *slot = match chunk.get(i) {
                Some(v) => (v.clamp(-1.0, 1.0) * 32767.0) as i16,
                None => 0,
            };
        }
        let webrtc_speech = self
            .webrtc
            .is_voice_segment(&frame)
            .unwrap_or(true); // on error, defer to Silero rather than drop audio

        if !webrtc_speech {
            return (false, 0.0);
        }

        let Some(silero) = self.silero.as_mut() else {
            self.speech_detected += 1;
            return (true, 1.0);
        };

        self.silero_checks += 1;
        let prob = match silero.probability(chunk, self.sample_rate) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!("Silero VAD failed ({e}); accepting the WebRTC verdict");
                self.speech_detected += 1;
                return (true, 1.0);
            }
        };
        let is_speech = prob > self.silero_sensitivity;
        if is_speech {
            self.speech_detected += 1;
        }
        (is_speech, prob)
    }

    /// Fraction of chunks WebRTC filtered out before Silero had to run.
    pub fn efficiency_percent(&self) -> f64 {
        if self.webrtc_checks == 0 {
            return 0.0;
        }
        (1.0 - self.silero_checks as f64 / self.webrtc_checks as f64) * 100.0
    }
}

impl Silero {
    fn load(path: &Path) -> Result<Self> {
        // Shared with the ASR engine: one environment per process.
        let env = audio8_asr::ort_env()?;
        let b = Session::builder(env).map_err(|e| anyhow!("session builder: {e}"))?;
        // One thread: this runs 31x/second on a 512-sample frame, where thread
        // hand-off costs more than the work itself.
        let mut b = b
            .with_intra_threads(1)
            .map_err(|e| anyhow!("with_intra_threads: {e}"))?;
        let session = b
            .commit_from_file(path)
            .map_err(|e| anyhow!("loading {}: {e}", path.display()))?;
        Ok(Self {
            session,
            state: vec![0.0; 2 * SILERO_STATE],
            context: vec![0.0; SILERO_CONTEXT],
        })
    }

    fn probability(&mut self, chunk: &[f32], sample_rate: usize) -> Result<f32> {
        let mut frame = [0.0f32; SILERO_FRAME];
        let n = chunk.len().min(SILERO_FRAME);
        frame[..n].copy_from_slice(&chunk[..n]);

        // [context | frame] -> 576 samples, as the export expects.
        let mut samples = Vec::with_capacity(SILERO_CONTEXT + SILERO_FRAME);
        samples.extend_from_slice(&self.context);
        samples.extend_from_slice(&frame);

        let input = Tensor::from_array((
            vec![1_i64, (SILERO_CONTEXT + SILERO_FRAME) as i64],
            samples.clone(),
        ))
        .map_err(|e| anyhow!("building input: {e}"))?;
        let state = Tensor::from_array((
            vec![2_i64, 1, SILERO_STATE as i64],
            self.state.clone(),
        ))
        .map_err(|e| anyhow!("building state: {e}"))?;
        // The graph declares `sr` with rank 0. Handing it a [1] tensor is
        // accepted but silently yields near-zero probabilities, which looks
        // exactly like a microphone that hears nothing.
        let sr = Tensor::from_array((Vec::<i64>::new(), vec![sample_rate as i64]))
            .map_err(|e| anyhow!("building sr: {e}"))?;

        let outputs = self
            .session
            .run(ort::inputs!["input" => input, "state" => state, "sr" => sr])
            .map_err(|e| anyhow!("silero run: {e}"))?;

        let (_, prob) = outputs["output"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("extracting output: {e}"))?;
        let probability = *prob.first().ok_or_else(|| anyhow!("empty output"))?;

        let (_, next) = outputs["stateN"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("extracting stateN: {e}"))?;
        if next.len() == self.state.len() {
            self.state.copy_from_slice(next);
        }
        // Carry the tail forward for the next call.
        self.context
            .copy_from_slice(&samples[samples.len() - SILERO_CONTEXT..]);
        Ok(probability)
    }
}
