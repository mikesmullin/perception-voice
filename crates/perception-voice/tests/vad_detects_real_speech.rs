//! The VAD must actually fire on known-good speech.
//!
//! Capture and decode can both be healthy while the VAD silently gates
//! everything out, which looks exactly like a dead microphone from outside.
//! This runs the recorded fixture through the detector and asserts that a
//! reasonable fraction of its speech frames register.

use std::path::{Path, PathBuf};

use perception_voice::vad::VoiceActivityDetector;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference/dump/input.wav")
}

fn silero_model() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_default())
        .join(".cache/audio8-asr-onnx/vad/silero_vad.onnx")
}

fn read_wav(path: &Path) -> Vec<f32> {
    let mut r = hound::WavReader::open(path).expect("open fixture");
    let spec = r.spec();
    let samples: Vec<i16> = r.samples::<i16>().map(|s| s.expect("sample")).collect();
    if spec.channels == 2 {
        samples
            .chunks(2)
            .map(|c| (c[0] as f32 + c[1] as f32) / 2.0 / 32768.0)
            .collect()
    } else {
        samples.iter().map(|&s| s as f32 / 32768.0).collect()
    }
}

/// Fraction of 512-sample chunks the detector calls speech.
fn speech_ratio(sensitivity: u8, silero: Option<&Path>, audio: &[f32]) -> f64 {
    let mut vad = VoiceActivityDetector::new(sensitivity, 0.05, silero, 16_000).unwrap();
    let mut hits = 0usize;
    let mut total = 0usize;
    for chunk in audio.chunks(512) {
        if chunk.len() < 512 {
            break;
        }
        total += 1;
        if vad.is_speech(chunk).0 {
            hits += 1;
        }
    }
    hits as f64 / total.max(1) as f64
}

#[test]
fn vad_fires_on_recorded_speech() {
    let wav = fixture();
    if !wav.exists() {
        eprintln!("skipping: no fixture at {}", wav.display());
        return;
    }
    let audio = read_wav(&wav);
    let model = silero_model();
    let silero = model.exists().then_some(model.as_path());

    // The fixture is ~4.75s of speech inside ~11.5s, so even a conservative
    // detector should flag a good share of it.
    for sensitivity in [0u8, 1, 2, 3] {
        let ratio = speech_ratio(sensitivity, silero, &audio);
        eprintln!("webrtc_sensitivity={sensitivity}: speech in {:.1}% of chunks", ratio * 100.0);
    }

    let ratio = speech_ratio(3, silero, &audio);
    assert!(
        ratio > 0.15,
        "VAD flagged only {:.1}% of chunks as speech on known-good audio; \
         capture and decode would both look healthy while nothing transcribes",
        ratio * 100.0
    );
}

#[test]
fn webrtc_alone_fires_on_recorded_speech() {
    let wav = fixture();
    if !wav.exists() {
        eprintln!("skipping: no fixture at {}", wav.display());
        return;
    }
    let audio = read_wav(&wav);
    // No Silero: isolates whether the WebRTC stage is the one gating.
    for sensitivity in [0u8, 1, 2, 3] {
        let ratio = speech_ratio(sensitivity, None, &audio);
        eprintln!(
            "webrtc-only sensitivity={sensitivity}: speech in {:.1}% of chunks",
            ratio * 100.0
        );
    }
    let ratio = speech_ratio(3, None, &audio);
    assert!(
        ratio > 0.15,
        "WebRTC alone flagged only {:.1}% of chunks",
        ratio * 100.0
    );
}
