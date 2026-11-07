//! End-to-end check: the Rust engine must agree with `reference/oracle.py`
//! on the transcript *and* on the intermediate token accounting.
//!
//! Comparing only the final string would hide a drift that happens to decode
//! to the same words; asserting the audio-token count and prompt length too
//! pins the stages that must line up for the prompt and encoder to agree.
//!
//! Skipped (not failed) when the model bundle or the reference dump is
//! missing, since neither is in git:
//!
//!     python reference/oracle.py <wav> --dump reference/dump

use std::path::{Path, PathBuf};

use audio8_asr::engine::{default_bundle_dir, read_wav_mono16k};
use audio8_asr::Engine;

fn dump_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../reference/dump")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from("/nonexistent"))
}

#[test]
fn transcript_matches_python_reference() {
    let dump = dump_dir();
    let wav = dump.join("input.wav");
    let expected_path = dump.join("expected.json");
    let bundle = default_bundle_dir();

    if !wav.exists() || !expected_path.exists() {
        eprintln!("skipping: no reference dump at {}", dump.display());
        return;
    }
    if !bundle.join("lm_cache_decode_int4.onnx").exists() {
        eprintln!("skipping: no model bundle at {}", bundle.display());
        return;
    }

    let expected: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&expected_path).expect("read expected.json"))
            .expect("parse expected.json");

    let audio = read_wav_mono16k(&wav).expect("read fixture wav");
    let mut engine = Engine::load(&bundle, 32).expect("load engine");
    let out = engine.transcribe(&audio, 128).expect("transcribe");

    let want_text = expected["text"].as_str().expect("expected.text");
    let want_audio_tokens = expected["audio_tokens"].as_u64().expect("audio_tokens") as usize;
    let want_generated = expected["generated"]
        .as_array()
        .expect("expected.generated")
        .len();

    eprintln!("rust:   {:?}", out.text);
    eprintln!("python: {want_text:?}");

    assert_eq!(
        out.audio_tokens, want_audio_tokens,
        "audio-token count drifted; the prompt and encoder would disagree"
    );
    assert_eq!(
        out.generated_tokens, want_generated,
        "generated a different number of tokens than the reference"
    );
    assert_eq!(out.text, want_text, "transcript differs from the reference");
}
