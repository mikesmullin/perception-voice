//! Toolchain probe: can `ort` load this bundle's graphs?
//!
//! The int4 LM graphs use MatMulNBits, which needs a recent ONNX Runtime --
//! if this fails, the int4 plan does not work from Rust and we fall back to
//! int8 (or reconsider the whole approach).
use std::path::PathBuf;

use ort::session::Session;

fn bundle_dir() -> PathBuf {
    std::env::var("AUDIO8_ASR_BUNDLE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").expect("HOME"))
                .join(".cache/audio8-asr-onnx/model_bundle")
        })
}

/// `SessionBuilder` methods return `Error<SessionBuilder>` (they hand the
/// builder back on failure), so they do not chain with `and_then`; `?` in a
/// helper converts them.
fn open(env: &ort::environment::Environment, path: &std::path::Path) -> anyhow::Result<Session> {
    // SessionBuilder errors carry the builder back, and that type is not
    // Send, so anyhow cannot absorb them directly -- flatten to a message.
    let b = Session::builder(env).map_err(|e| anyhow::anyhow!("builder: {e}"))?;
    let mut b = b
        .with_intra_threads(8)
        .map_err(|e| anyhow::anyhow!("with_intra_threads: {e}"))?;
    b.commit_from_file(path)
        .map_err(|e| anyhow::anyhow!("commit_from_file: {e}"))
}

fn main() -> anyhow::Result<()> {
    let env = ort::init()
        .with_name("audio8-probe")
        .build()
        .map_err(|e| anyhow::anyhow!("ort init: {e}"))?;
    let bundle = bundle_dir();
    println!("bundle: {}\n", bundle.display());

    for name in [
        "audio_hidden_int8.onnx",
        "lm_cache_prefill_int4.onnx",
        "lm_cache_decode_int4.onnx",
    ] {
        let path = bundle.join(name);
        println!("--- {name} ---");
        match open(&env, &path) {
            Ok(session) => {
                for i in session.inputs() {
                    println!("  in   {:24} {:?}", i.name(), i.dtype());
                }
                let outs = session.outputs();
                for o in outs.iter().take(3) {
                    println!("  out  {:24} {:?}", o.name(), o.dtype());
                }
                if outs.len() > 3 {
                    println!("  out  ... +{} more", outs.len() - 3);
                }
            }
            Err(e) => println!("  FAILED: {e}"),
        }
        println!();
    }
    Ok(())
}
