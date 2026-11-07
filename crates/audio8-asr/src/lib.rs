//! Audio8-ASR-0.1B speech recognition on the CPU via ONNX Runtime.
//!
//! See `reference/oracle.py` for the Python implementation this mirrors, and
//! the crate README for why the int4 LM / int8 encoder pair was chosen.

pub mod engine;
pub mod mel;

pub use engine::{default_bundle_dir, Engine, Transcription};
pub use mel::SAMPLE_RATE;

use anyhow::{anyhow, Result};
use ort::environment::Environment;

/// The process-wide ONNX Runtime environment.
///
/// ONNX Runtime allows exactly one environment per process, and a second
/// `ort::init()` fails with "only one environment is allowed per process".
/// Since both the ASR engine and the Silero VAD need sessions, they have to
/// share this one rather than each building their own.
pub fn ort_env() -> Result<&'static Environment> {
    static ENV: std::sync::OnceLock<std::result::Result<Environment, String>> =
        std::sync::OnceLock::new();
    ENV.get_or_init(|| {
        ort::init()
            .with_name("audio8")
            .build()
            .map(|env| {
                // ONNX Runtime logs every graph transform and node assignment
                // at INFO, which buries the service's own lines.
                env.set_log_level(ort::logging::LogLevel::Warning);
                env
            })
            .map_err(|e| e.to_string())
    })
    .as_ref()
    .map_err(|e| anyhow!("ort init: {e}"))
}
