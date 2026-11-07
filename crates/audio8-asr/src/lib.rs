//! Audio8-ASR-0.1B speech recognition on the CPU via ONNX Runtime.
//!
//! See `reference/oracle.py` for the Python implementation this mirrors, and
//! the crate README for why the int4 LM / int8 encoder pair was chosen.

pub mod engine;
pub mod mel;

pub use engine::{default_bundle_dir, Engine, Transcription};
pub use mel::SAMPLE_RATE;
