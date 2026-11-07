//! Numeric check of the Rust log-mel against `reference/oracle.py`.
//!
//! The audio encoder was traced against `transformers`' WhisperFeatureExtractor
//! output, so a drift here silently degrades transcription instead of failing
//! loudly. Comparing against a dumped reference catches it at the stage it
//! happens.
//!
//! Skipped (not failed) when the dump is absent, since it is gitignored:
//!
//!     python reference/oracle.py <wav> --dump reference/dump

use std::path::{Path, PathBuf};

use audio8_asr::mel::{MelExtractor, N_MELS};
use ndarray::Array3;
use ndarray_npy::read_npy;

fn dump_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR is crates/audio8-asr
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../reference/dump")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from("/nonexistent"))
}

fn read_wav_mono16k(path: &Path) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("open wav");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16_000, "fixture must be 16 kHz");
    assert_eq!(spec.bits_per_sample, 16, "fixture must be 16-bit");
    let samples: Vec<i16> = reader.samples::<i16>().map(|s| s.expect("sample")).collect();
    if spec.channels == 2 {
        samples
            .chunks(2)
            .map(|c| (c[0] as f32 + c[1] as f32) / 2.0 / 32768.0)
            .collect()
    } else {
        samples.iter().map(|&s| s as f32 / 32768.0).collect()
    }
}

#[test]
fn log_mel_matches_python_reference() {
    let dump = dump_dir();
    let wav = dump.join("input.wav");
    let mel_npy = dump.join("mel.npy");
    if !wav.exists() || !mel_npy.exists() {
        eprintln!("skipping: no reference dump at {}", dump.display());
        return;
    }

    // oracle dumps the padded feature as [1, 128, 3000]
    let reference: Array3<f32> = read_npy(&mel_npy).expect("read mel.npy");
    assert_eq!(reference.shape()[1], N_MELS);

    let audio = read_wav_mono16k(&wav);
    let mine = MelExtractor::new().log_mel(&audio);

    assert_eq!(mine.len(), N_MELS, "mel bin count");
    let n_frames = mine[0].len();
    assert!(
        n_frames <= reference.shape()[2],
        "rust produced {n_frames} frames, reference holds {}",
        reference.shape()[2]
    );

    let mut max_abs_diff = 0.0f32;
    let mut worst = (0usize, 0usize);
    for m in 0..N_MELS {
        for t in 0..n_frames {
            let d = (mine[m][t] - reference[[0, m, t]]).abs();
            if d > max_abs_diff {
                max_abs_diff = d;
                worst = (m, t);
            }
        }
    }

    eprintln!(
        "frames={n_frames} max_abs_diff={max_abs_diff:.3e} at mel={} frame={}",
        worst.0, worst.1
    );

    // f32 FFT against numpy's f64 path: small differences are expected, but
    // the features live in roughly [-1, 1] so anything above 1e-3 means a
    // real formulation mismatch (window, scaling, filterbank), not round-off.
    assert!(
        max_abs_diff < 1e-3,
        "log-mel diverges from the Python reference: max |diff| = {max_abs_diff:.6e} \
         at mel bin {}, frame {} -- check window, padding, or filterbank norm",
        worst.0,
        worst.1
    );
}
