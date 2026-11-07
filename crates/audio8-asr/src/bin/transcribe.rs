//! Transcribe a 16 kHz mono WAV with the CPU ONNX pipeline.
//!
//!     transcribe <wav> [--runs N] [--threads N]
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Result};
use audio8_asr::engine::{default_bundle_dir, read_wav_mono16k};
use audio8_asr::{Engine, SAMPLE_RATE};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut wav: Option<PathBuf> = None;
    let mut runs = 1usize;
    let mut threads = 32usize;
    let mut max_new_tokens = 128usize;

    while let Some(a) = args.next() {
        match a.as_str() {
            "--runs" => runs = args.next().ok_or_else(|| anyhow!("--runs needs a value"))?.parse()?,
            "--threads" => {
                threads = args.next().ok_or_else(|| anyhow!("--threads needs a value"))?.parse()?
            }
            "--max-new-tokens" => {
                max_new_tokens =
                    args.next().ok_or_else(|| anyhow!("--max-new-tokens needs a value"))?.parse()?
            }
            other if wav.is_none() => wav = Some(PathBuf::from(other)),
            other => return Err(anyhow!("unexpected argument: {other}")),
        }
    }
    let wav = wav.ok_or_else(|| anyhow!("usage: transcribe <wav> [--runs N] [--threads N]"))?;

    let bundle = default_bundle_dir();
    let audio = read_wav_mono16k(&wav)?;
    let seconds = audio.len() as f64 / SAMPLE_RATE as f64;

    let t0 = Instant::now();
    let mut engine = Engine::load(&bundle, threads)?;
    let init = t0.elapsed();

    println!("bundle:  {}", bundle.display());
    println!("audio:   {} ({seconds:.2}s)", wav.display());
    println!("init:    {:.2}s  (threads={threads})", init.as_secs_f64());

    let mut times = Vec::new();
    let mut last = String::new();
    for _ in 0..runs {
        let t = Instant::now();
        let out = engine.transcribe(&audio, max_new_tokens)?;
        times.push(t.elapsed().as_secs_f64());
        last = out.text.clone();
        if times.len() == 1 {
            println!(
                "tokens:  {} audio, {} generated, prompt_len {}",
                out.audio_tokens, out.generated_tokens, out.prompt_len
            );
        }
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    println!("cold:    {:.3}s", times.iter().cloned().fold(f64::MIN, f64::max));
    println!("median:  {median:.3}s  ({:.1}x realtime)", seconds / median);
    println!("text:    {last:?}");
    Ok(())
}
