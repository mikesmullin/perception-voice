//! Audio8-ASR-0.1B inference: CPU ONNX Runtime, int8 audio encoder + int4 LM.
//!
//! Mirrors `reference/oracle.py`, which reproduces the upstream transcript
//! exactly. The pipeline is:
//!
//! 1. 128-bin Whisper log-mel, padded to the encoder's fixed 3000 frames
//! 2. audio encoder -> `[390, 1024]` hidden states + a validity mask
//! 3. gather valid rows, pool them to the prompt's audio-token count
//! 4. projector: layer-norm, then linear into the LM's 512-wide space
//! 5. build the chat prompt with one `<|audio|>` per audio token and splice
//!    the projected audio embeddings over those positions
//! 6. int4 prefill for the whole prompt, then greedy decode one token at a
//!    time against an explicit KV cache
//!
//! `run()` needs `&mut Session`, so transcription is inherently serialized;
//! the server wraps this in a mutex, as the Python did.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use ndarray::{Array2, Array3};
use ndarray_npy::NpzReader;
use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::mel::{MelExtractor, HOP_LENGTH, N_MELS, SAMPLE_RATE};

/// The encoder graph has these baked in as fixed dimensions.
const FRAMES_PADDED: usize = 3000;
const MAX_AUDIO_TOKENS: usize = 390;
const ENCODER_HIDDEN: usize = 1024;

/// LM geometry, from the bundle's `graphs.lm_cache_decode`.
const LM_HIDDEN: usize = 512;
const NUM_LAYERS: usize = 8;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 64;
const MAX_TOTAL_LEN: usize = 512;

const MERGE_FACTOR: usize = 4;
const MAX_AUDIO_SECONDS: usize = 30;

/// Special token ids, from `metadata.json`'s `tokens` block.
const AUDIO_TOKEN_ID: u32 = 151646;
const PAD_TOKEN_ID: u32 = 151643;
const EOS_TOKEN_IDS: &[u32] = &[151645];
const BLOCKED_TOKEN_IDS: &[u32] = &[151647, 151650, 151648, 151646, 151649];

const USER_TOKEN: &str = "<|user|>";
const ASSISTANT_TOKEN: &str = "<|assistant|>";
const BOS_AUDIO_TOKEN: &str = "<|begin_of_audio|>";
const AUDIO_TOKEN: &str = "<|audio|>";
const EOS_AUDIO_TOKEN: &str = "<|end_of_audio|>";
const INSTRUCTION: &str = "Please transcribe this audio.";

pub fn default_bundle_dir() -> PathBuf {
    std::env::var("AUDIO8_ASR_BUNDLE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()))
                .join(".cache/audio8-asr-onnx/model_bundle")
        })
}

struct Projector {
    norm_weight: Vec<f32>,
    norm_bias: Vec<f32>,
    /// Stored pre-transposed as `[ENCODER_HIDDEN, LM_HIDDEN]` so the hot path
    /// is a plain `normed.dot(&w)` with no transpose per utterance.
    linear_weight: Array2<f32>,
    linear_bias: Vec<f32>,
}

pub struct Engine {
    // Sessions borrow from the environment, so it must outlive them; keeping
    // it in the struct (declared last for drop order) guarantees that.
    audio: Session,
    prefill: Session,
    decode: Session,
    tokenizer: Tokenizer,
    /// `[vocab][LM_HIDDEN]`, row-major. ~311 MB, loaded once.
    token_embedding: Vec<f32>,
    vocab: usize,
    projector: Projector,
    mel: MelExtractor,
    _env: Environment,
}

pub struct Transcription {
    pub text: String,
    pub raw: String,
    pub generated_tokens: usize,
    pub audio_tokens: usize,
    pub prompt_len: usize,
}

impl Engine {
    pub fn load(bundle: &Path, threads: usize) -> Result<Self> {
        let env = ort::init()
            .with_name("audio8-asr")
            .build()
            .map_err(|e| anyhow!("ort init: {e}"))?;

        let open = |name: &str| -> Result<Session> {
            let path = bundle.join(name);
            let b = Session::builder(&env).map_err(|e| anyhow!("session builder: {e}"))?;
            let mut b = b
                .with_intra_threads(threads)
                .map_err(|e| anyhow!("with_intra_threads: {e}"))?;
            b.commit_from_file(&path)
                .map_err(|e| anyhow!("loading {}: {e}", path.display()))
        };

        let audio = open("audio_hidden_int8.onnx")?;
        let prefill = open("lm_cache_prefill_int4.onnx")?;
        let decode = open("lm_cache_decode_int4.onnx")?;

        let tokenizer = Tokenizer::from_file(bundle.join("tokenizer.json"))
            .map_err(|e| anyhow!("loading tokenizer.json: {e}"))?;

        let te_path = bundle.join("weights/token_embedding.npy");
        let te: Array2<f32> = ndarray_npy::read_npy(&te_path)
            .with_context(|| format!("reading {}", te_path.display()))?;
        if te.shape()[1] != LM_HIDDEN {
            return Err(anyhow!(
                "token_embedding is {:?}, expected [_, {LM_HIDDEN}]",
                te.shape()
            ));
        }
        let vocab = te.shape()[0];
        let token_embedding = te.into_raw_vec_and_offset().0;

        let projector = load_projector(&bundle.join("weights/audio_projector.npz"))?;

        Ok(Self {
            audio,
            prefill,
            decode,
            tokenizer,
            token_embedding,
            vocab,
            projector,
            mel: MelExtractor::new(),
            _env: env,
        })
    }

    /// Transcribe mono f32 PCM at 16 kHz. Audio beyond 30s is truncated, as
    /// that is the encoder's hard limit.
    pub fn transcribe(&mut self, audio: &[f32], max_new_tokens: usize) -> Result<Transcription> {
        let max_samples = MAX_AUDIO_SECONDS * SAMPLE_RATE;
        let audio = if audio.len() > max_samples {
            &audio[..max_samples]
        } else {
            audio
        };
        let sample_count = audio.len().max(1);

        let feature = self.features(audio);
        let encoder_len = sample_count.div_ceil(HOP_LENGTH).clamp(1, FRAMES_PADDED);
        let audio_embeds = self.audio_embeddings(&feature, sample_count, encoder_len)?;
        let audio_tokens = audio_embeds.shape()[0];

        let (input_ids, embeds) = self.initial_embeddings(&audio_embeds)?;
        let prompt_len = input_ids.len();

        if prompt_len + max_new_tokens > MAX_TOTAL_LEN {
            return Err(anyhow!(
                "prompt ({prompt_len}) + max_new_tokens ({max_new_tokens}) exceeds the LM's \
                 KV cache limit of {MAX_TOTAL_LEN}; feed a shorter utterance"
            ));
        }

        let generated = self.generate(&embeds, prompt_len, max_new_tokens)?;
        let raw = self
            .tokenizer
            .decode(&generated, false)
            .map_err(|e| anyhow!("decoding tokens: {e}"))?;

        Ok(Transcription {
            text: normalize_text(&raw),
            raw,
            generated_tokens: generated.len(),
            audio_tokens,
            prompt_len,
        })
    }

    /// Log-mel, zero-padded (or truncated) to the encoder's fixed frame count.
    fn features(&self, audio: &[f32]) -> Vec<f32> {
        let mel = self.mel.log_mel(audio);
        let mut out = vec![0.0f32; N_MELS * FRAMES_PADDED];
        for (m, row) in mel.iter().enumerate() {
            let n = row.len().min(FRAMES_PADDED);
            out[m * FRAMES_PADDED..m * FRAMES_PADDED + n].copy_from_slice(&row[..n]);
        }
        out
    }

    fn audio_embeddings(
        &mut self,
        feature: &[f32],
        sample_count: usize,
        encoder_len: usize,
    ) -> Result<Array2<f32>> {
        let audios = Tensor::from_array((
            vec![1_i64, N_MELS as i64, FRAMES_PADDED as i64],
            feature.to_vec(),
        ))
        .map_err(|e| anyhow!("building audios tensor: {e}"))?;
        let lengths = Tensor::from_array((vec![1_i64], vec![encoder_len as i64]))
            .map_err(|e| anyhow!("building audio_feature_lengths tensor: {e}"))?;

        let outputs = self
            .audio
            .run(ort::inputs![
                "audios" => audios,
                "audio_feature_lengths" => lengths,
            ])
            .map_err(|e| anyhow!("audio encoder: {e}"))?;

        let (_, hidden) = outputs["audio_hidden"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("extracting audio_hidden: {e}"))?;
        let (_, mask) = outputs["audio_valid_mask"]
            .try_extract_tensor::<i64>()
            .map_err(|e| anyhow!("extracting audio_valid_mask: {e}"))?;

        // Gather the rows the encoder marked valid.
        let mut valid: Vec<f32> = Vec::with_capacity(MAX_AUDIO_TOKENS * ENCODER_HIDDEN);
        let mut n_valid = 0usize;
        for (i, &m) in mask.iter().enumerate().take(MAX_AUDIO_TOKENS) {
            if m != 0 {
                valid.extend_from_slice(&hidden[i * ENCODER_HIDDEN..(i + 1) * ENCODER_HIDDEN]);
                n_valid += 1;
            }
        }
        if n_valid == 0 {
            return Err(anyhow!("audio encoder returned no valid frames"));
        }

        // The prompt's audio-token count is derived from the sample count, not
        // from the mask, so the two must be reconciled by pooling over time.
        let want = ark_audio_token_count(sample_count);
        let pooled = if n_valid == want {
            valid
        } else {
            adaptive_avg_pool_time(&valid, n_valid, ENCODER_HIDDEN, want)
        };

        // layer-norm over the encoder-hidden axis...
        let p = &self.projector;
        let mut normed = Array2::<f32>::zeros((want, ENCODER_HIDDEN));
        for t in 0..want {
            let row = &pooled[t * ENCODER_HIDDEN..(t + 1) * ENCODER_HIDDEN];
            let mean = row.iter().sum::<f32>() / ENCODER_HIDDEN as f32;
            let var =
                row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / ENCODER_HIDDEN as f32;
            let inv = 1.0 / (var + 1e-5).sqrt();
            for (i, &v) in row.iter().enumerate() {
                normed[[t, i]] = (v - mean) * inv * p.norm_weight[i] + p.norm_bias[i];
            }
        }

        // ...then the linear projection. This is 143x1024x512 ~ 75M MACs; a
        // hand-rolled triple loop measured ~50ms, which was the whole gap
        // against numpy's BLAS. ndarray's `dot` uses a SIMD-blocked kernel.
        let mut out = normed.dot(&p.linear_weight);
        for mut row in out.rows_mut() {
            for (o, v) in row.iter_mut().enumerate() {
                *v += p.linear_bias[o];
            }
        }
        Ok(out)
    }

    /// Tokenize the prompt and splice audio embeddings over the audio slots.
    fn initial_embeddings(&self, audio_embeds: &Array2<f32>) -> Result<(Vec<u32>, Array2<f32>)> {
        let n_audio = audio_embeds.shape()[0];
        let prompt = format!(
            "{USER_TOKEN}{BOS_AUDIO_TOKEN}{}{EOS_AUDIO_TOKEN}{INSTRUCTION}{ASSISTANT_TOKEN}",
            AUDIO_TOKEN.repeat(n_audio)
        );
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("encoding prompt: {e}"))?;
        let input_ids = encoding.get_ids().to_vec();

        let mut embeds = Array2::<f32>::zeros((input_ids.len(), LM_HIDDEN));
        let mut audio_slot = 0usize;
        for (i, &id) in input_ids.iter().enumerate() {
            if id == AUDIO_TOKEN_ID {
                for d in 0..LM_HIDDEN {
                    embeds[[i, d]] = audio_embeds[[audio_slot, d]];
                }
                audio_slot += 1;
            } else {
                let base = id as usize * LM_HIDDEN;
                if base + LM_HIDDEN > self.token_embedding.len() {
                    return Err(anyhow!("token id {id} out of range (vocab {})", self.vocab));
                }
                for d in 0..LM_HIDDEN {
                    embeds[[i, d]] = self.token_embedding[base + d];
                }
            }
        }
        if audio_slot != n_audio {
            return Err(anyhow!(
                "prompt has {audio_slot} audio tokens but the encoder produced {n_audio}"
            ));
        }
        Ok((input_ids, embeds))
    }

    fn generate(
        &mut self,
        embeds: &Array2<f32>,
        prompt_len: usize,
        max_new_tokens: usize,
    ) -> Result<Vec<u32>> {
        // Two cache tensors per layer, each [1, kv_heads, MAX_TOTAL_LEN, head_dim].
        const CACHE_STRIDE: usize = NUM_KV_HEADS * MAX_TOTAL_LEN * HEAD_DIM;
        let mut caches: Vec<Vec<f32>> = (0..2 * NUM_LAYERS)
            .map(|_| vec![0.0f32; CACHE_STRIDE])
            .collect();

        let mut logits = self.run_prefill(embeds, prompt_len, &mut caches)?;

        let mut generated: Vec<u32> = Vec::new();
        let mut position = prompt_len;

        for _ in 0..max_new_tokens {
            for &b in BLOCKED_TOKEN_IDS {
                if (b as usize) < logits.len() {
                    logits[b as usize] = f32::NEG_INFINITY;
                }
            }
            let next = argmax(&logits) as u32;
            if EOS_TOKEN_IDS.contains(&next) || next == PAD_TOKEN_ID {
                break;
            }
            generated.push(next);
            if position + 1 >= MAX_TOTAL_LEN {
                break;
            }
            logits = self.run_decode_step(next, &mut caches, position)?;
            position += 1;
        }
        Ok(generated)
    }

    fn run_prefill(
        &mut self,
        embeds: &Array2<f32>,
        prompt_len: usize,
        caches: &mut [Vec<f32>],
    ) -> Result<Vec<f32>> {
        let flat = embeds.as_slice().expect("contiguous").to_vec();
        let inputs_embeds =
            Tensor::from_array((vec![1_i64, prompt_len as i64, LM_HIDDEN as i64], flat))
                .map_err(|e| anyhow!("building inputs_embeds: {e}"))?;
        let positions: Vec<i64> = (0..prompt_len as i64).collect();
        let cache_position = Tensor::from_array((vec![prompt_len as i64], positions))
            .map_err(|e| anyhow!("building cache_position: {e}"))?;

        let outputs = self
            .prefill
            .run(ort::inputs![
                "inputs_embeds" => inputs_embeds,
                "cache_position" => cache_position,
            ])
            .map_err(|e| anyhow!("lm prefill: {e}"))?;

        let (shape, logits) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("extracting prefill logits: {e}"))?;
        let vocab = *shape.last().ok_or_else(|| anyhow!("empty logits shape"))? as usize;
        let last = logits.len() - vocab;
        let out = logits[last..].to_vec();

        // Seed the cache with the prompt's keys/values.
        for layer in 0..NUM_LAYERS {
            for (half, kind) in ["key", "value"].iter().enumerate() {
                let name = format!("{kind}_delta_{layer}");
                let (_, delta) = outputs[name.as_str()]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| anyhow!("extracting {name}: {e}"))?;
                let cache = &mut caches[2 * layer + half];
                for h in 0..NUM_KV_HEADS {
                    for t in 0..prompt_len {
                        let src = (h * prompt_len + t) * HEAD_DIM;
                        let dst = (h * MAX_TOTAL_LEN + t) * HEAD_DIM;
                        cache[dst..dst + HEAD_DIM].copy_from_slice(&delta[src..src + HEAD_DIM]);
                    }
                }
            }
        }
        Ok(out)
    }

    fn run_decode_step(
        &mut self,
        token: u32,
        caches: &mut [Vec<f32>],
        position: usize,
    ) -> Result<Vec<f32>> {
        let base = token as usize * LM_HIDDEN;
        let embed = self.token_embedding[base..base + LM_HIDDEN].to_vec();
        let inputs_embeds = Tensor::from_array((vec![1_i64, 1, LM_HIDDEN as i64], embed))
            .map_err(|e| anyhow!("building inputs_embeds: {e}"))?;

        let mut mask = vec![0_i64; MAX_TOTAL_LEN];
        for m in mask.iter_mut().take(position + 1) {
            *m = 1;
        }
        let attention_mask = Tensor::from_array((vec![1_i64, MAX_TOTAL_LEN as i64], mask))
            .map_err(|e| anyhow!("building attention_mask: {e}"))?;
        let cache_position = Tensor::from_array((vec![1_i64], vec![position as i64]))
            .map_err(|e| anyhow!("building cache_position: {e}"))?;

        let mut feeds = ort::inputs![
            "inputs_embeds" => inputs_embeds,
            "attention_mask" => attention_mask,
            "cache_position" => cache_position,
        ];
        for layer in 0..NUM_LAYERS {
            for (half, kind) in ["key", "value"].iter().enumerate() {
                let t = Tensor::from_array((
                    vec![1_i64, NUM_KV_HEADS as i64, MAX_TOTAL_LEN as i64, HEAD_DIM as i64],
                    caches[2 * layer + half].clone(),
                ))
                .map_err(|e| anyhow!("building cache_{kind}_{layer}: {e}"))?;
                feeds.push((format!("cache_{kind}_{layer}").into(), t.into()));
            }
        }

        let outputs = self
            .decode
            .run(feeds)
            .map_err(|e| anyhow!("lm decode step: {e}"))?;

        let (shape, logits) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("extracting decode logits: {e}"))?;
        let vocab = *shape.last().ok_or_else(|| anyhow!("empty logits shape"))? as usize;
        let out = logits[logits.len() - vocab..].to_vec();

        for layer in 0..NUM_LAYERS {
            for (half, kind) in ["key", "value"].iter().enumerate() {
                let name = format!("{kind}_delta_{layer}");
                let (_, delta) = outputs[name.as_str()]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| anyhow!("extracting {name}: {e}"))?;
                let cache = &mut caches[2 * layer + half];
                for h in 0..NUM_KV_HEADS {
                    let src = h * HEAD_DIM;
                    let dst = (h * MAX_TOTAL_LEN + position) * HEAD_DIM;
                    cache[dst..dst + HEAD_DIM].copy_from_slice(&delta[src..src + HEAD_DIM]);
                }
            }
        }
        Ok(out)
    }
}

fn load_projector(path: &Path) -> Result<Projector> {
    let file =
        std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut npz = NpzReader::new(file).with_context(|| format!("reading {}", path.display()))?;

    let norm_weight: ndarray::Array1<f32> = npz.by_name("norm_weight.npy")?;
    let norm_bias: ndarray::Array1<f32> = npz.by_name("norm_bias.npy")?;
    let linear_weight: Array2<f32> = npz.by_name("linear_weight.npy")?;
    let linear_bias: ndarray::Array1<f32> = npz.by_name("linear_bias.npy")?;

    if linear_weight.shape() != [LM_HIDDEN, ENCODER_HIDDEN] {
        return Err(anyhow!(
            "linear_weight is {:?}, expected [{LM_HIDDEN}, {ENCODER_HIDDEN}]",
            linear_weight.shape()
        ));
    }
    Ok(Projector {
        norm_weight: norm_weight.to_vec(),
        norm_bias: norm_bias.to_vec(),
        // npz holds [LM_HIDDEN, ENCODER_HIDDEN]; transpose once at load.
        linear_weight: linear_weight.reversed_axes().as_standard_layout().to_owned(),
        linear_bias: linear_bias.to_vec(),
    })
}

/// How many `<|audio|>` slots the prompt gets for a given sample count.
/// Mirrors the upstream processor's rule exactly, including the integer
/// division order — the prompt and the encoder must agree or nothing lines up.
pub fn ark_audio_token_count(sample_count: usize) -> usize {
    let mel_frames = sample_count / HOP_LENGTH;
    let downsampled = (mel_frames + 1) / 2;
    (downsampled / MERGE_FACTOR).max(1)
}

/// Average-pool `rows` (`n` x `width`) down or up to `out_n` rows, matching
/// torch's `adaptive_avg_pool1d` bucket boundaries.
fn adaptive_avg_pool_time(rows: &[f32], n: usize, width: usize, out_n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_n * width];
    for i in 0..out_n {
        let start = i * n / out_n;
        let end = (((i + 1) * n) as f64 / out_n as f64).ceil() as usize;
        let end = end.max(start + 1).min(n);
        let count = (end - start) as f32;
        for t in start..end {
            for d in 0..width {
                out[i * width + d] += rows[t * width + d];
            }
        }
        for d in 0..width {
            out[i * width + d] /= count;
        }
    }
    out
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

/// Strip the model's turn structure and special tokens down to plain text.
pub fn normalize_text(raw: &str) -> String {
    let mut cut = raw.len();
    for marker in [USER_TOKEN, ASSISTANT_TOKEN, "<|im_end|>"] {
        if let Some(i) = raw.find(marker) {
            cut = cut.min(i);
        }
    }
    let text = &raw[..cut];

    // Any <|...|> run is a special token; drop it.
    let mut stripped = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < text.len() {
        if bytes[i..].starts_with(b"<|") {
            if let Some(end) = text[i..].find("|>") {
                i += end + 2;
                continue;
            }
        }
        let ch = text[i..].chars().next().unwrap();
        stripped.push(ch);
        i += ch.len_utf8();
    }

    // collapse whitespace, then trim leading punctuation noise
    let collapsed = stripped.split_whitespace().collect::<Vec<_>>().join(" ");
    collapsed
        .trim_start_matches([' ', ',', '.', ';', ':', '!', '?', '-'])
        .trim()
        .to_string()
}

/// 16 kHz mono f32 from a 16-bit WAV. Convenience for CLIs and tests.
pub fn read_wav_mono16k(path: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let spec = reader.spec();
    if spec.sample_rate as usize != SAMPLE_RATE {
        return Err(anyhow!(
            "{} is {} Hz; this pipeline needs {SAMPLE_RATE} Hz",
            path.display(),
            spec.sample_rate
        ));
    }
    if spec.bits_per_sample != 16 {
        return Err(anyhow!("{} is not 16-bit PCM", path.display()));
    }
    let samples: Vec<i16> = reader
        .samples::<i16>()
        .collect::<std::result::Result<_, _>>()
        .with_context(|| format!("reading samples from {}", path.display()))?;
    Ok(if spec.channels == 2 {
        samples
            .chunks(2)
            .map(|c| (c[0] as f32 + c[1] as f32) / 2.0 / 32768.0)
            .collect()
    } else {
        samples.iter().map(|&s| s as f32 / 32768.0).collect()
    })
}

/// A 3-D view helper used by tests that compare against dumped tensors.
pub fn feature_as_array(feature: &[f32]) -> Array3<f32> {
    Array3::from_shape_vec((1, N_MELS, FRAMES_PADDED), feature.to_vec())
        .expect("feature has the encoder's fixed shape")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_token_count_matches_upstream_rule() {
        // 11.479625s at 16 kHz -> 143 tokens, as reference/oracle.py reports.
        assert_eq!(ark_audio_token_count(183_674), 143);
        // never zero, however short the clip
        assert_eq!(ark_audio_token_count(1), 1);
    }

    #[test]
    fn pooling_is_identity_at_equal_length() {
        let rows = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(adaptive_avg_pool_time(&rows, 2, 2, 2), rows);
    }

    #[test]
    fn pooling_averages_adjacent_buckets() {
        // 4 rows of width 1 -> 2 rows: means of [1,2] and [3,4]
        let rows = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(adaptive_avg_pool_time(&rows, 4, 1, 2), vec![1.5, 3.5]);
    }

    #[test]
    fn normalize_strips_special_tokens_and_turn_tails() {
        assert_eq!(
            normalize_text("Hello world.<|im_end|><|user|>next turn"),
            "Hello world."
        );
        assert_eq!(normalize_text("<|text|>  spaced   out "), "spaced out");
        assert_eq!(normalize_text(", leading noise"), "leading noise");
    }
}
