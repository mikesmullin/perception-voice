#!/usr/bin/env python3
"""Independent reference implementation of the Audio8-ASR ONNX pipeline.

Written from the bundle metadata + graph signatures, NOT by importing the
vendor runtime. Its job is twofold:

  1. confirm the algorithm is understood correctly (it must reproduce the
     same transcript the vendor runtime produces), and
  2. dump intermediate tensors so the Rust port can be diffed stage by
     stage instead of only at the final string.

Usage:  oracle.py <wav> [--dump DIR]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort

BUNDLE = Path.home() / ".cache/audio8-asr-onnx/model_bundle"
SR = 16000


def load_wav_mono16k(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getframerate() == SR, f"expected {SR} Hz, got {w.getframerate()}"
        assert w.getsampwidth() == 2, "expected 16-bit PCM"
        raw = w.readframes(w.getnframes())
        a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if w.getnchannels() == 2:
            a = a.reshape(-1, 2).mean(axis=1)
    return a


def log_mel_128(audio: np.ndarray, n_fft=400, hop=160, n_mels=128) -> np.ndarray:
    """Whisper log-mel, matching transformers' WhisperFeatureExtractor."""
    from transformers.audio_utils import mel_filter_bank, spectrogram, window_function

    filters = mel_filter_bank(
        num_frequency_bins=1 + n_fft // 2,
        num_mel_filters=n_mels,
        min_frequency=0.0,
        max_frequency=SR / 2.0,
        sampling_rate=SR,
        norm="slaney",
        mel_scale="slaney",
    )
    spec = spectrogram(
        audio,
        window_function(n_fft, "hann"),
        frame_length=n_fft,
        hop_length=hop,
        power=2.0,
        mel_filters=filters,
        log_mel="log10",
        mel_floor=1e-10,
    ).astype(np.float32)
    # WhisperFeatureExtractor drops the trailing frame; match it so this
    # stays bit-comparable with the Rust port.
    return spec[:, :-1] if spec.shape[1] > 1 else spec


def ark_audio_token_count(sample_count: int, hop=160, merge=4) -> int:
    mel_frames = sample_count // hop
    return max(((mel_frames + 1) // 2) // merge, 1)


def layer_norm(x, w, b, eps=1e-5):
    x = x.astype(np.float32)
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return ((x - m) / np.sqrt(v + eps)) * w + b


def adaptive_avg_pool_time(x, out_n: int):
    n = x.shape[0]
    if n == out_n:
        return x.astype(np.float32)
    out = np.empty((out_n, x.shape[1]), np.float32)
    for i in range(out_n):
        s = math.floor(i * n / out_n)
        e = max(math.ceil((i + 1) * n / out_n), s + 1)
        out[i] = x[s:e].mean(axis=0)
    return out


SPECIAL = re.compile(r"<\|[^>]+\|>")
TURN_END = ("<|user|>", "<|assistant|>", "<|im_end|>")


def normalize_text(text: str) -> str:
    cut = len(text)
    for m in TURN_END:
        i = text.find(m)
        if i != -1 and i < cut:
            cut = i
    text = text[:cut].strip()
    text = SPECIAL.sub("", text).strip()
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"^[\s,.;:!?-]+", "", text).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", type=Path)
    ap.add_argument("--dump", type=Path, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    meta = json.loads((BUNDLE / "metadata.json").read_text())
    tok_meta = meta["tokens"]
    graph = meta["graphs"]["lm_cache_decode"]
    n_layers = graph["num_layers"]
    max_total = graph["max_total_len"]
    n_kv, head_dim = graph["num_key_value_heads"], graph["head_dim"]

    audio = load_wav_mono16k(args.wav)[: 30 * SR]
    sample_count = max(1, audio.shape[0])

    # --- features -------------------------------------------------------
    mel = log_mel_128(audio)                      # [128, frames]
    mel = np.maximum(mel, mel.max() - 8.0)
    mel = (mel + 4.0) / 4.0
    frames_padded = meta["graphs"]["audio_hidden_int8"]["frames_padded"]
    if mel.shape[-1] < frames_padded:
        mel = np.pad(mel, ((0, 0), (0, frames_padded - mel.shape[-1])))
    else:
        mel = mel[:, :frames_padded]
    feature = mel[None].astype(np.float32)        # [1,128,3000]
    enc_len = min(max(1, math.ceil(sample_count / 160)), frames_padded)

    so = ort.SessionOptions()
    so.log_severity_level = 3
    mk = lambda p: ort.InferenceSession(str(BUNDLE / p), sess_options=so,
                                        providers=["CPUExecutionProvider"])

    # --- audio encoder + projector --------------------------------------
    audio_sess = mk("audio_hidden_int8.onnx")
    hidden, valid = audio_sess.run(None, {
        "audios": feature,
        "audio_feature_lengths": np.asarray([enc_len], dtype=np.int64),
    })
    valid_hidden = hidden.astype(np.float32)[valid.astype(bool)]
    n_audio = ark_audio_token_count(sample_count)
    if valid_hidden.shape[0] != n_audio:
        valid_hidden = adaptive_avg_pool_time(valid_hidden, n_audio)

    proj = np.load(BUNDLE / meta["weights"]["audio_projector"])
    audio_embeds = layer_norm(valid_hidden, proj["norm_weight"], proj["norm_bias"])
    audio_embeds = audio_embeds @ proj["linear_weight"].T + proj["linear_bias"]
    audio_embeds = audio_embeds.astype(np.float32)

    # --- prompt ---------------------------------------------------------
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(BUNDLE / "tokenizer.json"))
    prompt = (
        f"{tok_meta['user_token']}{tok_meta['bos_audio_token']}"
        f"{tok_meta['audio_token'] * audio_embeds.shape[0]}"
        f"{tok_meta['eos_audio_token']}"
        "Please transcribe this audio."
        f"{tok_meta['assistant_token']}{meta.get('response_prefix','') or ''}"
    )
    input_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    token_embedding = np.load(BUNDLE / meta["weights"]["token_embedding"]).astype(np.float32)
    embeds = token_embedding[np.asarray(input_ids, dtype=np.int64)].copy()
    audio_pos = [i for i, t in enumerate(input_ids) if t == tok_meta["audio_token_id"]]
    assert len(audio_pos) == audio_embeds.shape[0], (len(audio_pos), audio_embeds.shape)
    embeds[np.asarray(audio_pos, dtype=np.int64)] = audio_embeds

    # --- prefill --------------------------------------------------------
    prefill = mk("lm_cache_prefill_int4.onnx")
    decode = mk("lm_cache_decode_int4.onnx")
    caches = [np.zeros((1, n_kv, max_total, head_dim), np.float32)
              for _ in range(2 * n_layers)]
    prompt_len = embeds.shape[0]
    outs = prefill.run(None, {
        "inputs_embeds": embeds[None].astype(np.float32),
        "cache_position": np.arange(prompt_len, dtype=np.int64),
    })
    logits = outs[0][0, -1, :].astype(np.float32)
    for i in range(n_layers):
        caches[2 * i][:, :, :prompt_len, :] = outs[1 + 2 * i]
        caches[2 * i + 1][:, :, :prompt_len, :] = outs[2 + 2 * i]

    # --- greedy decode --------------------------------------------------
    blocked = set(int(v) for v in tok_meta.get("extra_block_token_ids", []))
    eos = set(int(v) for v in tok_meta["eos_token_ids"]) | {int(tok_meta["pad_token_id"])}
    generated: list[int] = []
    pos = prompt_len
    for _ in range(args.max_new_tokens):
        step = logits.copy()
        for b in blocked:
            if 0 <= b < step.shape[0]:
                step[b] = -np.inf
        nxt = int(np.argmax(step))
        if nxt in eos:
            break
        generated.append(nxt)
        mask = np.zeros((1, max_total), np.int64)
        mask[:, : pos + 1] = 1
        feeds = {
            "inputs_embeds": token_embedding[nxt].reshape(1, 1, -1).astype(np.float32),
            "attention_mask": mask,
            "cache_position": np.asarray([pos], dtype=np.int64),
        }
        for i in range(n_layers):
            feeds[f"cache_key_{i}"] = caches[2 * i]
            feeds[f"cache_value_{i}"] = caches[2 * i + 1]
        o = decode.run(None, feeds)
        logits = o[0][0, -1, :].astype(np.float32)
        for i in range(n_layers):
            caches[2 * i][:, :, pos:pos + 1, :] = o[1 + 2 * i]
            caches[2 * i + 1][:, :, pos:pos + 1, :] = o[2 + 2 * i]
        pos += 1

    raw = tokenizer.decode(generated, skip_special_tokens=False)
    text = normalize_text(raw)

    print(f"audio_seconds:     {sample_count / SR:.2f}")
    print(f"audio_tokens:      {audio_embeds.shape[0]}")
    print(f"prompt_len:        {prompt_len}")
    print(f"generated_tokens:  {len(generated)}")
    print(f"raw:               {raw!r}")
    print(f"text:              {text!r}")

    if args.dump:
        args.dump.mkdir(parents=True, exist_ok=True)
        np.save(args.dump / "mel.npy", feature)
        np.save(args.dump / "audio_hidden.npy", hidden)
        np.save(args.dump / "audio_valid_mask.npy", valid)
        np.save(args.dump / "audio_embeds.npy", audio_embeds)
        np.save(args.dump / "prompt_embeds.npy", embeds)
        (args.dump / "expected.json").write_text(json.dumps({
            "input_ids": input_ids, "generated": generated,
            "raw": raw, "text": text,
            "audio_tokens": int(audio_embeds.shape[0]),
            "enc_len": int(enc_len), "sample_count": int(sample_count),
        }, indent=1))
        print(f"\ndumped reference tensors -> {args.dump}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
