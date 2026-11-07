# perception-voice (Rust)

System-level speech-to-text service. Rust rewrite of the Python implementation
on `main`.

| branch | implementation |
|---|---|
| `main` | Python: faster-whisper / Audio8 on the GPU, PortAudio capture |
| `rust` | Rust: Audio8-ASR on the CPU via ONNX Runtime (this branch) |

## Why the CPU

Measured on the same 11.48s clip, AMD Threadripper 7970X / RTX 5090:

| backend | init | median decode | VRAM |
|---|---|---|---|
| Python, GPU transformers bf16 | 5.26s | 0.086s | 627 MiB |
| Python, CPU ONNX int4/int8 | 0.57s | 0.275s | 0 |
| **Rust, CPU ONNX int4/int8** | **0.66s** | **0.278s** | **0** |

All three produce byte-identical text. The GPU is ~3x faster per utterance,
but that is invisible: the VAD already waits 0.6s of trailing silence before
decoding starts, so 0.2s of decode sits under a latency floor we have already
accepted. What the CPU buys is an 8x faster cold start, a 4x faster *first*
utterance after a restart, and 627 MiB of VRAM back for the LLM sharing the
card.

Rust and Python land in the same place on steady-state latency, which is
expected — the arithmetic happens inside ONNX Runtime either way. Rust's win is
process-level: no interpreter, no torch/transformers import, one static binary.

## Layout

| crate | role |
|---|---|
| `audio8-asr` | the inference engine: log-mel, audio encoder, projector, int4 LM |
| `perception-voice` | the server: capture, VAD, buffer, IPC *(next)* |

## Model bundle

Not in git (644 MB). Expected at `~/.cache/audio8-asr-onnx/model_bundle`,
overridable with `AUDIO8_ASR_BUNDLE`. It is the int4/int8 subset of
`Audio8/Audio8-ASR-0.1B-onnx-runtime`. `lm_logits.onnx` is deliberately absent:
the cache graphs emit logits directly, so dropping it saves 414 MB.

## reference/oracle.py

A clean-room Python implementation of the pipeline, written from the bundle
metadata and ONNX graph signatures rather than from the upstream runtime. It
reproduces the upstream transcript exactly, which is what licenses this port —
the algorithm is understood, so no third-party inference code is vendored here.

`--dump DIR` writes per-stage tensors (mel, encoder hidden, valid mask, audio
embeddings, prompt embeddings, expected token ids). The Rust tests diff against
those, so a regression is localized to a stage instead of showing up as a
wrong sentence:

    python reference/oracle.py path/to/16k-mono.wav --dump reference/dump
    cargo test --release

## Try it

    cargo run --release --bin transcribe -- path/to/16k-mono.wav --runs 5
    cargo run --bin probe        # verify ONNX Runtime can load the graphs
