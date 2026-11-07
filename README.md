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
| `perception-voice` | the server: supervised capture, dual VAD, buffer, IPC |

## Startup

    server listening on ./perception.sock          13ms
    audio stream started (16000 Hz, 2 ch)          13ms
    STT engine ready in 0.65s                     650ms

The socket and the microphone come up first; the model loads behind them on its
own thread. Audio captured during the load is still segmented and queued. The
Python build on `main` reached the same point in 1.76s, and before that fix,
8.2s.

## Capture is supervised, not opened once

A capture stream that fails to open, or opens and then stops delivering
callbacks, is retried forever with backoff. Both used to be terminal in the
Python build, which left systemd reporting a healthy unit with a dead
microphone. A gap of more than 3s in audio callbacks counts as a failure.

Note the limit of this: the watchdog recovers when the *node dies*. Merely
changing the PipeWire default source does not disturb an already-open ALSA
stream, so the new default is picked up on the next reopen, not immediately.

## Model bundle

Not in git (644 MB). Expected at `~/.cache/audio8-asr-onnx/model_bundle`,
overridable with `AUDIO8_ASR_BUNDLE`. It is the int4/int8 subset of
`Audio8/Audio8-ASR-0.1B-onnx-runtime`. `lm_logits.onnx` is deliberately absent:
the cache graphs emit logits directly, so dropping it saves 414 MB.

## Try it

    cargo run --release --bin transcribe -- path/to/16k-mono.wav --runs 5
    cargo run --bin probe        # verify ONNX Runtime can load the graphs
