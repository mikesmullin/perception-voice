# perception-voice

A system-level speech-to-text service that runs continuously as a systemd unit, providing shared voice transcription to multiple client processes via Unix socket IPC.

## Problem

Only one process can hold the microphone at a time, and each application that wants voice input otherwise loads its own copy of a speech model. Neither scales: the second app to start finds the mic busy, and every app pays the model's memory and startup cost again.

## Solution

**perception-voice** runs a single daemon that:
- Loads the speech model once, on the CPU
- Holds the microphone and continuously transcribes speech
- Stores utterances in a time-indexed memory buffer
- Allows clients to retrieve transcriptions via Unix socket (poll with
  `get`/`set`, or **subscribe to real-time push streams** — see below)
- Publishes **~60 Hz audio feature frames** (rms + 4 spectral bands +
  VAD) for driving visualizations, like a voice-reactive avatar

## Requirements

- Rust toolchain (edition 2021)
- Microphone
- Linux with PipeWire or PulseAudio
- **No GPU.** Inference is CPU-only; the GPU is left entirely free for
  other workloads.

## STT Backend

[Audio8-ASR-0.1B](https://huggingface.co/Audio8/Audio8-ASR-0.1B) via ONNX Runtime on the CPU — an int4-quantized language model paired with an int8 audio encoder.

| | value |
|---|---|
| GPU VRAM | **zero** |
| disk | 647 MB |
| engine init | 0.66s |
| decode | 0.28s median for an 11.5s utterance (41x realtime) |
| languages | en, zh, yue, fr, de, ja, ko |
| licence | CC-BY-NC-4.0 (personal use) |

Quantization costs nothing measurable here: int4/int8 produces text identical to the same model in bfloat16 on the GPU, which is ~3x faster per utterance but needs 627 MB of VRAM and 5.3s to initialize.

Per-utterance decode latency is not what you feel anyway. The VAD waits `audio.post_speech_silence_duration` of trailing silence before it even starts decoding, so a few hundred milliseconds of inference sits underneath a delay already built into the pipeline. What the CPU buys is a sub-second cold start and the whole GPU.

Utterances longer than one pass of the model's 512-slot KV cache (about 30s) are split at the quietest point near the boundary and rejoined, so long dictation is not truncated.

## Installation

```bash
cargo build --release
```

### Model bundle

Not included in the repo. Fetch the int4/int8 subset of the ONNX export into the cache directory:

```bash
hf download Audio8/Audio8-ASR-0.1B-onnx-runtime \
  --local-dir ~/.cache/audio8-asr-onnx \
  --include 'model_bundle/audio_hidden_int8.onnx' \
            'model_bundle/lm_cache_prefill_int4.onnx*' \
            'model_bundle/lm_cache_decode_int4.onnx*' \
            'model_bundle/*.json' 'model_bundle/tokenizer.json' \
            'model_bundle/merges.txt' 'model_bundle/vocab.json' \
            'model_bundle/weights/*' \
            'model_bundle/qwen3_asr_feature_extractor/*'
```

`lm_logits.onnx` is deliberately **not** fetched: the cache graphs emit logits directly, so it would be 414 MB of dead weight.

Override the location with `audio8.bundle_dir` in the config, or `$AUDIO8_ASR_BUNDLE`.

### Silero VAD model

```bash
hf download snakers4/silero-vad --local-dir /tmp/silero \
  --include 'src/silero_vad/data/silero_vad.onnx'
mkdir -p ~/.cache/audio8-asr-onnx/vad
cp /tmp/silero/src/silero_vad/data/silero_vad.onnx ~/.cache/audio8-asr-onnx/vad/
```

Override with `vad.silero_model_path`. Without it the service still runs, on WebRTC VAD alone — noticeably more trigger-happy on noise.

## Quick Start

### 1. Configure (optional)

Copy the example config and customize:

```bash
cp config.example.yml config.yml
```

### 2. Start the server

```bash
./target/release/perception-voice serve
```

The server will:
- Bind a Unix socket at `perception.sock` and open the microphone
  immediately (~15 ms)
- Load the model on a background thread (~0.65s); audio captured while it
  loads is still buffered and transcribed

### 3. Use from client processes

```bash
# Set read marker (call when your app starts listening)
perception-voice client set myapp

# ... user speaks ...

# Get transcriptions since marker (returns JSONL)
perception-voice client get myapp
```

## CLI Reference

### Server Mode

```bash
perception-voice [-c CONFIG] serve [-v]
```

| Option | Description |
|--------|-------------|
| `-c, --config FILE` | Config file (default: `config.yml` in the working directory) |
| `-v, --verbose` | Enable debug-level logging (metadata only) |

`RUST_LOG` overrides `-v` when set.

### Client Mode

```bash
perception-voice client set <uid>       # Set read marker to now
perception-voice client get <uid>       # Get transcriptions since marker
perception-voice client stream words    # Follow the words push stream (debug)
perception-voice client stream levels   # Follow the levels push stream (debug)
```

### Diagnostics

```bash
cargo run --release --bin transcribe -- path/to/16k-mono.wav --runs 5
cargo run --bin probe        # verify ONNX Runtime can load the model graphs
```

## Real-time push streams (`subscribe`)

Polling with `get`/`set` adds latency; latency-sensitive consumers (e.g.
the [Ada](https://github.com/mikesmullin/ada) voice assistant) can turn a
connection into a push stream instead. Send one framed JSON request
(4-byte big-endian length prefix + JSON, same framing as `get`/`set`):

```json
{"command": "subscribe", "channel": "words"}
```

After an `{"status":"ok"}` ack the server pushes to you forever (one
channel per connection):

- **`words`** — framed JSON events the moment they exist:
  - `{"ev":"utterance", "ts":…, "text":"Turn on the desk light.",
    "t_start":…, "t_end":…}` — the finalized transcription, with
    unix-epoch start/end times of the speech itself
- **`levels`** — raw binary feature frames at 62.5 Hz for visualization:
  36 bytes little-endian — magic `'AVF1'` (u32), stream_id (u32, 0=mic),
  rms (f32), 4 spectral band energies (f32, 0–300/300–1k/1k–3k/3k–8k Hz),
  pitch_hint (f32, unused), flags (u32: bit0 VAD, bit1 utterance-start,
  bit2 utterance-end)

There are no `{"ev":"partial"}` events. The secondary model that produced live hypotheses was the last thing requiring the GPU, and went with it.

Fan-out happens on a dedicated thread with a bounded queue — a slow or
stuck subscriber gets dropped, never stalls capture or other clients. The
full wire spec lives in Ada's repo:
[docs/PROTOCOL.md](https://github.com/mikesmullin/ada/blob/main/docs/PROTOCOL.md).

## Output Format (JSONL)

The `get` command returns JSONL with ISO 8601 timestamps:

```jsonl
{"ts": "2026-01-20T14:32:05.123-08:00", "text": "turn on lights"}
{"ts": "2026-01-20T14:32:12.456-08:00", "text": "hello world"}
```

A client that has never called `set` gets nothing, rather than the whole retained buffer — otherwise the first `get` after a restart would replay half an hour of backlog.

## Configuration

See `config.example.yml` for all options. Key settings:

```yaml
server:
  socket_path: "perception.sock"
  buffer_retention_minutes: 30
  transcript_log_path: null  # set to a path to keep an append-only timestamped log

audio:
  # Trailing silence that ends an utterance. This, not decode time,
  # dominates how responsive dictation feels.
  post_speech_silence_duration: 0.3
  # Utterances shorter than this are discarded as noise.
  min_utterance_duration: 1.1

vad:
  # WebRTC runs first and short-circuits, so at 3 it also rejects quiet
  # speech before Silero gets a say. Lower this before suspecting the
  # model if speech is being missed on a quiet microphone.
  webrtc_sensitivity: 3
  silero_sensitivity: 0.05

audio8:
  bundle_dir: null   # null = $AUDIO8_ASR_BUNDLE, else ~/.cache/audio8-asr-onnx/model_bundle
  threads: 0         # 0 = all cores
```

Config discovery: the `-c/--config` flag, otherwise `config.yml` in the current working directory. Because the current directory *is* searched, prefer an explicit `-c` when invoking the CLI from anywhere but the checkout.

Fields belonging to the previous Python implementation (`transcription.model`, `device`, `compute_type`, `beam_size`, and the `realtime:` and `muse:` blocks) are accepted and ignored, so one config file can front either version.

## Systemd Integration

Copy the repo's `perception-voice.service` to
`~/.config/systemd/user/perception-voice.service`, or create your own:

```ini
[Unit]
Description=Perception Voice - Speech to Text Service
# sound.target is a no-op under PipeWire -- it does not imply a capture
# device exists yet.
After=pipewire.service pipewire-pulse.service wireplumber.service
Wants=pipewire.service pipewire-pulse.service wireplumber.service
After=sound.target
# Startup is sub-second, so the default limit of 5 starts per 10s is easy
# to trip just by restarting a few times by hand -- and hitting it leaves
# the unit in `failed`.
StartLimitIntervalSec=30
StartLimitBurst=15

[Service]
Type=simple
WorkingDirectory=/path/to/perception-voice
ExecStart=/path/to/perception-voice/target/release/perception-voice serve
Restart=on-failure
RestartSec=5
TimeoutStopSec=10s
Environment=PULSE_SERVER=unix:/run/user/%U/pulse/native

[Install]
WantedBy=default.target
```

Enable and start:

```bash
systemctl --user enable perception-voice
systemctl --user start perception-voice
```

Restart-to-ready is about 0.76s, measured over 10 consecutive restarts.

Capture is supervised rather than opened once: a stream that fails to open, or that opens and then stops delivering callbacks, is retried indefinitely with backoff, and 3s without an audio callback counts as a failure. The device is re-resolved on every attempt. Note the limit — this recovers when a device *node dies*, but merely changing the PipeWire default source leaves an already-open stream reading the old node, so a new default is picked up on the next reopen.

## Example Workflow

```
# Boot: systemd starts server
$ systemctl --user start perception-voice

# Later: your app starts
$ perception-voice client set mari

# User speaks: "turn on lights"

# Your app checks for commands
$ perception-voice client get mari
{"ts": "2026-01-20T14:32:05.123-08:00", "text": "turn on lights"}

# Your app processes the command...
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Any error, including usage errors |
