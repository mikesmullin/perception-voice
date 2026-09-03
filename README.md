# perception-voice

A system-level speech-to-text service that runs continuously as a systemd unit, providing shared voice transcription to multiple client processes via Unix socket IPC.

## Problem

When multiple applications need voice input, each must load its own copy of the Whisper AI model into GPU VRAM. This doesn't scale well.

## Solution

**perception-voice** runs a single daemon that:
- Loads the Whisper model once into GPU VRAM
- Continuously captures audio and transcribes speech
- Stores utterances in a time-indexed memory buffer
- Allows clients to retrieve transcriptions via Unix socket (poll with
  `get`/`set`, or **subscribe to real-time push streams** — see below)
- Optionally runs a second small model that decodes the in-progress
  utterance every ~0.5s, pushing **partial hypotheses** before the
  speaker even finishes the sentence
- Publishes **~60 Hz audio feature frames** (rms + 4 spectral bands +
  VAD) for driving visualizations, like a voice-reactive avatar

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA support (recommended for the `local` backend;
  not needed at all with the `muse` backend)
- Microphone

## STT Backends

Set `transcription.backend` in `config.yml`:

| Backend | What it is | GPU VRAM |
|---|---|---|
| `local` (default) | faster-whisper running on this machine | ~2-3 GB |
| `muse` | [Meta Muse Voice Transcribe](https://ai.developer.meta.com/docs/speech-to-text) hosted API | **zero** |
| `audio8` | [Audio8-ASR-0.1B](https://huggingface.co/Audio8/Audio8-ASR-0.1B), tiny local open-weight model | ~1.2 GB |

`audio8` is the offline middle ground: small enough to share the GPU
with a large LLM (~1.15 GB measured footprint incl. CUDA context,
~80 ms warm inference per utterance on an RTX 5090), free, and private —
at the cost of fewer languages (en, zh, yue, fr, de, ja, ko) and a
CC-BY-NC-4.0 license (personal use only). Install its extra deps with
`uv pip install -e '.[audio8]'`; weights download on first startup. It
has no partials of its own — enable the `realtime:` block for
`subscribe words` partials, or use `muse` for API-native ones.

The `muse` backend loads no model at all — useful when you need the VRAM
for something else (e.g. a local LLM). Everything else stays identical:
VAD gating, the utterance buffer, `get`/`set`, and both push streams.
Because speech detection stays local, only VAD-detected utterances are
ever sent to Meta (~$0.003 per spoken minute — $3.00/1,000 audio min),
never the idle microphone.

With `muse.streaming: true` (default), each utterance opens one realtime
WebSocket: audio is forwarded while you're still speaking, live partial
hypotheses feed `subscribe words` (replacing the local `tiny.en` realtime
model), and the final transcript lands ~0.16s after end of speech. If a
stream fails mid-utterance, the locally buffered audio is POSTed to the
file endpoint automatically — no audio is lost.

Auth: set `muse.api_key` in `config.yml`, or put `META_API_KEY=...` in
an uncommitted `.env` next to `config.yml` (loaded automatically), or
export `META_API_KEY` (create a key at <https://ai.developer.meta.com/>
→ API keys; they look like `LLM|...`).

## Installation

Using `uv`:

```bash
uv tool install --editable . --with webrtcvad-wheels
```

## Quick Start

### 1. Configure (optional)

Copy the example config and customize:

```bash
cp config.example.yml config.yml
```

### 2. Start the server

```bash
perception-voice serve
```

The server will:
- Download the Whisper model on first run (if needed)
- Start listening on the microphone
- Create a Unix socket at `perception.sock`

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
perception-voice serve [-v]
```

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Enable verbose logging (metadata only) |

### Client Mode

```bash
perception-voice client set <uid>       # Set read marker to now
perception-voice client get <uid>       # Get transcriptions since marker
perception-voice client stream words    # Follow the words push stream (debug)
perception-voice client stream levels   # Follow the levels push stream (debug)
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
  - `{"ev":"partial", "ts":…, "text":"turn on the des"}` — live hypothesis
    from the secondary realtime model while the utterance is still being
    spoken (see the `realtime:` config block)
  - `{"ev":"utterance", "ts":…, "text":"Turn on the desk light.",
    "t_start":…, "t_end":…}` — the finalized main-model transcription,
    with unix-epoch start/end times of the speech itself
- **`levels`** — raw binary feature frames at 62.5 Hz for visualization:
  36 bytes little-endian — magic `'AVF1'` (u32), stream_id (u32, 0=mic),
  rms (f32), 4 spectral band energies (f32, 0–300/300–1k/1k–3k/3k–8k Hz),
  pitch_hint (f32, unused), flags (u32: bit0 VAD, bit1 utterance-start,
  bit2 utterance-end)

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

## Configuration

See `config.example.yml` for all options. Key settings:

```yaml
server:
  socket_path: "perception.sock"
  buffer_retention_minutes: 30

transcription:
  model: "large-v3-turbo"
  device: "cuda"
  compute_type: "float16"

realtime:
  # Secondary small model decoding the in-progress utterance for the
  # `subscribe words` partial-hypothesis events
  enabled: true
  model: "tiny.en"
  interval: 0.5
```

Config discovery order (first match wins): `-c/--config` flag,
`$PERCEPTION_VOICE_CONFIG`, `config.yml` next to the package (a dev
checkout), then `~/.config/perception-voice/config.yml` — symlink that to
your checkout's config for `uv tool` installs. The current directory is
deliberately **not** searched, so spawning the CLI from a directory with
an unrelated `config.yml` is safe.

## Systemd Integration

Copy the repo's `perception-voice.service` to
`~/.config/systemd/user/perception-voice.service`, or create your own:

```ini
[Unit]
Description=Perception Voice - Speech to Text Service
After=sound.target

[Service]
Type=simple
# Run from the checkout's venv: a tool install re-resolves dependencies
# on every reinstall (a cu13 torch once slipped in and broke faster-
# whisper's cu12 CUDA decode); the checkout venv is pinned and stable.
WorkingDirectory=/path/to/perception-voice
ExecStart=/path/to/perception-voice/.venv/bin/perception-voice serve
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

Enable and start:

```bash
systemctl --user enable perception-voice
systemctl --user start perception-voice
```

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
| 1 | General error |
| 2 | Usage error |
