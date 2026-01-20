# perception-voice: Plan Document

## Overview

**perception-voice** is a system-level speech-to-text service that runs continuously as a systemd unit. It captures audio from the microphone, transcribes it using a Whisper model, and stores the transcribed text in a time-indexed memory buffer. Other processes can query this service via a Unix domain socket to retrieve transcribed speech.

### Problem Statement

The prior project `whisper` (located at `tmp/whisper/`) is a capable voice transcription tool, but each process that needs voice input must load its own copy of the AI model into GPU VRAM. This architecture doesn't scale well when multiple applications need voice capabilities simultaneously.

### Solution

Decompose the voice transcription into a shared service:

1. **Single Process Model Loading**: One daemon process manages the Whisper model in GPU VRAM
2. **Persistent Memory Buffer**: Stores transcribed text with timestamps (configurable retention, default 30 minutes)
3. **IPC via Unix Domain Socket**: Client processes communicate with the server to retrieve transcriptions
4. **Per-Client Read Markers**: Each client maintains its own "read position" in the buffer via a unique identifier (uid)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     perception-voice serve                       │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────┐  │
│  │ Microphone  │──▶│ Whisper Model│──▶│ Memory Buffer        │  │
│  │   Input     │   │ (GPU/CUDA)   │   │ (timestamped text)   │  │
│  └─────────────┘   └──────────────┘   └──────────────────────┘  │
│                                                │                 │
│                    ┌───────────────────────────┘                 │
│                    ▼                                             │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Unix Domain Socket (IPC)                       ││
│  │                  /workspace/perception-voice/perception.sock ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
           │                              │
           ▼                              ▼
    ┌─────────────┐                ┌─────────────┐
    │ Client: mari│                │ Client: bob │
    │  uid="mari" │                │  uid="bob"  │
    └─────────────┘                └─────────────┘
```

---

## CLI Interface

### Server Mode

```bash
perception-voice serve [-v]
```

Starts the server daemon. Loads the Whisper model (downloading if needed on first run), opens the microphone, and begins transcribing audio continuously. Listens on Unix domain socket for client connections.

**Options:**
- `-v, --verbose`: Enable verbose logging (logs metadata like word count, timestamps; does not log transcribed text)

By default, minimal output to stdout (suitable for journald). Server exits cleanly on SIGTERM.

### Client Mode

```bash
perception-voice client <subcommand> <uid>
```

#### Subcommands

| Command | Description |
|---------|-------------|
| `set <uid>` | Set the read marker for `<uid>` to the current time. Returns nothing, exits 0. |
| `get <uid>` | Retrieve all transcribed text since the read marker for `<uid>` as JSONL, then update the read marker to now. Prints JSONL to stdout (empty if no new text). Exits 0. |

#### Examples

```bash
# Client "mari" initializes read marker
perception-voice client set mari

# ... time passes, server hears "turn on lights" ...

# Client "mari" retrieves new transcriptions
perception-voice client get mari
# Output (JSONL):
# {"ts": "2026-01-20T14:32:05.123-08:00", "text": "turn on lights"}
```

---

## Components (Inspired by tmp/whisper)

### From Whisper to Reuse/Adapt

| Whisper Module | Purpose | Adaptation for perception-voice |
|----------------|---------|--------------------------------|
| `lib/audio_recorder.py` | Multiprocessing audio capture + dual-model transcription | Extract core transcription loop; remove keyboard/hotkey logic; use single model only |
| `lib/vad.py` | Voice Activity Detection (WebRTC + Silero) | Reuse as-is |
| `lib/config.py` | YAML config loading | Adapt for `config.yml` with our settings |
| `lib/sound.py` | Audio feedback SFX | **Not needed (remove)** |
| `lib/keyboard_output.py` | Keyboard typing | **Not needed (remove)** |

### New Components for perception-voice

| Module | Purpose |
|--------|---------|
| `perception_voice/server.py` | Main server loop, IPC handling, memory buffer management |
| `perception_voice/client.py` | CLI client for `set` and `get` commands |
| `perception_voice/buffer.py` | Time-indexed transcription storage with configurable TTL |
| `perception_voice/ipc.py` | Unix domain socket protocol (request/response) |
| `perception_voice/transcriber.py` | Whisper model wrapper (adapted from audio_recorder.py, single model) |
| `perception_voice/config.py` | Configuration loading from `config.yml` |
| `perception_voice/cli.py` | CLI entry point with subcommands |

### Whisper Model

- Uses the same model as Whisper project: **`large-v3`**
- Model is **not committed to git** (gitignored)
- Model is **automatically downloaded on first startup** by `faster_whisper` library
- No separate download script needed; `faster_whisper.WhisperModel()` handles caching

---

## Configuration (`config.yml`)

```yaml
# perception-voice configuration

server:
  # Unix domain socket path (relative to workspace root, or absolute path)
  socket_path: "perception.sock"
  
  # How long to retain transcriptions in memory (minutes)
  buffer_retention_minutes: 30

audio:
  sample_rate: 16000
  buffer_size: 512
  mic_device: null  # auto-detect

transcription:
  # Model for transcription (same as Whisper project)
  # Downloaded automatically on first startup by faster_whisper
  model: "large-v3"
  device: "cuda"
  compute_type: "float16"
  language: "en"
  beam_size: 5

vad:
  webrtc_sensitivity: 3
  silero_sensitivity: 0.05
  silero_use_onnx: true
```

---

## Project Structure

```
perception-voice/
├── pyproject.toml
├── config.yml
├── config.example.yml
├── .gitignore
├── README.md
├── docs/
│   └── PLAN.md
├── perception_voice/
│   ├── __init__.py
│   ├── cli.py           # Entry point, argparse
│   ├── server.py        # Server daemon main loop
│   ├── client.py        # Client commands (set, get)
│   ├── buffer.py        # Timestamped text storage
│   ├── ipc.py           # Unix socket protocol
│   ├── transcriber.py   # Whisper model + audio capture
│   ├── vad.py           # Voice activity detection (from whisper)
│   └── config.py        # Config loader
└── tmp/                  # (gitignored, contains whisper reference)
```

---

## IPC Protocol (Unix Domain Socket)

Simple JSON-based request/response over Unix socket.

### Request Format

```json
{"command": "set", "uid": "mari"}
{"command": "get", "uid": "mari"}
```

### Response Format

**For `set`:**
```json
{"status": "ok"}
```

**For `get`:**
```json
{"status": "ok", "text": "{JSONL data}"}
```

**On error:**
```json
{"status": "error", "message": "unknown command"}
```

### JSONL Output Format (for `get`)

Each utterance is returned as a JSONL line with timestamp (ISO 8601 with milliseconds and timezone) and text:

```jsonl
{"ts": "2026-01-20T14:32:05.123-08:00", "text": "turn on lights"}
{"ts": "2026-01-20T14:32:12.456-08:00", "text": "hello world"}
```

If no new text since last read marker, returns empty string.

---

## CLI Exit Codes

| Exit Code | Meaning |
|-----------|---------|
| `0` | Success (including `get` with no new text) |
| `1` | General error (e.g., server not running, connection failed) |
| `2` | Invalid arguments / usage error |

---

## Systemd Unit

```ini
# /etc/systemd/user/perception-voice.service
[Unit]
Description=Perception Voice - Speech to Text Service
After=sound.target

[Service]
Type=simple
ExecStart=/path/to/perception-voice serve
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

---

## Development Guidelines

- **Python with `uv`** for package management
- **No file > 500 lines**
- **No function > 50 lines**
- **CUDA/GPU acceleration** preferred (device: "cuda", compute_type: "float16")
- **Single model architecture** (large-v3 only, no realtime preview model)
- **Logging**: stdout for journald, `-v` flag for verbose mode
- **No audio SFX** (removed from scope)
- Do **not** edit files under `tmp/`

---

## Workflow Example

```
[Boot]
  └─▶ systemd starts `perception-voice serve`
        └─▶ Loads Whisper model into GPU (downloads on first run)
        └─▶ Opens microphone
        └─▶ Listens on Unix socket

[Later: mari starts]
  └─▶ mari runs: `perception-voice client set mari`
        └─▶ Server sets read_marker["mari"] = now

[User speaks: "turn on lights"]
  └─▶ Server transcribes, stores: {ts: "2026-01-20T14:32:05.123-08:00", text: "turn on lights"}

[User speaks: "hello world"]
  └─▶ Server transcribes, stores: {ts: "2026-01-20T14:32:12.456-08:00", text: "hello world"}

[Later: mari checks in]
  └─▶ mari runs: `perception-voice client get mari`
        └─▶ Server returns JSONL since read_marker["mari"]:
            {"ts": "2026-01-20T14:32:05.123-08:00", "text": "turn on lights"}
            {"ts": "2026-01-20T14:32:12.456-08:00", "text": "hello world"}
        └─▶ Server updates read_marker["mari"] = now
        └─▶ Client prints JSONL to stdout
        └─▶ Client exits 0
  └─▶ mari parses JSONL and processes commands
```

---

## Decisions Made

| Topic | Decision |
|-------|----------|
| Output format | JSONL with ISO 8601 timestamp (milliseconds + timezone) and utterance text |
| Utterance handling | Each utterance is its own timestamped entry |
| Read marker on `get` | Always advances to now |
| Socket location | Configurable via `config.yml`, defaults to workspace root |
| Empty `get` result | Returns empty string, exits 0 |
| PID file | Not needed |
| Signal handling | Standard (exits on SIGTERM) |
| Authentication | Not needed (Unix socket permissions suffice) |
| Logging | stdout for journald; `-v` for verbose (metadata only, not transcribed text) |
| Model architecture | Single model only (`large-v3`); may revisit if speed is an issue |
| Health check | Not needed |
| Audio SFX | Not included |
