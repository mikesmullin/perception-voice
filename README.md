# perception-voice

A system-level speech-to-text service that runs continuously as a systemd unit, providing shared voice transcription to multiple client processes via Unix socket IPC.

## Problem

When multiple applications need voice input, each must load its own copy of the Whisper AI model into GPU VRAM. This doesn't scale well.

## Solution

**perception-voice** runs a single daemon that:
- Loads the Whisper model once into GPU VRAM
- Continuously captures audio and transcribes speech
- Stores utterances in a time-indexed memory buffer
- Allows clients to retrieve transcriptions via Unix socket

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA support (recommended)
- Microphone

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
perception-voice client set <uid>   # Set read marker to now
perception-voice client get <uid>   # Get transcriptions since marker
```

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
  model: "large-v3"
  device: "cuda"
  compute_type: "float16"
```

## Systemd Integration

Create `/etc/systemd/user/perception-voice.service`:

```ini
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
