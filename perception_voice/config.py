"""
Configuration management for perception-voice
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Server configuration"""
    socket_path: str
    buffer_retention_minutes: int
    discard_phrases: List[str] = field(default_factory=list)
    # Optional path to a transcript log file. Null or empty = disabled
    # (in-memory buffer only). When set, every buffered utterance is
    # appended to the file as a timestamp-prefixed line
    # (`Sun, Sep 6 @ 4:09p | text`), surviving restarts while the
    # in-memory buffer keeps its TTL.
    transcript_log_path: Optional[str] = None


@dataclass
class AudioConfig:
    """Audio configuration"""
    sample_rate: int
    buffer_size: int
    mic_device: Optional[int] = None
    min_utterance_duration: float = 1.1
    post_speech_silence_duration: float = 0.6
    pre_recording_buffer_duration: float = 1.0


@dataclass
class TranscriptionConfig:
    """Transcription configuration"""
    model: str
    device: str
    compute_type: str
    language: str
    beam_size: int
    # "local"  = faster-whisper on this machine (GPU/CPU per `device`);
    # "muse"   = Meta Muse Voice Transcribe hosted API (no local model
    #            loaded at all — frees the GPU entirely; see `muse:` block)
    # "audio8" = Audio8-ASR-0.1B, tiny local open-weight model (~1-2 GB
    #            VRAM — small enough to share the GPU with a large LLM;
    #            see `audio8:` block). CC-BY-NC license, personal use.
    backend: str = "local"


@dataclass
class Audio8Config:
    """Audio8-ASR-0.1B backend settings (backend: "audio8")"""
    model: str = "Audio8/Audio8-ASR-0.1B"
    device: str = "cuda"           # "cuda" or "cpu"
    dtype: str = "bfloat16"        # "bfloat16" (GPU) or "float32" (CPU)
    max_new_tokens: int = 128
    max_clip_seconds: int = 30     # model cap; longer clips are chunked


@dataclass
class MuseConfig:
    """Meta Muse Voice Transcribe backend settings (backend: "muse")"""
    # API key; null falls back to $META_API_KEY. Keys look like "LLM|..."
    # and come from https://ai.developer.meta.com/ -> API keys.
    api_key: Optional[str] = None
    model: str = "muse-voice-transcribe-1.0"
    # True: one realtime WebSocket per utterance — audio streams while you
    # speak, partial hypotheses feed `subscribe words`, final lands fast.
    # False: buffered WAV POSTed after each utterance (no partials).
    # Either way the stream-failure fallback is the buffered POST.
    streaming: bool = True
    realtime_url: str = "wss://api.meta.ai/v1/asr/realtime"
    transcribe_url: str = "https://api.meta.ai/v1/asr/transcribe"
    language_bias: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    request_timeout: float = 30.0


@dataclass
class VADConfig:
    """Voice Activity Detection configuration"""
    webrtc_sensitivity: int
    silero_sensitivity: float
    silero_use_onnx: bool


@dataclass
class RealtimeConfig:
    """Secondary-model partial-hypothesis decoding (for `subscribe words`)"""
    enabled: bool = True
    model: str = "tiny.en"
    device: str = "cuda"
    compute_type: str = "float16"
    interval: float = 0.5
    beam_size: int = 1


@dataclass
class Config:
    """Main configuration container"""
    server: ServerConfig
    audio: AudioConfig
    transcription: TranscriptionConfig
    vad: VADConfig
    config_path: Path
    realtime: RealtimeConfig = field(default_factory=RealtimeConfig)
    muse: MuseConfig = field(default_factory=MuseConfig)
    audio8: Audio8Config = field(default_factory=Audio8Config)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Config":
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to config file. If None, searches for config.yml
                        in the project root (relative to package location).
        
        Returns:
            Config object
        
        Raises:
            SystemExit: If config file not found
        """
        # Find config file. Chain: explicit -c, $PERCEPTION_VOICE_CONFIG,
        # package parent (relative — resolves to the project root in a dev
        # checkout, exactly as before), then the XDG per-user location
        # ~/.config/perception-voice/config.yml (which covers `uv tool`
        # installs, where the package parent is site-packages — symlink it
        # at the repo's config.yml or copy one there).
        #
        # Deliberately NO cwd fallback: callers spawn this CLI from
        # arbitrary directories (mari's REPL polls via `perception-voice
        # client` from /workspace/mari, which has its own unrelated
        # config.yml), and picking up a foreign config.yml crashes parsing.
        # And no hardcoded absolute repo path: this must work on any
        # machine, not one specific checkout location.
        import os
        if config_path is not None:
            resolved_path = config_path
        else:
            package_dir = Path(__file__).parent
            xdg = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
            candidates = []
            if os.environ.get("PERCEPTION_VOICE_CONFIG"):
                candidates.append(Path(os.environ["PERCEPTION_VOICE_CONFIG"]))
            candidates.append(package_dir.parent / "config.yml")
            candidates.append(xdg / "perception-voice" / "config.yml")
            resolved_path = next((c for c in candidates if c.exists()), candidates[-1])

        # Resolve symlinks so a relative server.socket_path anchors at the
        # real config's directory (e.g. the repo root), not the symlink's.
        resolved_path = resolved_path.resolve()

        # Secrets live in an uncommitted .env beside the config file
        # (e.g. META_API_KEY), never in config.yml itself.
        _load_dotenv(resolved_path.parent / ".env")

        if not resolved_path.exists():
            logger.error(f"Config file not found: {resolved_path}")
            logger.error("Please copy config.example.yml to config.yml and customize it.")
            sys.exit(1)
        
        config_data = _load_yaml(resolved_path)
        logger.info(f"Loaded config from {resolved_path}")
        
        transcription = TranscriptionConfig(**config_data.get("transcription", {}))
        if transcription.backend not in ("local", "muse", "audio8"):
            logger.error(
                f"Invalid transcription.backend: {transcription.backend!r} "
                "(expected 'local', 'muse', or 'audio8')"
            )
            sys.exit(1)

        return cls(
            server=ServerConfig(**config_data.get("server", {})),
            audio=AudioConfig(**config_data.get("audio", {})),
            transcription=transcription,
            vad=VADConfig(**config_data.get("vad", {})),
            realtime=RealtimeConfig(**config_data.get("realtime", {})),
            muse=MuseConfig(**config_data.get("muse", {})),
            audio8=Audio8Config(**config_data.get("audio8", {})),
            config_path=resolved_path.parent,
        )
    
    def get_socket_path(self) -> Path:
        """Get the absolute path to the socket file"""
        socket_path = Path(self.server.socket_path)
        if socket_path.is_absolute():
            return socket_path
        return self.config_path / socket_path

    def get_transcript_log_path(self) -> Optional[Path]:
        """Get the absolute path to the transcript log file, if enabled.

        Returns None when `server.transcript_log_path` is null or empty
        (logging disabled). Relative paths anchor at the config file's
        directory, same convention as `get_socket_path`.
        """
        raw = self.server.transcript_log_path
        if raw is None:
            return None
        raw = os.path.expandvars(str(raw).strip())
        if not raw:
            return None
        log_path = Path(raw).expanduser()
        if log_path.is_absolute():
            return log_path
        return self.config_path / log_path


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines into os.environ (existing vars win)."""
    if not path.exists():
        return
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        logger.info(f"Loaded environment from {path}")
    except Exception as e:
        logger.warning(f"Could not read {path}: {e}")


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML file and return as dict"""
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
            return data if data else {}
    except Exception as e:
        logger.error(f"Error loading config file {path}: {e}")
        sys.exit(1)
