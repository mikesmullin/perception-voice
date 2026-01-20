"""
Configuration management for perception-voice
"""

import logging
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


@dataclass
class VADConfig:
    """Voice Activity Detection configuration"""
    webrtc_sensitivity: int
    silero_sensitivity: float
    silero_use_onnx: bool


@dataclass
class Config:
    """Main configuration container"""
    server: ServerConfig
    audio: AudioConfig
    transcription: TranscriptionConfig
    vad: VADConfig
    config_path: Path

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
        # Find config file
        if config_path is not None:
            resolved_path = config_path
        else:
            # Resolve relative to package location (project root is parent of perception_voice/)
            package_dir = Path(__file__).parent
            project_root = package_dir.parent
            resolved_path = project_root / "config.yml"
        
        if not resolved_path.exists():
            logger.error(f"Config file not found: {resolved_path}")
            logger.error("Please copy config.example.yml to config.yml and customize it.")
            sys.exit(1)
        
        config_data = _load_yaml(resolved_path)
        logger.info(f"Loaded config from {resolved_path}")
        
        return cls(
            server=ServerConfig(**config_data.get("server", {})),
            audio=AudioConfig(**config_data.get("audio", {})),
            transcription=TranscriptionConfig(**config_data.get("transcription", {})),
            vad=VADConfig(**config_data.get("vad", {})),
            config_path=resolved_path.parent,
        )
    
    def get_socket_path(self) -> Path:
        """Get the absolute path to the socket file"""
        socket_path = Path(self.server.socket_path)
        if socket_path.is_absolute():
            return socket_path
        return self.config_path / socket_path


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML file and return as dict"""
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
            return data if data else {}
    except Exception as e:
        logger.error(f"Error loading config file {path}: {e}")
        sys.exit(1)
