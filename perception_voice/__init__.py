"""
perception-voice: System-level speech-to-text service

A shared service that manages a single Whisper model in GPU VRAM,
providing voice transcription to multiple client processes via Unix socket IPC.
"""

__version__ = "0.1.0"
