"""
Voice Activity Detection (VAD) module

Implements dual-VAD system:
- WebRTC VAD: Fast pre-filter (~1ms per chunk)
- Silero VAD: Accurate verification (~10-20ms per chunk)

This approach minimizes CPU usage while maintaining high accuracy.
Adapted from whisper project.
"""

import logging
from typing import Tuple

import numpy as np
import torch
import webrtcvad

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """
    Combined WebRTC + Silero VAD for optimal performance
    
    WebRTC quickly filters out silence, Silero verifies potential speech.
    This reduces Silero calls by ~95%, significantly improving performance.
    """
    
    def __init__(
        self,
        webrtc_sensitivity: int = 3,
        silero_sensitivity: float = 0.05,
        silero_use_onnx: bool = True,
        sample_rate: int = 16000
    ):
        """
        Initialize dual VAD system
        
        Args:
            webrtc_sensitivity: 0-3, higher = less sensitive to noise
            silero_sensitivity: 0.0-1.0, lower = more sensitive
            silero_use_onnx: Use ONNX for faster CPU inference
            sample_rate: Audio sample rate in Hz
        """
        self.webrtc_sensitivity = webrtc_sensitivity
        self.silero_sensitivity = silero_sensitivity
        self.sample_rate = sample_rate
        
        # Initialize WebRTC VAD (fast pre-filter)
        logger.info(f"Initializing WebRTC VAD (sensitivity={webrtc_sensitivity})")
        self.webrtc_vad = webrtcvad.Vad(webrtc_sensitivity)
        logger.info("WebRTC VAD loaded")
        
        # Initialize Silero VAD (accurate verification)
        logger.info(f"Initializing Silero VAD (sensitivity={silero_sensitivity}, onnx={silero_use_onnx})")
        self.silero_vad, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            verbose=False,
            onnx=silero_use_onnx,
            trust_repo=True
        )
        logger.info("Silero VAD loaded")
        
        # Statistics
        self.webrtc_checks = 0
        self.silero_checks = 0
        self.speech_detected_count = 0
    
    def is_speech(self, audio_chunk: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if audio chunk contains speech
        
        Args:
            audio_chunk: Audio data as numpy array (float32, -1.0 to 1.0)
        
        Returns:
            Tuple of (is_speech, confidence)
        """
        self.webrtc_checks += 1
        
        # WebRTC requires 480 samples (30ms at 16kHz)
        webrtc_frame_size = 480
        
        # Convert float32 to int16 for WebRTC
        audio_clipped = np.clip(audio_chunk, -1.0, 1.0)
        audio_int16 = (audio_clipped * 32767).astype(np.int16)
        
        # Prepare WebRTC frame
        if len(audio_int16) < webrtc_frame_size:
            webrtc_audio = np.pad(audio_int16, (0, webrtc_frame_size - len(audio_int16)))
        else:
            webrtc_audio = audio_int16[:webrtc_frame_size]
        
        try:
            webrtc_speech = self.webrtc_vad.is_speech(
                webrtc_audio.tobytes(),
                self.sample_rate
            )
        except Exception as e:
            logger.warning(f"WebRTC VAD error: {e}, using Silero only")
            webrtc_speech = True
        
        # If WebRTC says no speech, skip expensive Silero check
        if not webrtc_speech:
            return False, 0.0
        
        # Verify with Silero
        self.silero_checks += 1
        
        # Silero requires at least 512 samples
        silero_min_size = 512
        if len(audio_chunk) < silero_min_size:
            silero_audio = np.pad(audio_chunk, (0, silero_min_size - len(audio_chunk)))
        else:
            silero_audio = audio_chunk
        
        audio_tensor = torch.from_numpy(silero_audio).float()
        with torch.no_grad():
            speech_prob = self.silero_vad(audio_tensor, self.sample_rate).item()
        
        is_speech = speech_prob > self.silero_sensitivity
        
        if is_speech:
            self.speech_detected_count += 1
        
        return is_speech, speech_prob
    
    def get_statistics(self) -> dict:
        """Get VAD performance statistics"""
        efficiency = 0.0
        if self.webrtc_checks > 0:
            efficiency = (1 - self.silero_checks / self.webrtc_checks) * 100
        
        return {
            'webrtc_checks': self.webrtc_checks,
            'silero_checks': self.silero_checks,
            'speech_detected': self.speech_detected_count,
            'efficiency_percent': efficiency,
        }
    
    def reset_statistics(self):
        """Reset statistics counters"""
        self.webrtc_checks = 0
        self.silero_checks = 0
        self.speech_detected_count = 0
