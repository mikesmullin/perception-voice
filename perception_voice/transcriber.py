"""
Audio transcription using Whisper model

Captures audio from microphone, detects speech using VAD,
and transcribes using faster-whisper.
"""

import collections
import logging
import threading
import time
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from perception_voice.config import AudioConfig, TranscriptionConfig, VADConfig
from perception_voice.vad import VoiceActivityDetector

logger = logging.getLogger(__name__)


class Transcriber:
    """
    Real-time audio transcriber using Whisper
    
    Features:
    - Continuous audio capture from microphone
    - Dual VAD (WebRTC + Silero) for accurate speech detection
    - Single Whisper model for transcription
    - Callback-based output
    """
    
    def __init__(
        self,
        audio_config: AudioConfig,
        transcription_config: TranscriptionConfig,
        vad_config: VADConfig,
        on_transcription: Optional[Callable[[str], None]] = None,
        verbose: bool = False,
    ):
        """
        Initialize transcriber
        
        Args:
            audio_config: Audio settings
            transcription_config: Whisper model settings
            vad_config: VAD settings
            on_transcription: Callback for completed transcriptions
            verbose: Enable verbose logging
        """
        self.audio_config = audio_config
        self.transcription_config = transcription_config
        self.vad_config = vad_config
        self.on_transcription = on_transcription
        self.verbose = verbose
        
        # State
        self.is_running = False
        self.is_recording = False
        self._recording_thread: Optional[threading.Thread] = None
        self._mic_device: Optional[int] = None
        
        # Recording buffers
        self._audio_buffer: collections.deque = collections.deque(
            maxlen=int(
                (audio_config.sample_rate // audio_config.buffer_size)
                * audio_config.pre_recording_buffer_duration
            )
        )
        self._frames: list = []
        self._silence_count = 0
        self._max_silence_chunks = int(
            (audio_config.sample_rate / audio_config.buffer_size)
            * audio_config.post_speech_silence_duration
        )
        self._recording_start_time = 0.0
        
        # Statistics
        self.transcription_count = 0
        
        # Initialize VAD
        logger.info("Initializing VAD...")
        self._vad = VoiceActivityDetector(
            webrtc_sensitivity=vad_config.webrtc_sensitivity,
            silero_sensitivity=vad_config.silero_sensitivity,
            silero_use_onnx=vad_config.silero_use_onnx,
            sample_rate=audio_config.sample_rate,
        )
        
        # Load Whisper model
        logger.info(f"Loading Whisper model: {transcription_config.model}")
        self._model = WhisperModel(
            transcription_config.model,
            device=transcription_config.device,
            compute_type=transcription_config.compute_type,
        )
        logger.info(f"Whisper model loaded: {transcription_config.model}")
    
    def start(self) -> None:
        """Start the transcriber"""
        if self.is_running:
            logger.warning("Transcriber already running")
            return
        
        self.is_running = True
        
        # Auto-detect microphone if not specified
        if self.audio_config.mic_device is None:
            self._mic_device = self._auto_detect_microphone()
        else:
            self._mic_device = self.audio_config.mic_device
        
        # Start recording thread
        self._recording_thread = threading.Thread(
            target=self._recording_worker,
            daemon=True,
        )
        self._recording_thread.start()
        
        logger.info(f"Transcriber started (mic device: {self._mic_device})")
    
    def stop(self) -> None:
        """Stop the transcriber"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self._recording_thread:
            self._recording_thread.join(timeout=2.0)
        
        logger.info(f"Transcriber stopped (transcriptions: {self.transcription_count})")
    
    def _auto_detect_microphone(self) -> int:
        """Auto-detect default microphone device"""
        try:
            default_idx = sd.default.device[0]
            device_info = sd.query_devices(default_idx)
            if device_info['max_input_channels'] > 0:
                logger.info(f"Auto-detected microphone: [{default_idx}] {device_info['name']}")
                return default_idx
        except Exception as e:
            logger.warning(f"Could not auto-detect default mic: {e}")
        
        # Fallback: find first microphone
        try:
            devices = sd.query_devices()
            for idx, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    logger.info(f"Using first available mic: [{idx}] {device['name']}")
                    return idx
        except Exception as e:
            logger.error(f"Could not detect any microphone: {e}")
        
        return 0
    
    def _recording_worker(self) -> None:
        """Recording worker thread - captures audio and processes VAD"""
        
        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Audio status: {status}")
            
            # Extract mono audio
            audio_chunk = indata[:, 0].copy()
            
            # Add to circular buffer (for pre-recording)
            self._audio_buffer.append(audio_chunk)
            
            # Check for speech
            is_speech, confidence = self._vad.is_speech(audio_chunk)
            
            if is_speech:
                if not self.is_recording:
                    self._start_recording()
                
                self._frames.append(audio_chunk)
                self._silence_count = 0
                
            elif self.is_recording:
                self._frames.append(audio_chunk)
                self._silence_count += 1
                
                if self._silence_count >= self._max_silence_chunks:
                    self._stop_recording()
        
        try:
            with sd.InputStream(
                device=self._mic_device,
                channels=1,
                samplerate=self.audio_config.sample_rate,
                blocksize=self.audio_config.buffer_size,
                callback=audio_callback,
            ):
                logger.info("Audio stream started")
                while self.is_running:
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"Recording worker error: {e}")
    
    def _start_recording(self) -> None:
        """Start recording (called when speech detected)"""
        self.is_recording = True
        self._recording_start_time = time.time()
        
        # Include pre-recording buffer
        self._frames = list(self._audio_buffer)
        self._silence_count = 0
        
        if self.verbose:
            logger.info("Recording started")
    
    def _stop_recording(self) -> None:
        """Stop recording (called when silence detected)"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        recording_duration = time.time() - self._recording_start_time
        
        # Check minimum duration
        if recording_duration < self.audio_config.min_utterance_duration:
            if self.verbose:
                logger.debug(f"Recording too short ({recording_duration:.2f}s), discarding")
            self._frames = []
            return
        
        # Concatenate frames and transcribe
        audio_data = np.concatenate(self._frames)
        self._frames = []
        
        if self.verbose:
            logger.info(f"Recording stopped ({recording_duration:.2f}s), transcribing...")
        
        # Transcribe in separate thread
        threading.Thread(
            target=self._transcribe,
            args=(audio_data,),
            daemon=True,
        ).start()
    
    def _transcribe(self, audio: np.ndarray) -> None:
        """Transcribe audio with Whisper model"""
        try:
            segments, info = self._model.transcribe(
                audio,
                language=self.transcription_config.language,
                beam_size=self.transcription_config.beam_size,
            )
            
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text.strip())
            
            full_text = " ".join(text_parts).strip()
            
            if not full_text:
                return
            
            self.transcription_count += 1
            word_count = len(full_text.split())
            
            if self.verbose:
                logger.info(f"Transcribed: {word_count} words")
            
            if self.on_transcription:
                try:
                    self.on_transcription(full_text)
                except Exception as e:
                    logger.error(f"Error in transcription callback: {e}")
        
        except Exception as e:
            logger.error(f"Transcription error: {e}")
    
    def get_stats(self) -> dict:
        """Get transcriber statistics"""
        return {
            "transcription_count": self.transcription_count,
            "is_running": self.is_running,
            "is_recording": self.is_recording,
            "vad_stats": self._vad.get_statistics(),
        }
