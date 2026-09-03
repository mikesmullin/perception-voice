"""
Audio transcription using Whisper model

Captures audio from microphone, detects speech using VAD,
and transcribes using faster-whisper.
"""

import collections
import ctypes
import glob
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np
import sounddevice as sd


def _preload_cuda12_libs() -> None:
    """ctranslate2 dlopens libcublas.so.12 / libcudnn.so.9 by soname with no
    RPATH into the pip `nvidia` packages, so it only works if something else
    (e.g. a cu12-flavored torch) already loaded them. A cu13 torch (what a
    fresh dependency resolve now installs) breaks decode with "Library
    libcublas.so.12 is not found". Preload the cu12 libs explicitly when the
    wheels are present; harmless no-op otherwise."""
    try:
        import nvidia
        bases = list(nvidia.__path__)
    except ImportError:
        return
    for rel in ("cublas/lib/libcublasLt.so.12", "cublas/lib/libcublas.so.12",
                "cudnn/lib/libcudnn.so.9"):
        for base in bases:
            for path in glob.glob(os.path.join(base, rel)):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


_preload_cuda12_libs()

from faster_whisper import WhisperModel  # noqa: E402  (after CUDA preload)

from perception_voice.config import (
    Audio8Config,
    AudioConfig,
    MuseConfig,
    RealtimeConfig,
    TranscriptionConfig,
    VADConfig,
)
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
        stream_hub=None,
        realtime_config: Optional[RealtimeConfig] = None,
        muse_config: Optional[MuseConfig] = None,
        audio8_config: Optional[Audio8Config] = None,
    ):
        """
        Initialize transcriber

        Args:
            audio_config: Audio settings
            transcription_config: Whisper model settings
            vad_config: VAD settings
            on_transcription: Callback for completed transcriptions
            verbose: Enable verbose logging
            stream_hub: Optional StreamHub for level/word push streams
            realtime_config: Optional secondary-model partial-decode settings
            muse_config: Meta Muse Voice settings (used when backend == "muse")
            audio8_config: Audio8-ASR settings (used when backend == "audio8")
        """
        self.audio_config = audio_config
        self.transcription_config = transcription_config
        self.vad_config = vad_config
        self.on_transcription = on_transcription
        self.verbose = verbose
        self._hub = stream_hub
        
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
        
        # STT engine: local faster-whisper model, or Meta's hosted Muse
        # Voice Transcribe API (backend "muse" loads nothing on the GPU —
        # VAD gating stays local, so only detected speech is sent out).
        self._model: Optional[WhisperModel] = None
        self._muse_file = None
        self._muse_key: Optional[str] = None
        self._muse_session = None  # in-flight MuseStreamSession, if any
        self._audio8 = None
        self.muse_config = muse_config or MuseConfig()

        if transcription_config.backend == "audio8":
            from perception_voice.audio8 import Audio8Engine
            self._audio8 = Audio8Engine(audio8_config or Audio8Config())
        elif transcription_config.backend == "muse":
            from perception_voice import muse
            self._muse_mod = muse
            try:
                self._muse_key = muse.resolve_api_key(self.muse_config)
            except muse.MuseAuthError as e:
                logger.error(str(e))
                raise SystemExit(1)
            self._muse_file = muse.MuseFileClient(self.muse_config, self._muse_key)
            logger.info(
                f"Using Meta Muse Voice backend: {self.muse_config.model} "
                f"(streaming={self.muse_config.streaming}; no local model loaded)"
            )
        else:
            logger.info(f"Loading Whisper model: {transcription_config.model}")
            self._model = WhisperModel(
                transcription_config.model,
                device=transcription_config.device,
                compute_type=transcription_config.compute_type,
            )
            logger.info(f"Whisper model loaded: {transcription_config.model}")

        # Secondary realtime model: decodes the in-progress utterance buffer
        # periodically so `subscribe words` subscribers get partial
        # hypotheses before the VAD tail closes (Ada latency path).
        # With the muse backend, partials come from the realtime WebSocket
        # instead — no local model needed. The audio8 backend has no
        # partials of its own, so the local realtime model still applies.
        self._partial_decoder: Optional[_PartialDecoder] = None
        if stream_hub and realtime_config and realtime_config.enabled:
            if transcription_config.backend == "muse":
                logger.info(
                    "realtime.enabled: partial hypotheses come from the Muse "
                    "stream (local realtime model not loaded)"
                )
            else:
                self._partial_decoder = _PartialDecoder(
                    self, realtime_config, transcription_config, stream_hub
                )
    
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
        
        logger.info("Stopping transcriber...")
        self.is_running = False
        
        # Stop sounddevice streams to unblock the recording thread
        try:
            sd.stop()
        except Exception as e:
            logger.debug(f"Error stopping sounddevice: {e}")
        
        if self._recording_thread:
            self._recording_thread.join(timeout=2.0)
            if self._recording_thread.is_alive():
                logger.warning("Recording thread did not stop in time")
        
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

            was_recording = self.is_recording

            if is_speech:
                if not self.is_recording:
                    self._start_recording()

                self._frames.append(audio_chunk)
                if self._muse_session:
                    self._muse_session.feed(audio_chunk)
                self._silence_count = 0

            elif self.is_recording:
                self._frames.append(audio_chunk)
                if self._muse_session:
                    self._muse_session.feed(audio_chunk)
                self._silence_count += 1

                if self._silence_count >= self._max_silence_chunks:
                    self._stop_recording()

            # Level stream: copy + enqueue only (feature math happens on the
            # hub worker, never on this realtime thread)
            if self._hub:
                self._hub.on_audio_chunk(
                    audio_chunk,
                    vad_active=is_speech,
                    started=(not was_recording and self.is_recording),
                    ended=(was_recording and not self.is_recording),
                )
        
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

        # Muse streaming: open a per-utterance realtime session and replay
        # the pre-recording buffer into it. feed() only enqueues, so this
        # stays safe on the audio callback thread; if the socket later
        # fails, _transcribe falls back to POSTing the buffered audio.
        if self._muse_file and self.muse_config.streaming:
            self._muse_session = self._muse_mod.MuseStreamSession(
                self.muse_config,
                self._muse_key,
                sample_rate=self.audio_config.sample_rate,
                on_partial=self._on_muse_partial,
            )
            for chunk in self._frames:
                self._muse_session.feed(chunk)

        if self.verbose:
            logger.info("Recording started")

    def _on_muse_partial(self, text: str) -> None:
        """Publish a live hypothesis from the Muse stream to `subscribe words`"""
        if self._hub:
            self._hub.publish_words(
                {"ev": "partial", "ts": time.time(), "text": text}
            )
    
    def _stop_recording(self) -> None:
        """Stop recording (called when silence detected)"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        recording_duration = time.time() - self._recording_start_time

        # Detach any in-flight Muse stream session from the callback path
        muse_session, self._muse_session = self._muse_session, None

        # Check minimum duration
        if recording_duration < self.audio_config.min_utterance_duration:
            if self.verbose:
                logger.debug(f"Recording too short ({recording_duration:.2f}s), discarding")
            self._frames = []
            if muse_session:
                muse_session.abort()
            return

        # Concatenate frames and transcribe
        audio_data = np.concatenate(self._frames)
        self._frames = []

        if self.verbose:
            logger.info(f"Recording stopped ({recording_duration:.2f}s), transcribing...")

        # Capture utterance start/end times as datetimes
        utterance_start = datetime.fromtimestamp(self._recording_start_time, tz=timezone.utc).astimezone()
        utterance_end = datetime.now(tz=timezone.utc).astimezone()

        # Transcribe in separate thread
        threading.Thread(
            target=self._transcribe,
            args=(audio_data, utterance_start, utterance_end, muse_session),
            daemon=True,
        ).start()

    def _transcribe(self, audio: np.ndarray, utterance_start: datetime,
                    utterance_end: Optional[datetime] = None,
                    muse_session=None) -> None:
        """Transcribe audio (local Whisper, Audio8, or Muse Voice API)"""
        try:
            if self._audio8:
                full_text = (self._audio8.transcribe(
                    audio, self.audio_config.sample_rate
                ) or "").strip()
            elif self._muse_file:
                # Streaming session already has the audio; ask it to finalize.
                # On any stream failure, POST the buffered audio instead.
                text = None
                if muse_session:
                    text = muse_session.finish(
                        timeout=self.muse_config.request_timeout
                    )
                if text is None:
                    text = self._muse_file.transcribe(
                        audio, self.audio_config.sample_rate
                    )
                full_text = (text or "").strip()
            else:
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
                    self.on_transcription(full_text, utterance_start, utterance_end)
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


class _PartialDecoder:
    """
    Secondary (small) Whisper model that re-decodes the accumulating
    utterance buffer every `interval` seconds while recording is active,
    publishing {"ev": "partial", ...} events to `subscribe words`
    subscribers. The main model still produces the finalized utterance.

    Same dual-model split /workspace/whisper's config describes
    (tiny.en preview + large final) — partials are for reacting early,
    the finalized utterance is what gets committed.
    """

    def __init__(self, transcriber: Transcriber, realtime_config: RealtimeConfig,
                 transcription_config: TranscriptionConfig, hub):
        self._t = transcriber
        self._cfg = realtime_config
        self._language = transcription_config.language
        self._hub = hub

        logger.info(f"Loading realtime Whisper model: {realtime_config.model}")
        self._model = WhisperModel(
            realtime_config.model,
            device=realtime_config.device,
            compute_type=realtime_config.compute_type,
        )
        logger.info(f"Realtime model loaded: {realtime_config.model}")

        self._decoded_chunks = 0
        threading.Thread(target=self._run, daemon=True, name="partial-decoder").start()

    def _run(self) -> None:
        while True:
            time.sleep(self._cfg.interval)

            if not self._t.is_recording:
                self._decoded_chunks = 0
                continue
            if not self._hub.has_subscribers("words"):
                continue

            # Snapshot the in-progress buffer (list() is atomic under the GIL;
            # chunks themselves are immutable once appended)
            frames = list(self._t._frames)
            if not frames or len(frames) <= self._decoded_chunks:
                continue
            self._decoded_chunks = len(frames)

            audio = np.concatenate(frames)
            try:
                segments, _ = self._model.transcribe(
                    audio,
                    language=self._language,
                    beam_size=self._cfg.beam_size,
                    condition_on_previous_text=False,
                )
                text = " ".join(s.text.strip() for s in segments).strip()
            except Exception as e:
                logger.debug(f"Partial decode error: {e}")
                continue

            if text:
                self._hub.publish_words(
                    {"ev": "partial", "ts": time.time(), "text": text}
                )
