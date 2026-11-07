"""
Real-time push streams (Ada's primary input path).

Clients send a normal framed JSON request {"command": "subscribe",
"channel": "levels" | "words"}; after the ok ack the connection becomes a
push stream owned by the StreamHub:

- levels: raw binary FeatureFrames (36 bytes, magic 'AVF1'), two per
  512-sample mic chunk @ 16 kHz = 62.5 Hz. Consumed by `ada avatar`.
- words:  framed JSON events — partial hypotheses while an utterance is
  still being spoken, then the finalized utterance. Consumed by the Ada
  brain.

Wire spec: /workspace/ada/docs/PROTOCOL.md.

All fan-out happens on a single worker thread so the realtime audio
callback and the decode threads never block on a slow client; the queue
drops oldest entries under pressure (animation frames are disposable).
"""

import logging
import queue
import struct
import threading
import time
from typing import Any, Dict, List

import numpy as np

from perception_voice.ipc import send_message

logger = logging.getLogger(__name__)

# 36 bytes little-endian: magic, stream_id, rms, band[4], pitch_hint, flags
FRAME_STRUCT = struct.Struct("<II6fI")
FRAME_MAGIC = 0x31465641  # 'AVF1'
STREAM_MIC = 0

FLAG_VAD = 1 << 0
FLAG_START = 1 << 1
FLAG_END = 1 << 2

CHANNELS = ("levels", "words")


def pack_frame(rms: float, bands, pitch: float = 0.0, flags: int = 0) -> bytes:
    return FRAME_STRUCT.pack(FRAME_MAGIC, STREAM_MIC, rms, *bands, pitch, flags)


class FeatureExtractor:
    """rms + 4 spectral band energies (low / low-mid / high-mid / high),
    each softly compressed into 0..1 for driving shader uniforms."""

    BAND_EDGES_HZ = (0, 300, 1000, 3000, 8000)

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self._n = 0

    def _prepare(self, n: int) -> None:
        freqs = np.fft.rfftfreq(n, 1.0 / self.sample_rate)
        self._masks = [
            (freqs >= lo) & (freqs < hi)
            for lo, hi in zip(self.BAND_EDGES_HZ, self.BAND_EDGES_HZ[1:])
        ]
        self._window = np.hanning(n).astype(np.float32)
        self._n = n

    def features(self, chunk: np.ndarray):
        n = len(chunk)
        if n != self._n:
            self._prepare(n)
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        # normalized magnitude spectrum: a full-scale sine peaks ~1.0
        spec = np.abs(np.fft.rfft(chunk * self._window)) * (2.0 / n)
        bands = []
        for mask in self._masks:
            peak = float(spec[mask].max()) if mask.any() else 0.0
            bands.append(min(1.0, (peak * 8.0) ** 0.7))
        return min(1.0, (rms * 8.0) ** 0.7), bands


class StreamHub:
    """Subscriber registry + single fan-out worker."""

    def __init__(self, sample_rate: int):
        self._extractor = FeatureExtractor(sample_rate)
        self._subs: Dict[str, List] = {ch: [] for ch in CHANNELS}
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue(maxsize=512)
        threading.Thread(target=self._worker, daemon=True, name="stream-hub").start()

    # -- subscription (called from client handler threads) ----------------

    def add_subscriber(self, channel: str, sock) -> None:
        sock.settimeout(2.0)  # a stuck client gets dropped, not waited on
        with self._lock:
            self._subs[channel].append(sock)
        logger.info(f"Stream subscriber added: {channel}")

    def has_subscribers(self, channel: str) -> bool:
        return bool(self._subs[channel])

    # -- publishing (called from audio callback / decode threads) ---------

    def on_audio_chunk(self, chunk: np.ndarray, vad_active: bool,
                       started: bool, ended: bool) -> None:
        """Realtime audio thread: copy + enqueue only, no math here."""
        if not self.has_subscribers("levels"):
            return
        self._enqueue(("_chunk", (chunk.copy(), vad_active, started, ended)))

    def publish_words(self, event: Dict[str, Any]) -> None:
        if not self.has_subscribers("words"):
            return
        self._enqueue(("words", event))

    def _enqueue(self, item) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()  # drop oldest; frames are disposable
                self._queue.put_nowait(item)
            except queue.Empty:
                pass

    # -- worker ------------------------------------------------------------

    def _worker(self) -> None:
        while True:
            kind, payload = self._queue.get()
            if kind == "_chunk":
                chunk, vad_active, started, ended = payload
                # two half-chunk windows per 512-sample chunk -> 62.5 Hz
                half = len(chunk) // 2
                flags = FLAG_VAD if vad_active else 0
                first = pack_frame(*self._extractor.features(chunk[:half]),
                                   flags=flags | (FLAG_START if started else 0))
                second = pack_frame(*self._extractor.features(chunk[half:]),
                                    flags=flags | (FLAG_END if ended else 0))
                self._fanout("levels", first + second)
            else:
                self._fanout("words", payload)

    def _fanout(self, channel: str, payload) -> None:
        with self._lock:
            subs = list(self._subs[channel])
        for sock in subs:
            try:
                if channel == "levels":
                    sock.sendall(payload)
                else:
                    send_message(sock, payload)
            except Exception:
                self._drop(channel, sock)

    def _drop(self, channel: str, sock) -> None:
        with self._lock:
            try:
                self._subs[channel].remove(sock)
            except ValueError:
                pass
        try:
            sock.close()
        except Exception:
            pass
        logger.info(f"Stream subscriber dropped: {channel}")
