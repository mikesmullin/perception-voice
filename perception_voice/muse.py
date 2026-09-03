"""
Meta Muse Voice Transcribe backend (hosted STT via the Meta Model API).

Two paths, both fed by the existing local VAD pipeline (only detected
speech ever leaves the machine — cost is per spoken minute, not per
wall-clock hour):

- MuseStreamSession: one realtime WebSocket per utterance
  (wss://.../v1/asr/realtime, PUSH_TO_TALK). Audio is forwarded while
  the user is still speaking, so partial hypotheses arrive live (they
  replace the local tiny.en realtime model for `subscribe words`
  subscribers) and the final transcript lands almost immediately after
  the VAD tail closes.
- MuseFileClient: multipart POST of the buffered utterance WAV
  (https://.../v1/asr/transcribe). Used when streaming is disabled, and
  as the fallback whenever a stream session fails mid-utterance — the
  Transcriber keeps buffering frames regardless, so no audio is lost.

Auth: muse.api_key in config.yml, else $META_API_KEY. Keys come from
https://ai.developer.meta.com/ (API keys -> Create API key, "LLM|...").
"""

import io
import json
import logging
import os
import queue
import threading
import urllib.error
import urllib.request
import uuid
import wave
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Realtime endpoint wants ~80 ms raw PCM frames; at 16 kHz s16le that is
# 1280 samples = 2560 bytes per send.
_STREAM_BATCH_SECONDS = 0.08

_END = object()    # sentinel: flush + endStream, wait for the final transcript
_ABORT = object()  # sentinel: drop the session (utterance was too short)


class MuseAuthError(RuntimeError):
    pass


def resolve_api_key(muse_config) -> str:
    """Key from config, else $META_API_KEY — usually via the uncommitted
    .env beside config.yml. MUSE_API_KEY/MODEL_API_KEY are legacy aliases."""
    key = (
        getattr(muse_config, "api_key", None)
        or os.environ.get("META_API_KEY")
        or os.environ.get("MUSE_API_KEY")
        or os.environ.get("MODEL_API_KEY")
    )
    if not key:
        raise MuseAuthError(
            "Meta Model API key not found. Set muse.api_key in config.yml, or "
            "put META_API_KEY=... in a .env next to config.yml, or export "
            "META_API_KEY. Create one at https://ai.developer.meta.com/ "
            "(API keys -> Create API key; keys look like 'LLM|...')."
        )
    return key


def _audio_to_int16_bytes(audio: np.ndarray) -> bytes:
    """float32 [-1,1] mono -> signed 16-bit little-endian PCM bytes."""
    if audio.dtype == np.int16:
        return audio.tobytes()
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(_audio_to_int16_bytes(audio))
    return buf.getvalue()


def _pcm_encoding_for_rate(sample_rate: int) -> str:
    return "PCM_24KHZ" if sample_rate == 24000 else "PCM_16KHZ"


def _extract_text(payload) -> Optional[str]:
    """Pull transcript text out of an API response, tolerating shape drift."""
    if isinstance(payload, str):
        return payload.strip() or None
    if isinstance(payload, list):
        parts = [t for t in (_extract_text(item) for item in payload) if t]
        return " ".join(parts) or None
    if isinstance(payload, dict):
        for key in ("transcript", "text"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for key in ("segments", "results", "transcripts"):
            if key in payload:
                return _extract_text(payload[key])
    return None


class MuseFileClient:
    """Batch transcription via POST /v1/asr/transcribe (multipart WAV)."""

    def __init__(self, muse_config, api_key: str):
        self._cfg = muse_config
        self._key = api_key

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Optional[str]:
        request_json = {
            "model": self._cfg.model,
            "audioEncoding": "WAV",
            "mode": "PUSH_TO_TALK",
        }
        if self._cfg.language_bias:
            request_json["languageBias"] = list(self._cfg.language_bias)
        if self._cfg.keywords:
            request_json["keywords"] = list(self._cfg.keywords)

        boundary = "----perception-voice-" + uuid.uuid4().hex
        parts = [
            ("request", None, "application/json",
             json.dumps(request_json).encode()),
            ("audio", "utterance.wav", "audio/wav",
             _audio_to_wav_bytes(audio, sample_rate)),
        ]
        body = bytearray()
        for name, filename, ctype, data in parts:
            disposition = f'form-data; name="{name}"'
            if filename:
                disposition += f'; filename="{filename}"'
            body += (
                f"--{boundary}\r\n"
                f"Content-Disposition: {disposition}\r\n"
                f"Content-Type: {ctype}\r\n\r\n"
            ).encode()
            body += data + b"\r\n"
        body += f"--{boundary}--\r\n".encode()

        url = f"{self._cfg.transcribe_url}?sessionId=perception-{uuid.uuid4().hex[:12]}"
        req = urllib.request.Request(
            url,
            data=bytes(body),
            method="POST",
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._cfg.request_timeout) as resp:
                payload = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode()[:500]
            except Exception:
                pass
            if e.code == 429:
                logger.warning("Muse API rate limit hit (HTTP 429): %s", detail)
            else:
                logger.error("Muse transcribe failed (HTTP %s): %s", e.code, detail)
            return None
        except Exception as e:
            logger.error("Muse transcribe request error: %s", e)
            return None

        text = _extract_text(payload)
        if text is None:
            logger.warning("Muse transcribe: no transcript in response: %.500s", payload)
        return text


class MuseStreamSession:
    """
    One realtime WebSocket session for one utterance.

    Created on the audio callback thread when the VAD opens; feed() only
    enqueues (never blocks the realtime thread) while a worker thread
    connects, handshakes, and forwards audio in ~80 ms batches. A
    receiver thread surfaces partial transcripts through on_partial.

    finish() flushes, sends endStream, and returns the final transcript —
    or None on any failure, in which case the caller falls back to
    MuseFileClient with the locally buffered audio.
    """

    def __init__(self, muse_config, api_key: str, sample_rate: int,
                 on_partial: Optional[Callable[[str], None]] = None):
        self._cfg = muse_config
        self._key = api_key
        self._sample_rate = sample_rate
        self._on_partial = on_partial
        self._batch_bytes = int(sample_rate * _STREAM_BATCH_SECONDS) * 2

        # ~64s of 512-sample chunks; if the socket stalls this long, fail
        # the session and let the caller fall back to the file endpoint.
        self._q: queue.Queue = queue.Queue(maxsize=2000)
        self._final_text: Optional[str] = None
        self._partial_text: Optional[str] = None
        self._complete = threading.Event()
        self._done = threading.Event()
        self._failed = False

        threading.Thread(target=self._run, daemon=True,
                         name="muse-stream").start()

    def feed(self, chunk: np.ndarray) -> None:
        if self._failed or self._done.is_set():
            return
        try:
            self._q.put_nowait(chunk)
        except queue.Full:
            logger.warning("Muse stream backlog full, failing session")
            self._failed = True

    def finish(self, timeout: float = 30.0) -> Optional[str]:
        """End the stream and return the final transcript (None on failure)."""
        try:
            self._q.put_nowait(_END)
        except queue.Full:
            self._failed = True
        if not self._done.wait(timeout):
            logger.warning("Muse stream did not finish within %.0fs", timeout)
            return None
        if self._failed:
            return None
        text = self._final_text or self._partial_text
        return text.strip() if text and text.strip() else None

    def abort(self) -> None:
        """Discard the session (utterance too short); no transcript wanted."""
        try:
            self._q.put_nowait(_ABORT)
        except queue.Full:
            pass

    def _run(self) -> None:
        try:
            from websockets.sync.client import connect
        except ImportError:
            logger.error(
                "websockets package not installed; muse streaming unavailable "
                "(pip install websockets). Falling back to file transcription."
            )
            self._failed = True
            self._done.set()
            return

        url = f"{self._cfg.realtime_url}?sessionId=perception-{uuid.uuid4().hex[:12]}"
        handshake = {
            "authorization": {"accessToken": f"Bearer {self._key}"},
            "audioEncoding": _pcm_encoding_for_rate(self._sample_rate),
            "model": self._cfg.model,
            "mode": "PUSH_TO_TALK",
            "partialMode": "CUMULATIVE",
        }
        if self._cfg.language_bias:
            handshake["languageBias"] = list(self._cfg.language_bias)
        if self._cfg.keywords:
            handshake["keywords"] = list(self._cfg.keywords)

        try:
            with connect(url, open_timeout=15) as ws:
                ws.send(json.dumps(handshake))
                ws.recv(timeout=15)  # handshake ack (carries sessionId)

                receiver = threading.Thread(
                    target=self._receiver, args=(ws,), daemon=True,
                    name="muse-stream-recv",
                )
                receiver.start()

                pending = bytearray()
                while True:
                    item = self._q.get()
                    if item is _ABORT:
                        break
                    if item is _END:
                        if pending:
                            ws.send(bytes(pending))
                        ws.send(json.dumps({"type": "endStream"}))
                        if not self._complete.wait(self._cfg.request_timeout):
                            logger.warning("Muse stream: no completion event")
                        break
                    pending += _audio_to_int16_bytes(item)
                    if len(pending) >= self._batch_bytes:
                        ws.send(bytes(pending))
                        pending.clear()
        except Exception as e:
            logger.warning("Muse realtime stream error: %s", e)
            self._failed = True
        self._done.set()

    def _receiver(self, ws) -> None:
        try:
            while True:
                msg = ws.recv()
                if not isinstance(msg, str):
                    continue
                event = json.loads(msg)
                etype = event.get("type")
                if etype == "transcript":
                    text = event.get("transcript") or event.get("text") or ""
                    if event.get("final"):
                        # partialMode CUMULATIVE: last final wins
                        self._final_text = text
                    else:
                        self._partial_text = text
                        if self._on_partial and text:
                            try:
                                self._on_partial(text)
                            except Exception as e:
                                logger.debug("on_partial error: %s", e)
                elif etype in ("speechComplete", "sessionComplete"):
                    self._complete.set()
        except Exception:
            # Normal close after endStream also lands here
            self._complete.set()
