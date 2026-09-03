"""
Audio8-ASR-0.1B backend (tiny local open-weight ASR via transformers).

Clip-then-transcribe: the VAD pipeline already hands us finalized
utterance clips, which is exactly the shape this model wants. Clips
longer than max_clip_seconds (model cap: 30 s) are chunked and joined.

~0.32B params end-to-end (0.10B LM + Qwen3-ASR encoder); bf16 weights
fit in well under 2 GB of VRAM, so it can share a GPU with a large LLM
where whisper large-v3 cannot. 7 languages (en, zh, yue, fr, de, ja,
ko). No partial hypotheses — pair with the `realtime:` tiny.en decoder
if `subscribe words` partials are needed.

License note: CC-BY-NC-4.0 — fine for personal use, not for shipping
in a commercial product.
"""

import logging
import os
import tempfile
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class Audio8Engine:
    """Loads Audio8-ASR once and serializes transcribe() calls."""

    def __init__(self, audio8_config):
        self._cfg = audio8_config
        self._lock = threading.Lock()

        # Heavy imports stay local so local/muse backends never pay them
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._torch = torch
        device = audio8_config.device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available; Audio8 falling back to CPU")
            device = "cpu"
        self._device = device
        dtype = getattr(torch, audio8_config.dtype, None)
        if dtype is None or device == "cpu":
            dtype = torch.float32

        logger.info(f"Loading Audio8 model: {audio8_config.model} "
                    f"({device}, {audio8_config.dtype})")
        self._processor = AutoProcessor.from_pretrained(
            audio8_config.model, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            audio8_config.model,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation="eager",
        ).to(device)
        self._model.eval()

        if device == "cuda":
            mb = torch.cuda.memory_allocated() / (1024 ** 2)
            logger.info(f"Audio8 model loaded ({mb:.0f} MiB VRAM allocated)")
        else:
            logger.info("Audio8 model loaded (CPU)")

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Optional[str]:
        max_samples = int(self._cfg.max_clip_seconds * sample_rate)
        parts = []
        for start in range(0, len(audio), max_samples):
            text = self._transcribe_clip(audio[start:start + max_samples],
                                         sample_rate)
            if text:
                parts.append(text)
        return " ".join(parts).strip() or None

    def _transcribe_clip(self, audio: np.ndarray, sample_rate: int) -> str:
        # The model's chat template takes a wav *path*; round-trip through
        # a temp file (16-bit PCM via the muse helper, no soundfile dep).
        from perception_voice.muse import _audio_to_wav_bytes

        fd, path = tempfile.mkstemp(suffix=".wav", prefix="audio8-")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(_audio_to_wav_bytes(audio, sample_rate))

            conversation = [{
                "role": "user",
                "content": [
                    {"type": "audio", "path": path},
                    {"type": "text", "text": "Please transcribe this audio."},
                ],
            }]
            with self._lock:
                batch = self._processor.apply_chat_template(
                    conversation,
                    return_tensors="pt",
                    sampling_rate=sample_rate,
                    audio_padding="longest",
                    add_generation_prompt=True,
                    audio_max_length=self._cfg.max_clip_seconds * sample_rate,
                    text_kwargs={"padding": "longest", "truncation": True,
                                 "max_length": 1000},
                )
                batch = {k: v.to(self._device) if hasattr(v, "to") else v
                         for k, v in dict(batch).items()}
                with self._torch.inference_mode():
                    out = self._model.generate(
                        **batch,
                        max_new_tokens=self._cfg.max_new_tokens,
                        do_sample=False,
                    )
                prompt_len = int(batch["input_ids"].shape[1])
                return self._processor.decode(
                    out[0, prompt_len:], skip_special_tokens=True
                ).strip()
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
