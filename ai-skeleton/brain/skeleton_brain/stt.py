"""Speech-to-text engines. All take 16 kHz mono float32 and return plain text."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class SttEngine:
    name = "base"

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:  # pragma: no cover
        raise NotImplementedError


class ParakeetOnnx(SttEngine):
    """NVIDIA Parakeet TDT 0.6B v3 through onnx-asr (int8, CPU). ~36x real time on a fast CPU."""

    def __init__(self, model: str, model_dir: Path, quantization: str = "int8"):
        import onnx_asr

        if not model_dir.is_dir():
            raise FileNotFoundError(f"{model_dir} is missing - run: skeleton-brain fetch-models")
        self.model = onnx_asr.load_model(model, model_dir, quantization=quantization or None,
                                         providers=["CPUExecutionProvider"])
        self.name = f"parakeet-v3/{quantization or 'fp32'}/cpu"

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if audio.size < sample_rate // 10:  # under 0.1 s: nothing to hear
            return ""
        return str(self.model.recognize(audio.astype(np.float32), sample_rate=sample_rate)).strip()


class FakeStt(SttEngine):
    """For tests and porch development: always 'hears' the same thing if there was any audio."""

    name = "fake"

    def __init__(self, text: str = "trick or treat"):
        self.text = text

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        return self.text if audio.size else ""


def fetch_parakeet(model: str, model_dir: Path, quantization: str = "int8") -> None:
    """Download the model files into model_dir (onnx-asr pulls from Hugging Face when the dir is absent)."""
    import onnx_asr

    if model_dir.is_dir() and any(model_dir.glob("*.onnx")):
        return
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    onnx_asr.load_model(model, model_dir, quantization=quantization or None)
