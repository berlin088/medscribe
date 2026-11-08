"""Speech recognition: NVIDIA Parakeet TDT 0.6B v2, run locally on CPU via onnx-asr."""
import os
import threading

import numpy as np

_model = None
_lock = threading.Lock()


def load_model():
    """Load once. First run downloads the model from Hugging Face (~600 MB, cached after)."""
    global _model
    if _model is None:
        import onnx_asr

        quant = os.getenv("PARAKEET_QUANT") or None
        _model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2", quantization=quant)
    return _model


def transcribe(pcm16k: np.ndarray) -> str:
    """pcm16k: mono float32 in [-1, 1] at 16 kHz (what the browser VAD emits)."""
    model = load_model()
    with _lock:  # one inference at a time; keeps CPU usage predictable
        text = model.recognize(pcm16k.astype(np.float32))
    return (text or "").strip()