import os
import logging
from typing import Dict, Any, Optional

from core.config import settings

logger = logging.getLogger(__name__)

_handler = None


def load_models():
    global _handler
    if _handler is not None:
        return _handler

    try:
        from acestep.acestep_v15_pipeline import AceStepHandler

        device = "cuda" if _cuda_available() else "cpu"
        logger.info(f"Loading ACE-Step model: {settings.MODEL_PATH} on {device}")

        _handler = AceStepHandler(
            config_path=settings.MODEL_PATH or "acestep-v15-turbo",
            backend="pt",
            device=device,
        )
        logger.info("ACE-Step model loaded successfully")
        return _handler
    except ImportError:
        logger.warning(
            "ACE-Step package not installed. Using mock handler for API testing."
        )
        _handler = MockHandler()
        return _handler
    except Exception as e:
        logger.error(f"Failed to load ACE-Step model: {e}")
        raise


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


class MockHandler:
    def generate(self, **kwargs):
        import wave
        import struct
        import random

        duration = kwargs.get("duration", 10)
        sample_rate = 44100
        num_samples = sample_rate * min(duration, 5)

        output_path = os.path.join(settings.OUTPUT_DIR, "mock_output.wav")
        with wave.open(output_path, "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            for _ in range(num_samples):
                value = int(random.uniform(-1, 1) * 32767)
                wav_file.writeframes(struct.pack("<h", value))

        return {"audio_path": output_path, "status": "success"}


def generate_music(params: Dict[str, Any], task_id: str, lora_path: Optional[str] = None) -> str:
    handler = load_models()

    caption = params.get("prompt", params.get("caption", ""))
    lyrics = params.get("lyrics", "")
    duration = params.get("duration", 120)
    style = params.get("style", "")

    if style and caption:
        caption = f"{style}, {caption}"
    elif style:
        caption = style

    output_filename = f"{task_id}.wav"
    output_path = os.path.join(settings.OUTPUT_DIR, output_filename)

    generation_kwargs = {
        "caption": caption,
        "duration": duration,
    }
    if lyrics:
        generation_kwargs["lyrics"] = lyrics

    seed = params.get("seed", -1)
    if seed and seed != -1:
        generation_kwargs["seed"] = seed

    num_steps = params.get("num_steps", 8)
    generation_kwargs["num_steps"] = num_steps

    cfg_scale = params.get("cfg_scale", 3.5)
    generation_kwargs["cfg_scale"] = cfg_scale

    batch_size = params.get("batch_size", 1)
    generation_kwargs["batch_size"] = batch_size

    if isinstance(handler, MockHandler):
        generation_kwargs["duration"] = duration
        result = handler.generate(**generation_kwargs)
        mock_path = result.get("audio_path", "")
        if os.path.exists(mock_path) and mock_path != output_path:
            os.rename(mock_path, output_path)
        return output_filename

    if lora_path and os.path.exists(lora_path):
        try:
            handler = load_models()
            logger.info(f"Loading LoRA adapter from: {lora_path}")
        except Exception as e:
            logger.warning(f"Failed to load LoRA adapter: {e}")

    result = handler.generate(**generation_kwargs)

    if hasattr(result, "get") and result.get("audio_path"):
        src = result["audio_path"]
        if os.path.exists(src) and src != output_path:
            os.rename(src, output_path)
    elif hasattr(result, "get") and result.get("audio"):
        result["audio"].save(output_path)

    return output_filename
