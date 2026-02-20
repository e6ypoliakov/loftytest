import os
import logging
import shutil
import threading
from typing import Any, Dict, Optional, Tuple

import soundfile as sf
import torch

from core.config import settings

logger = logging.getLogger(__name__)

_dit_handler = None
_llm_handler = None
_init_lock = threading.Lock()
_initialized = False


def _get_device() -> str:
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        logger.info(f"GPU detected: {gpu_name}, VRAM: {vram:.1f} GB")
        return "cuda"
    else:
        logger.warning("No CUDA GPU detected! ACE-Step requires GPU with VRAM >= 6GB.")
        logger.warning("Falling back to CPU (will be extremely slow or may fail).")
        return "cpu"


def load_models():
    global _dit_handler, _llm_handler, _initialized

    if _initialized:
        return _dit_handler, _llm_handler

    with _init_lock:
        if _initialized:
            return _dit_handler, _llm_handler

        try:
            from acestep.handler import AceStepHandler

            device = _get_device()
            config_path = settings.MODEL_PATH

            logger.info(f"Initializing ACE-Step handler on device={device}")

            _dit_handler = AceStepHandler()

            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            logger.info(f"Loading model: {config_path} (this may download models on first run)")
            status_msg, success = _dit_handler.initialize_service(
                project_root=project_root,
                config_path=config_path,
                device=device,
                offload_to_cpu=False,
            )
            if not success:
                logger.error(f"Model initialization failed: {status_msg}")
                raise RuntimeError(f"Model init failed: {status_msg}")

            logger.info(f"ACE-Step model loaded successfully: {status_msg}")

            if device == "cuda":
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                logger.info(f"GPU memory: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")

            _llm_handler = None
            if settings.ACESTEP_INIT_LLM:
                try:
                    from acestep.llm_inference import LLMHandler
                    _llm_handler = LLMHandler()
                    logger.info("LLM handler initialized")
                except Exception as e:
                    logger.warning(f"LLM handler not available: {e}")

            _initialized = True
            return _dit_handler, _llm_handler

        except ImportError as e:
            logger.error(f"ACE-Step import failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load ACE-Step model: {e}")
            raise


def _build_generation_params(params: Dict[str, Any]) -> Tuple:
    from acestep.inference import GenerationParams, GenerationConfig

    caption = params.get("prompt", params.get("caption", ""))
    lyrics = params.get("lyrics", "")
    style = params.get("style", "")

    if style and caption:
        caption = f"{style}, {caption}"
    elif style:
        caption = style

    seed = params.get("seed", -1)

    gen_params = GenerationParams(
        caption=caption,
        lyrics=lyrics if lyrics else "[Instrumental]",
        duration=float(params.get("duration", 120)),
        inference_steps=params.get("num_steps", 8),
        guidance_scale=params.get("cfg_scale", 3.5),
        seed=seed if seed and seed != -1 else -1,
        thinking=False,
    )

    gen_config = GenerationConfig(
        batch_size=params.get("batch_size", 1),
        use_random_seed=(seed == -1 or seed is None),
        audio_format="wav",
    )

    return gen_params, gen_config, caption


def _apply_lora(dit_handler, lora_path: str) -> None:
    try:
        logger.info(f"Loading LoRA adapter from: {lora_path}")
        lora_msg = dit_handler.load_lora(lora_path)
        if lora_msg.startswith("\u274c"):
            logger.warning(f"LoRA loading issue: {lora_msg}")
        else:
            logger.info(f"LoRA loaded: {lora_msg}")
    except Exception as e:
        logger.warning(f"Failed to load LoRA adapter: {e}")


def _save_result_audio(result, task_id: str, save_dir: str) -> str:
    output_filename = f"{task_id}.wav"
    output_path = os.path.join(save_dir, output_filename)

    if not result.audios:
        raise RuntimeError("No audio was generated")

    first_audio = result.audios[0]
    first_path = first_audio.get("path", "")

    if first_path and os.path.exists(first_path):
        if os.path.abspath(first_path) != os.path.abspath(output_path):
            shutil.move(first_path, output_path)
    else:
        tensor = first_audio.get("tensor")
        if tensor is None:
            raise RuntimeError("No audio data in generation result")
        sample_rate = first_audio.get("sample_rate", 48000)
        audio_np = tensor.cpu().numpy()
        if audio_np.ndim == 2:
            audio_np = audio_np.T
        sf.write(output_path, audio_np, sample_rate)

    return output_filename


def generate_music(params: Dict[str, Any], task_id: str, lora_path: Optional[str] = None) -> str:
    from acestep.inference import generate_music as ace_generate

    dit_handler, llm_handler = load_models()

    gen_params, gen_config, caption = _build_generation_params(params)

    if lora_path and os.path.exists(lora_path):
        _apply_lora(dit_handler, lora_path)

    save_dir = os.path.abspath(settings.OUTPUT_DIR)
    os.makedirs(save_dir, exist_ok=True)

    duration = params.get("duration", 120)
    logger.info(f"Starting generation for task {task_id}: caption='{caption[:80]}...', duration={duration}s")

    result = ace_generate(
        dit_handler=dit_handler,
        llm_handler=llm_handler,
        params=gen_params,
        config=gen_config,
        save_dir=save_dir,
    )

    if not result.success:
        raise RuntimeError(f"Generation failed: {result.error or result.status_message}")

    output_filename = _save_result_audio(result, task_id, save_dir)
    logger.info(f"Generation complete: {output_filename}")
    return output_filename
