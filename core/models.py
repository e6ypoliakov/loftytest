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
    instrumental = params.get("instrumental", False)

    if style and caption:
        caption = f"{style}, {caption}"
    elif style:
        caption = style

    if instrumental:
        lyrics = "[Instrumental]"
    elif not lyrics:
        lyrics = "[Instrumental]"

    seed = params.get("seed", -1)

    gen_kwargs = {
        "caption": caption,
        "lyrics": lyrics,
        "duration": float(params.get("duration", 120)),
        "inference_steps": params.get("num_steps", 8),
        "guidance_scale": params.get("cfg_scale", 3.5),
        "seed": seed if seed and seed != -1 else -1,
        "thinking": params.get("thinking", False),
        "instrumental": instrumental,
    }

    task_type = params.get("task_type")
    if task_type:
        gen_kwargs["task_type"] = task_type

    reference_audio = params.get("reference_audio")
    if reference_audio:
        gen_kwargs["reference_audio"] = reference_audio

    src_audio = params.get("src_audio")
    if src_audio:
        gen_kwargs["src_audio"] = src_audio

    vocal_language = params.get("vocal_language")
    if vocal_language:
        gen_kwargs["vocal_language"] = vocal_language

    bpm = params.get("bpm")
    if bpm is not None:
        gen_kwargs["bpm"] = bpm

    keyscale = params.get("keyscale")
    if keyscale:
        gen_kwargs["keyscale"] = keyscale

    timesignature = params.get("timesignature")
    if timesignature:
        gen_kwargs["timesignature"] = timesignature

    use_adg = params.get("use_adg")
    if use_adg is not None:
        gen_kwargs["use_adg"] = use_adg

    cfg_interval_start = params.get("cfg_interval_start")
    if cfg_interval_start is not None:
        gen_kwargs["cfg_interval_start"] = cfg_interval_start

    cfg_interval_end = params.get("cfg_interval_end")
    if cfg_interval_end is not None:
        gen_kwargs["cfg_interval_end"] = cfg_interval_end

    shift = params.get("shift")
    if shift is not None:
        gen_kwargs["shift"] = shift

    infer_method = params.get("infer_method")
    if infer_method:
        gen_kwargs["infer_method"] = infer_method

    repainting_start = params.get("repainting_start")
    if repainting_start is not None:
        gen_kwargs["repainting_start"] = repainting_start

    repainting_end = params.get("repainting_end")
    if repainting_end is not None:
        gen_kwargs["repainting_end"] = repainting_end

    audio_cover_strength = params.get("audio_cover_strength")
    if audio_cover_strength is not None:
        gen_kwargs["audio_cover_strength"] = audio_cover_strength

    lm_temperature = params.get("lm_temperature")
    if lm_temperature is not None:
        gen_kwargs["lm_temperature"] = lm_temperature

    lm_top_p = params.get("lm_top_p")
    if lm_top_p is not None:
        gen_kwargs["lm_top_p"] = lm_top_p

    lm_top_k = params.get("lm_top_k")
    if lm_top_k is not None:
        gen_kwargs["lm_top_k"] = lm_top_k

    lm_max_tokens = params.get("lm_max_tokens")
    if lm_max_tokens is not None:
        gen_kwargs["lm_max_tokens"] = lm_max_tokens

    gen_params = GenerationParams(**gen_kwargs)

    audio_format = params.get("audio_format", "wav")

    gen_config = GenerationConfig(
        batch_size=params.get("batch_size", 1),
        use_random_seed=(seed == -1 or seed is None),
        audio_format=audio_format,
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
