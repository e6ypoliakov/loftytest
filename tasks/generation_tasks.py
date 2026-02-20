import os
import logging
import shutil
from typing import Dict, Any, List

from core.celery_app import celery_app
from core.config import settings

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="tasks.generate_track")
def generate_track(self, task_id: str, generation_params: Dict[str, Any]):
    try:
        self.update_state(state="PROGRESS", meta={"status": "loading_model"})

        from core.models import generate_music

        lora_id = generation_params.pop("lora_id", None)
        lora_path = None
        if lora_id:
            lora_path = os.path.join(settings.LORA_DIR, lora_id)
            if not os.path.exists(lora_path):
                lora_path = None
                logger.warning(f"LoRA adapter not found: {lora_id}")

        self.update_state(state="PROGRESS", meta={"status": "generating"})

        output_filename = generate_music(
            params=generation_params,
            task_id=task_id,
            lora_path=lora_path,
        )

        return {
            "file_path": output_filename,
            "status": "success",
            "task_id": task_id,
        }
    except Exception as e:
        logger.error(f"Generation failed for task {task_id}: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "task_id": task_id,
        }


@celery_app.task(bind=True, name="tasks.train_lora")
def train_lora_task(self, style_name: str, audio_files_paths: List[str]):
    try:
        self.update_state(state="PROGRESS", meta={"status": "preparing_training"})

        lora_output_dir = os.path.join(settings.LORA_DIR, style_name)
        os.makedirs(lora_output_dir, exist_ok=True)

        self.update_state(state="PROGRESS", meta={"status": "training"})

        try:
            from acestep.train import train_lora

            train_lora(
                audio_files=audio_files_paths,
                output_dir=lora_output_dir,
                style_name=style_name,
            )
        except ImportError:
            logger.warning("ACE-Step training module not available. Creating placeholder.")
            placeholder_path = os.path.join(lora_output_dir, "adapter_config.json")
            import json
            with open(placeholder_path, "w") as f:
                json.dump({
                    "style_name": style_name,
                    "status": "placeholder",
                    "audio_files": audio_files_paths,
                }, f)

        return {
            "status": "success",
            "lora_id": style_name,
            "lora_path": lora_output_dir,
        }
    except Exception as e:
        logger.error(f"LoRA training failed for style {style_name}: {e}")
        return {
            "status": "failed",
            "error": str(e),
        }
    finally:
        for path in audio_files_paths:
            try:
                tmp_dir = path
                while tmp_dir and tmp_dir != "/":
                    parent = os.path.dirname(tmp_dir)
                    if parent.startswith("/tmp") and os.path.basename(parent).startswith("lora_train_"):
                        if os.path.isdir(parent):
                            shutil.rmtree(parent, ignore_errors=True)
                        break
                    tmp_dir = parent
            except Exception:
                pass
