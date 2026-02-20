import os
import logging
import shutil
import tempfile
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


def _find_lora_train_tmp_dir(audio_paths: List[str]) -> str | None:
    for path in audio_paths:
        parent = os.path.dirname(path)
        if parent.startswith("/tmp") and os.path.basename(parent).startswith("lora_train_"):
            return parent
        grandparent = os.path.dirname(parent)
        if grandparent.startswith("/tmp") and os.path.basename(grandparent).startswith("lora_train_"):
            return grandparent
    return None


@celery_app.task(bind=True, name="tasks.train_lora")
def train_lora_task(self, style_name: str, audio_files_paths: List[str]):
    scan_dir = None
    tensor_dir = None
    upload_tmp_dir = _find_lora_train_tmp_dir(audio_files_paths)

    try:
        self.update_state(state="PROGRESS", meta={"status": "loading_model"})

        from core.models import load_models
        from acestep.training.dataset_builder import DatasetBuilder
        from acestep.training.trainer import LoRATrainer
        from acestep.training.configs import LoRAConfig, TrainingConfig

        dit_handler, _ = load_models()

        lora_output_dir = os.path.join(settings.LORA_DIR, style_name)
        os.makedirs(lora_output_dir, exist_ok=True)

        self.update_state(state="PROGRESS", meta={"status": "preprocessing_audio"})

        scan_dir = tempfile.mkdtemp(prefix="lora_scan_")
        for audio_path in audio_files_paths:
            if os.path.exists(audio_path):
                dest = os.path.join(scan_dir, os.path.basename(audio_path))
                shutil.copy2(audio_path, dest)
                caption_file = dest.rsplit(".", 1)[0] + ".caption.txt"
                with open(caption_file, "w") as f:
                    f.write(style_name)

        builder = DatasetBuilder()
        builder.metadata.custom_tag = style_name
        samples, scan_status = builder.scan_directory(scan_dir)
        logger.info(f"Scan result: {scan_status}")

        if not samples:
            return {"status": "failed", "error": "No valid audio files found after scanning"}

        for sample in builder.samples:
            if not sample.labeled:
                sample.caption = style_name
                sample.labeled = True

        labeled_count = sum(1 for s in builder.samples if s.labeled)
        logger.info(f"Preprocessing {labeled_count} labeled audio files for LoRA training")

        tensor_dir = tempfile.mkdtemp(prefix="lora_tensors_")
        output_paths, preprocess_status = builder.preprocess_to_tensors(
            dit_handler=dit_handler,
            output_dir=tensor_dir,
        )
        logger.info(f"Preprocess result: {preprocess_status}")

        if not output_paths:
            return {"status": "failed", "error": f"Preprocessing failed: {preprocess_status}"}

        self.update_state(state="PROGRESS", meta={"status": "training"})

        lora_config = LoRAConfig(r=8, alpha=16, dropout=0.1)
        training_config = TrainingConfig(
            learning_rate=1e-4,
            batch_size=1,
            max_epochs=100,
            save_every_n_epochs=25,
            output_dir=lora_output_dir,
        )

        trainer = LoRATrainer(
            dit_handler=dit_handler,
            lora_config=lora_config,
            training_config=training_config,
        )

        last_step = 0
        last_loss = 0.0
        for step, loss, status_msg in trainer.train_from_preprocessed(tensor_dir):
            last_step = step
            last_loss = loss
            if step % 50 == 0:
                logger.info(f"LoRA training step {step}: loss={loss:.4f} - {status_msg}")
                self.update_state(state="PROGRESS", meta={
                    "status": "training",
                    "step": step,
                    "loss": loss,
                })

        logger.info(f"LoRA training complete: {last_step} steps, final loss={last_loss:.4f}")

        return {
            "status": "success",
            "lora_id": style_name,
            "lora_path": lora_output_dir,
            "steps": last_step,
            "final_loss": last_loss,
        }

    except ImportError as e:
        logger.error(f"ACE-Step training modules not available: {e}")
        return {"status": "failed", "error": f"Training modules not available: {e}"}
    except Exception as e:
        logger.error(f"LoRA training failed for style {style_name}: {e}")
        return {"status": "failed", "error": str(e)}
    finally:
        for d in (tensor_dir, scan_dir, upload_tmp_dir):
            if d and os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
