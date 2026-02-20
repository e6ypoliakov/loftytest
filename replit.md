# ACE-Step Music Generation API

## Overview
REST API сервис для генерации музыки через модель ACE-Step 1.5. Проект предназначен для локального запуска на компьютере с NVIDIA GPU. FastAPI + Celery + Redis.

## Recent Changes
- 2026-02-20: Рефакторинг: удалены main.py и requirements.txt (дубликат pyproject.toml), core/models.py разбит на функции, ZIP-обработка вынесена из эндпоинта, Redis connection pool, упрощена очистка tmp-директорий, Dockerfile переведён на pyproject.toml.
- 2026-02-20: Полный аудит: GenerationResult.audios, DatasetBuilder.scan_directory, AudioSample без tags, pydantic-settings defaults, .opus вместо .m4a.
- 2026-02-20: Проект переделан для локального запуска с GPU. Docker Compose для полного стека.

## Architecture
- **FastAPI** serves REST endpoints on port 5000
- **Celery** handles async music generation and LoRA training tasks (GPU)
- **Redis** acts as message broker and result backend
- **ACE-Step 1.5** generates audio on NVIDIA GPU
- **Docker Compose** orchestrates Redis + API + GPU Worker
- **start.sh** for local run without Docker

## ACE-Step 1.5 API Notes
Key API details verified against actual library source:
- `GenerationResult.audios` — list of dicts with keys: "path", "tensor", "key", "sample_rate", "params"
- `AudioSample` fields: audio_path, filename, caption, genre, lyrics, custom_tag, labeled, duration (NO `tags` field)
- `DatasetBuilder.scan_directory(dir)` → `(samples, status)`. NO `add_sample()` method
- `preprocess_to_tensors(dit_handler, output_dir)` → `(output_paths, status)`. Only processes labeled=True samples
- `LoRATrainer.train_from_preprocessed(tensor_dir)` → yields `(step, loss, status_msg)`
- `AceStepHandler.load_lora(path)` → returns status string (doesn't raise)
- `AceStepHandler.initialize_service(...)` → returns `(status_msg, success_bool)`
- Supported audio formats: .wav, .mp3, .flac, .ogg, .opus (NO .m4a)

## Project Structure
```
api/main.py              - FastAPI app with all endpoints
core/config.py           - Pydantic settings (env vars)
core/models.py           - ACE-Step model loading & generation (CUDA)
core/celery_app.py       - Celery configuration
tasks/generation_tasks.py - Celery tasks (generate, train LoRA)
generated_audio/         - Output audio files (gitignored)
lora_models/             - LoRA adapter storage (gitignored)
checkpoints/             - Model weights (gitignored, auto-downloaded)
docker-compose.yml       - Full stack: Redis + API + GPU Worker
Dockerfile               - GPU worker image (CUDA 12.1)
pyproject.toml           - Single source of dependencies
setup.sh                 - Full setup + model download
start.sh                 - Local startup script
start_docker.sh          - Docker startup helper
README.md                - Documentation (Russian)
```

## Key Endpoints
- POST /generate - Submit music generation task
- GET /status/{task_id} - Check task status
- GET /files/{filename} - Download generated audio
- POST /train/lora - Submit LoRA training (5-10 audio files in ZIP)
- GET /docs - Swagger UI

## Environment Variables
- REDIS_URL (default: redis://localhost:6379/0)
- MODEL_PATH (default: acestep-v15-turbo)
- OUTPUT_DIR (default: generated_audio)
- HF_TOKEN (optional)
- LORA_DIR (default: lora_models)
- ACESTEP_INIT_LLM (default: false)

## User Preferences
- Language: Russian
- Target: local machine with NVIDIA GPU
