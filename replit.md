# ACE-Step Music Generation API

## Overview
REST API service for music generation using the ACE-Step 1.5 model. Built with FastAPI, Celery, and Redis for async task processing.

## Recent Changes
- 2026-02-20: Initial project setup with full API, Celery workers, and Redis integration.

## Architecture
- **FastAPI** serves REST endpoints on port 5000
- **Celery** handles async music generation and LoRA training tasks
- **Redis** acts as message broker and result backend
- **ACE-Step 1.5** (or mock handler) generates audio
- **start.sh** orchestrates Redis, Celery worker, and Uvicorn startup

## Project Structure
```
api/main.py            - FastAPI app with all endpoints
core/config.py         - Pydantic settings (env vars)
core/models.py         - ACE-Step model loading & generation
core/celery_app.py     - Celery configuration
tasks/generation_tasks.py - Celery tasks (generate, train LoRA)
generated_audio/       - Output audio files
lora_models/           - LoRA adapter storage
Dockerfile             - GPU worker image for scaling
start.sh               - Startup script (Redis + Celery + Uvicorn)
```

## Key Endpoints
- POST /generate - Submit music generation task
- GET /status/{task_id} - Check task status
- GET /files/{filename} - Download generated audio
- POST /train/lora - Submit LoRA training
- GET /docs - Swagger UI

## Environment Variables
- REDIS_URL (default: redis://localhost:6379/0)
- MODEL_PATH (default: acestep-v15-turbo)
- OUTPUT_DIR (default: generated_audio)
- HF_TOKEN (optional)
- LORA_DIR (default: lora_models)
