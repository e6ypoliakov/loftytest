# ACE-Step Music Generation API

REST API service for music generation using the ACE-Step 1.5 model.

## Architecture

- **FastAPI** — REST API server with Swagger UI
- **Celery** — async task queue for music generation
- **Redis** — message broker and result backend
- **ACE-Step 1.5** — music generation model (Turbo variant)

## Project Structure

```
├── api/
│   └── main.py           # FastAPI application and endpoints
├── core/
│   ├── config.py          # Configuration (Pydantic Settings)
│   ├── models.py          # ACE-Step model loading and generation
│   └── celery_app.py      # Celery configuration
├── tasks/
│   └── generation_tasks.py # Celery tasks (generation, LoRA training)
├── generated_audio/       # Output directory for generated tracks
├── lora_models/           # LoRA adapters directory
├── Dockerfile             # GPU worker Docker image
├── setup.sh               # Setup script
└── README.md
```

## API Endpoints

| Method | Path              | Description                        |
|--------|-------------------|------------------------------------|
| GET    | `/`               | Service info                       |
| GET    | `/health`         | Health check (Redis status)        |
| POST   | `/generate`       | Submit music generation task       |
| GET    | `/status/{id}`    | Check generation task status       |
| GET    | `/files/{name}`   | Download generated audio file      |
| POST   | `/train/lora`     | Submit LoRA training task          |
| GET    | `/docs`           | Swagger UI                         |

## Environment Variables

| Variable     | Default                    | Description                    |
|-------------|----------------------------|--------------------------------|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string        |
| `MODEL_PATH`| `acestep-v15-turbo`        | ACE-Step model variant         |
| `OUTPUT_DIR`| `generated_audio`          | Output directory               |
| `HF_TOKEN`  | (none)                     | Hugging Face token (optional)  |
| `LORA_DIR`  | `lora_models`              | LoRA adapters directory        |

## Running Locally

```bash
# 1. Start Redis
redis-server --daemonize yes

# 2. Start Celery worker
celery -A core.celery_app worker --loglevel=info --concurrency=1

# 3. Start API server
uvicorn api.main:app --host 0.0.0.0 --port 5000
```

## GPU Worker Deployment (Scaling)

For deploying workers on a GPU farm:

1. Set up a Redis instance accessible from all workers (e.g., Redis Cloud)
2. Set the `REDIS_URL` environment variable to point to your Redis instance
3. Build the Docker image:

```bash
docker build -t acestep-worker .
```

4. Run the container with GPU access:

```bash
docker run -d \
  --gpus all \
  -e REDIS_URL=redis://your-redis-host:6379/0 \
  -e HF_TOKEN=your_hf_token \
  acestep-worker
```

The worker will automatically connect to Redis and start processing generation tasks from the queue.

## Generation Example

```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Upbeat indie pop with jangly guitars",
    "duration": 90,
    "lyrics": "[Verse 1]\nWalking down the street\n[Chorus]\nWe are alive tonight"
  }'
```

Response:
```json
{"task_id": "abc123-...", "status": "pending"}
```

Check status:
```bash
curl http://localhost:5000/status/abc123-...
```
