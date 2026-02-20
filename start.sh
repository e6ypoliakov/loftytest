#!/bin/bash

export PYTHONPATH="$(pwd):$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false

if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. ACE-Step requires NVIDIA GPU with >= 6GB VRAM."
    echo "Generation will likely fail without GPU."
fi

echo ""

if ! command -v redis-server &> /dev/null; then
    echo "ERROR: redis-server not found. Install Redis first:"
    echo "  Ubuntu/Debian: sudo apt install redis-server"
    echo "  macOS: brew install redis"
    echo "  Or use Docker: docker run -d -p 6379:6379 redis:7-alpine"
    exit 1
fi

redis-server --daemonize yes --port 6379 2>/dev/null || true
sleep 1

echo "Starting Celery worker (GPU)..."
celery -A core.celery_app worker --loglevel=info --pool=solo --concurrency=1 &
CELERY_PID=$!

sleep 2

echo "Starting API server on http://0.0.0.0:5000 ..."
uvicorn api.main:app --host 0.0.0.0 --port 5000 &
API_PID=$!

echo ""
echo "=== ACE-Step Music Generation API ==="
echo "API:    http://localhost:5000"
echo "Docs:   http://localhost:5000/docs"
echo "Health: http://localhost:5000/health"
echo ""

cleanup() {
    echo "Shutting down..."
    kill $API_PID 2>/dev/null
    kill $CELERY_PID 2>/dev/null
    redis-cli shutdown 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

wait
