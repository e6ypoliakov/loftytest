#!/bin/bash

export PYTHONPATH="/home/runner/workspace:$PYTHONPATH"

redis-server --daemonize yes --port 6379 2>/dev/null || true

sleep 1

celery -A core.celery_app worker --loglevel=info --concurrency=1 &

sleep 2

exec uvicorn api.main:app --host 0.0.0.0 --port 5000 --reload
