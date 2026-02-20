#!/bin/bash

echo "=== ACE-Step Music Generation API (Docker) ==="
echo ""

if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not found. Install Docker first."
    exit 1
fi

if ! command -v docker compose &> /dev/null && ! command -v docker-compose &> /dev/null; then
    echo "ERROR: Docker Compose not found."
    exit 1
fi

if ! nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. Make sure NVIDIA drivers and nvidia-container-toolkit are installed."
    echo "See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    echo ""
fi

echo "Building and starting services..."
docker compose up --build -d

echo ""
echo "Services started:"
echo "  API:    http://localhost:5000"
echo "  Docs:   http://localhost:5000/docs"
echo "  Health: http://localhost:5000/health"
echo ""
echo "Commands:"
echo "  View logs:    docker compose logs -f"
echo "  Worker logs:  docker compose logs -f worker"
echo "  Stop:         docker compose down"
echo "  Restart:      docker compose restart"
