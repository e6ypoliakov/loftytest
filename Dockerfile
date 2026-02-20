FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    curl \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY requirements.txt .

RUN uv pip install --system -r requirements.txt

RUN uv pip install --system torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN uv pip install --system git+https://github.com/ace-step/ACE-Step-1.5.git

COPY . .

RUN mkdir -p generated_audio lora_models

ENV REDIS_URL=redis://redis:6379/0
ENV MODEL_PATH=acestep-v15-turbo
ENV OUTPUT_DIR=generated_audio
ENV LORA_DIR=lora_models
ENV TOKENIZERS_PARALLELISM=false

EXPOSE 5000

CMD ["celery", "-A", "core.celery_app", "worker", "--loglevel=info", "--pool=solo", "--concurrency=1"]
