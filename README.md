# ACE-Step Music Generation API

REST API сервис для генерации музыки с помощью модели ACE-Step 1.5.

## Требования

- **Docker** + Docker Compose
- **GPU (опционально)**: NVIDIA с VRAM >= 6 GB + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- **Python 3.11** (только для запуска без Docker)
- **Redis 7+** (только для запуска без Docker)

## Быстрый старт (Docker)

Один скрипт — три режима: CPU, GPU, GPU-ферма.

```bash
chmod +x deploy.sh
./deploy.sh
```

Или напрямую без меню:

```bash
./deploy.sh cpu     # Без видеокарты (тест/разработка)
./deploy.sh gpu     # Одна NVIDIA GPU
./deploy.sh farm    # Несколько GPU + мониторинг
./deploy.sh stop    # Остановить все
```

### 1. CPU — без видеокарты

Подходит для тестирования и разработки. Генерация медленная (5-15 мин/трек).

```bash
./deploy.sh cpu

# Или вручную:
docker compose -f docker-compose.cpu.yml up --build -d
```

### 2. GPU — одна видеокарта

Стандартный режим. Быстрая генерация (~30-60 сек/трек).

```bash
./deploy.sh gpu

# Или вручную:
docker compose -f docker-compose.gpu.yml up --build -d
```

### 3. GPU Farm — несколько видеокарт

Масштабируемая ферма с мониторингом через Flower.

```bash
./deploy.sh farm

# Или вручную с настройками:
docker compose -f docker-compose.farm.yml up --build -d --scale worker=4
```

Настройки фермы (через переменные окружения или `.env` файл):

| Переменная | По умолчанию | Описание |
|-----------|-------------|----------|
| `GPU_WORKERS` | 1 | Количество GPU-воркеров |
| `API_WORKERS` | 2 | Количество процессов uvicorn |
| `API_PORT` | 5000 | Порт API |
| `FLOWER_PORT` | 5555 | Порт Flower (мониторинг) |
| `WORKER_MEMORY` | 8G | Лимит RAM на воркер |
| `API_CPUS` | 2.0 | Лимит CPU для API |
| `API_MEMORY` | 2G | Лимит RAM для API |
| `HF_TOKEN` | — | Токен HuggingFace |

## Запуск без Docker

```bash
# 1. Установите Redis
sudo apt install redis-server   # Ubuntu/Debian
# или
brew install redis               # macOS

# 2. Полная автоматическая установка
chmod +x setup.sh
./setup.sh

# 3. Запуск
chmod +x start.sh
./start.sh
```

Или вручную:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e .
./start.sh
```

## Структура проекта

```
├── api/
│   └── main.py                # FastAPI приложение и эндпоинты
├── core/
│   ├── config.py              # Конфигурация (Pydantic Settings)
│   ├── models.py              # Загрузка модели ACE-Step и генерация
│   └── celery_app.py          # Конфигурация Celery
├── tasks/
│   └── generation_tasks.py    # Celery задачи (генерация, LoRA)
├── generated_audio/           # Выходные аудиофайлы
├── lora_models/               # LoRA адаптеры
├── deploy.sh                  # Единый скрипт развёртывания
├── docker-compose.cpu.yml     # Docker: CPU-режим
├── docker-compose.gpu.yml     # Docker: GPU-режим
├── docker-compose.farm.yml    # Docker: GPU-ферма
├── Dockerfile                 # Образ с CUDA 12.1
├── Dockerfile.cpu             # Образ без CUDA
├── pyproject.toml             # Зависимости Python
├── setup.sh                   # Установка без Docker
├── start.sh                   # Локальный запуск
└── README.md
```

## API Endpoints

### POST /generate — Генерация музыки

```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "energetic electronic dance music with heavy bass",
    "duration": 60,
    "lyrics": "[verse]\nFeel the beat\nMove your feet\n[chorus]\nDance all night",
    "num_steps": 8,
    "cfg_scale": 3.5
  }'
```

Ответ:
```json
{"task_id": "uuid-here", "status": "pending"}
```

### GET /status/{task_id} — Статус задачи

```bash
curl http://localhost:5000/status/{task_id}
```

Ответ:
```json
{"task_id": "uuid", "status": "success", "file_url": "/files/uuid.wav", "error": null}
```

Возможные статусы: `pending`, `processing`, `success`, `failed`.

### GET /files/{filename} — Скачать аудио

```bash
curl -O http://localhost:5000/files/{task_id}.wav
```

### POST /train/lora — Обучение LoRA адаптера

```bash
curl -X POST http://localhost:5000/train/lora \
  -F "style_name=my_style" \
  -F "audio_archive=@my_samples.zip"
```

ZIP-архив должен содержать 5-10 аудиофайлов (wav, mp3, flac, ogg, opus).

### GET /health — Проверка здоровья

```bash
curl http://localhost:5000/health
```

## Параметры генерации

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `prompt` | string | обязательный | Описание стиля музыки |
| `duration` | int | 120 | Длительность (10-600 сек) |
| `lyrics` | string | null | Текст песни с маркерами [verse], [chorus] и т.д. |
| `style` | string | null | Дополнительные теги стиля |
| `seed` | int | -1 | Сид (-1 = случайный) |
| `num_steps` | int | 8 | Количество шагов диффузии |
| `cfg_scale` | float | 3.5 | Шкала classifier-free guidance |
| `batch_size` | int | 1 | Количество вариаций (1-8) |
| `lora_id` | string | null | ID LoRA адаптера |

## Переменные окружения

| Переменная | По умолчанию | Описание |
|-----------|-------------|----------|
| `REDIS_URL` | redis://localhost:6379/0 | URL Redis |
| `MODEL_PATH` | acestep-v15-turbo | Путь/имя модели |
| `OUTPUT_DIR` | generated_audio | Директория для аудио |
| `LORA_DIR` | lora_models | Директория для LoRA |
| `HF_TOKEN` | — | Токен HuggingFace (если нужен) |
| `ACESTEP_INIT_LLM` | false | Инициализировать LLM хендлер |

## Архитектура

```
┌─────────────┐    ┌─────────┐    ┌──────────────────┐
│  FastAPI     │───>│  Redis  │<───│  Celery Worker   │
│  (API)       │    │ (broker)│    │  (GPU, ACE-Step) │
│  port 5000   │    │         │    │                  │
└─────────────┘    └─────────┘    └──────────────────┘
       │                                    │
       ▼                                    ▼
  /docs (Swagger)                  generated_audio/
```

### GPU-ферма

```
                    ┌─────────────┐
                    │  Flower     │ ← мониторинг (порт 5555)
                    │  Dashboard  │
                    └──────┬──────┘
                           │
┌─────────────┐    ┌───────┴───┐    ┌──────────────────┐
│  FastAPI     │───>│   Redis   │<───│  Worker 1 (GPU0) │
│  (N workers) │    │  (broker) │    ├──────────────────┤
│  port 5000   │    │           │<───│  Worker 2 (GPU1) │
└─────────────┘    └───────────┘    ├──────────────────┤
                                 <───│  Worker N (GPUn) │
                                     └──────────────────┘
```

- **FastAPI** — принимает HTTP-запросы, отдаёт файлы
- **Redis** — брокер сообщений для Celery
- **Celery Worker** — выполняет генерацию на GPU (1 воркер = 1 GPU)
- **Flower** — веб-панель мониторинга воркеров (только в Farm-режиме)
- **ACE-Step 1.5** — модель генерации музыки
