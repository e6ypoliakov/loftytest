# ACE-Step Music Generation API

REST API сервис для генерации музыки с помощью модели ACE-Step 1.5.

## Требования

- **GPU**: NVIDIA с VRAM >= 6 GB (рекомендуется >= 8 GB)
- **NVIDIA Driver**: >= 530
- **CUDA**: 12.1+
- **Python**: 3.11
- **Redis**: 7+
- **Docker** (опционально): Docker + nvidia-container-toolkit

## Быстрый старт (Docker — рекомендуется)

Самый простой способ запуска — через Docker Compose. Он автоматически поднимет Redis, API-сервер и GPU-воркер.

```bash
# 1. Убедитесь, что установлен nvidia-container-toolkit:
#    https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# 2. Запуск
chmod +x start_docker.sh
./start_docker.sh

# 3. Проверка
curl http://localhost:5000/health
```

API будет доступен по адресу `http://localhost:5000`.
Документация Swagger: `http://localhost:5000/docs`.

## Запуск без Docker

```bash
# 1. Установите Redis
sudo apt install redis-server   # Ubuntu/Debian
# или
brew install redis               # macOS

# 2. Установите PyTorch с CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Установите ACE-Step 1.5
pip install git+https://github.com/ace-step/ACE-Step-1.5.git

# 4. Установите зависимости API
pip install -r requirements.txt

# 5. Запуск
chmod +x start.sh
./start.sh
```

Или используйте автоматический скрипт установки:
```bash
chmod +x setup.sh
./setup.sh
```

## Структура проекта

```
├── api/
│   └── main.py              # FastAPI приложение и эндпоинты
├── core/
│   ├── config.py             # Конфигурация (Pydantic Settings)
│   ├── models.py             # Загрузка модели ACE-Step и генерация
│   └── celery_app.py         # Конфигурация Celery
├── tasks/
│   └── generation_tasks.py   # Celery задачи (генерация, LoRA)
├── generated_audio/          # Выходные аудиофайлы
├── lora_models/              # LoRA адаптеры
├── docker-compose.yml        # Docker Compose для полного стека
├── Dockerfile                # Docker образ с GPU
├── requirements.txt          # Python зависимости
├── start.sh                  # Локальный запуск
├── start_docker.sh           # Запуск через Docker
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

- **FastAPI** — принимает HTTP-запросы, отдаёт файлы
- **Redis** — брокер сообщений для Celery
- **Celery Worker** — выполняет генерацию на GPU
- **ACE-Step 1.5** — модель генерации музыки

## Масштабирование (GPU-ферма)

Для деплоя воркеров на GPU-ферме:

1. Запустите Redis, доступный из сети (например, Redis Cloud):
```bash
# Или свой Redis-сервер с открытым портом
redis-server --bind 0.0.0.0 --port 6379
```

2. Установите переменную окружения `REDIS_URL`:
```bash
export REDIS_URL=redis://your-redis-host:6379/0
```

3. Соберите Docker-образ:
```bash
docker build -t acestep-worker .
```

4. Запустите контейнер на GPU-сервере:
```bash
docker run -d \
  --gpus all \
  -e REDIS_URL=redis://your-redis-host:6379/0 \
  -v ./generated_audio:/app/generated_audio \
  acestep-worker
```

5. Для масштабирования — запустите несколько воркеров (1 воркер = 1 GPU):
```bash
# Docker Compose
docker compose up --scale worker=2

# Без Docker — дополнительный воркер
celery -A core.celery_app worker --loglevel=info --pool=solo --concurrency=1 &
```

API-сервер может работать на отдельной машине без GPU.
Все воркеры подключаются к общему Redis и обрабатывают задачи из очереди.
