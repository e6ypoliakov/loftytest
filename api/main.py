import os
import uuid
import zipfile
import shutil
import tempfile
import logging
from enum import Enum
from typing import List, Optional

import mimetypes
import redis as redis_lib
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from core.config import settings
from core.celery_app import celery_app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".opus"}
MIN_LORA_FILES = 5
MAX_LORA_FILES = 10

_redis_pool = redis_lib.ConnectionPool.from_url(settings.REDIS_URL)

API_DESCRIPTION = """
## Генерация музыки с помощью ACE-Step 1.5

Этот API позволяет генерировать музыку, отслеживать статус задач и обучать собственные стили (LoRA).

### Как пользоваться

**1. Отправьте запрос на генерацию:**
```bash
curl -X POST http://localhost:5000/generate \\
  -H "Content-Type: application/json" \\
  -d '{"prompt": "energetic electronic dance music", "duration": 60}'
```

**2. Проверьте статус по task_id:**
```bash
curl http://localhost:5000/status/{task_id}
```

**3. Скачайте готовый файл:**
```bash
curl -O http://localhost:5000/files/{filename}
```

### Типы задач (task_type)
| Тип | Описание | Обязательные поля |
|-----|----------|-------------------|
| `text2music` | Создание трека по описанию (по умолчанию) | `prompt` |
| `cover` | Перенос стиля на исходное аудио | `src_audio`, `prompt` |
| `repaint` | Перегенерация фрагмента трека | `src_audio`, `repainting_start/end` |
| `lego` | Генерация дорожки поверх аудио | `src_audio`, `prompt` |
| `vocal2bgm` | Аккомпанемент под вокал | `src_audio` |
| `retake` | Повторная генерация с другим сидом | `prompt` |

### Группы параметров
- **Основные:** `prompt`, `task_type`, `duration`, `lyrics`, `instrumental`, `style`
- **Метаданные:** `vocal_language`, `bpm`, `keyscale`, `timesignature`
- **Диффузия:** `num_steps`, `cfg_scale`, `seed`, `use_adg`, `cfg_interval_start/end`, `shift`, `infer_method`
- **Задачи с аудио:** `src_audio`, `reference_audio`, `repainting_start/end`, `audio_cover_strength`
- **LLM (thinking):** `thinking`, `lm_temperature`, `lm_top_p`, `lm_top_k`, `lm_max_tokens`
- **Вывод:** `batch_size`, `audio_format`, `lora_id`

### Выходные форматы
`.wav` (без сжатия) · `.mp3` (компактный) · `.flac` (без потерь)

### Статусы задач
| Статус | Описание |
|--------|----------|
| `pending` | Задача в очереди |
| `processing` | Идёт генерация |
| `success` | Готово, файл доступен |
| `failed` | Ошибка, см. поле `error` |
"""

TAGS_METADATA = [
    {
        "name": "Генерация",
        "description": "Создание музыки и проверка статуса задач. "
        "Отправьте промпт — получите уникальный аудиотрек.",
    },
    {
        "name": "Файлы",
        "description": "Скачивание готовых аудиофайлов по имени.",
    },
    {
        "name": "Обучение LoRA",
        "description": "Обучение собственного стиля на ваших аудиозаписях (5–10 файлов в ZIP). "
        "После обучения используйте `lora_id` в запросе генерации.",
    },
    {
        "name": "Состояние",
        "description": "Проверка работоспособности API, Redis и модели.",
    },
]

app = FastAPI(
    title="ACE-Step Music Generation API",
    description=API_DESCRIPTION,
    version="1.0.0",
    openapi_tags=TAGS_METADATA,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TaskType(str, Enum):
    TEXT2MUSIC = "text2music"
    COVER = "cover"
    REPAINT = "repaint"
    LEGO = "lego"
    VOCAL2BGM = "vocal2bgm"
    RETAKE = "retake"


class AudioFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"


class InferMethod(str, Enum):
    ODE = "ode"
    SDE = "sde"


class GenerationRequest(BaseModel):
    prompt: str = Field(
        ...,
        description="Описание стиля музыки. Чем подробнее — тем точнее результат. "
        "Тип: строка, обязательное поле",
        json_schema_extra={"examples": ["energetic electronic dance music with heavy bass and synth leads"]},
    )
    task_type: Optional[TaskType] = Field(
        default=TaskType.TEXT2MUSIC,
        description="Тип задачи генерации. "
        "Допустимые значения: text2music | cover | repaint | lego | vocal2bgm | retake. "
        "По умолчанию: text2music. "
        "text2music — создание трека по текстовому описанию. "
        "cover — перенос стиля на исходное аудио (нужен src_audio). "
        "repaint — перегенерация фрагмента трека (нужен src_audio + repainting_start/end). "
        "lego — генерация отдельной дорожки поверх исходного аудио (нужен src_audio). "
        "vocal2bgm — создание аккомпанемента под вокал (нужен src_audio). "
        "retake — повторная генерация с другим сидом",
    )
    duration: int = Field(
        default=120,
        ge=10,
        le=600,
        description="Длительность трека в секундах. "
        "Диапазон: 10–600. По умолчанию: 120 (2 минуты)",
    )
    lyrics: Optional[str] = Field(
        default=None,
        description="Текст песни с маркерами структуры: [verse], [chorus], [bridge], [intro], [outro]. "
        "Тип: строка или null. По умолчанию: null (инструментал). "
        "Если не указан — генерируется инструментал",
        json_schema_extra={"examples": ["[verse]\nFeel the rhythm in your soul\n[chorus]\nDance all night long"]},
    )
    instrumental: Optional[bool] = Field(
        default=False,
        description="Генерировать только инструментал (без вокала). "
        "Допустимые значения: true | false. По умолчанию: false. "
        "Если true — поле lyrics игнорируется",
    )
    style: Optional[str] = Field(
        default=None,
        description="Дополнительные теги стиля через запятую. "
        "Тип: строка или null. По умолчанию: null. "
        "Уточняют жанр и настроение",
        json_schema_extra={"examples": ["electronic, dance, upbeat, 128bpm"]},
    )
    reference_audio: Optional[str] = Field(
        default=None,
        description="Референсное аудио в формате Base64. "
        "Тип: строка (Base64) или null. По умолчанию: null. "
        "Модель постарается создать трек похожего звучания",
    )
    src_audio: Optional[str] = Field(
        default=None,
        description="Исходное аудио в формате Base64. "
        "Тип: строка (Base64) или null. По умолчанию: null. "
        "Обязательно для задач cover, repaint, lego, vocal2bgm. "
        "Это аудио, которое модель будет трансформировать",
    )
    lora_id: Optional[str] = Field(
        default=None,
        description="ID обученного LoRA-адаптера (имя стиля из /train/lora). "
        "Тип: строка или null. По умолчанию: null. "
        "Применяет ваш собственный стиль к генерации",
        json_schema_extra={"examples": ["my_lofi_style"]},
    )

    vocal_language: Optional[str] = Field(
        default=None,
        description="Язык вокала. "
        "Допустимые значения: en, zh, ru, es, ja, de, fr, pt, it, ko, ar, tr, nl, sv, pl, id, th, vi, he, fi. "
        "По умолчанию: null (автоопределение). "
        "Топ-10 языков дают лучшее качество",
        json_schema_extra={"examples": ["en"]},
    )
    bpm: Optional[int] = Field(
        default=None,
        ge=40,
        le=300,
        description="Темп в ударах в минуту (BPM). "
        "Диапазон: 40–300. По умолчанию: null (автоопределение). "
        "Ориентиры: 60–80 баллады, 100–120 поп, 120–140 танцевальная, 140–180 драм-н-бейс",
    )
    keyscale: Optional[str] = Field(
        default=None,
        description="Тональность трека. "
        "Допустимые значения: нота + лад, например: C major, A minor, F# minor, Bb major, D dorian. "
        "По умолчанию: null (автоопределение)",
        json_schema_extra={"examples": ["C major"]},
    )
    timesignature: Optional[str] = Field(
        default=None,
        description="Музыкальный размер. "
        "Допустимые значения: 2/4, 3/4, 4/4, 5/4, 6/8, 7/8, 12/8. "
        "По умолчанию: null (обычно 4/4)",
        json_schema_extra={"examples": ["4/4"]},
    )

    seed: Optional[int] = Field(
        default=-1,
        ge=-1,
        le=2147483647,
        description="Сид генерации для воспроизводимости результата. "
        "Диапазон: -1–2147483647. По умолчанию: -1 (случайный). "
        "Одинаковый сид + одинаковые параметры = одинаковый результат",
    )
    num_steps: Optional[int] = Field(
        default=8,
        ge=1,
        le=100,
        description="Количество шагов диффузии. "
        "Диапазон: 1–100. По умолчанию: 8. "
        "Turbo-модель: рекомендуется 8. SFT-модель: рекомендуется 50. "
        "Больше шагов = выше качество, но дольше генерация",
    )
    cfg_scale: Optional[float] = Field(
        default=3.5,
        ge=0.0,
        le=15.0,
        description="Сила следования промпту (classifier-free guidance). "
        "Диапазон: 0.0–15.0. По умолчанию: 3.5. Рекомендуется: 2.0–7.0. "
        "Выше = точнее следует описанию, но может звучать менее естественно",
    )
    use_adg: Optional[bool] = Field(
        default=False,
        description="Включить Advanced Dynamic Guidance — адаптивное управление генерацией. "
        "Допустимые значения: true | false. По умолчанию: false. "
        "Может улучшить качество при высоких значениях cfg_scale",
    )
    cfg_interval_start: Optional[float] = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Начало интервала применения CFG по шагам диффузии. "
        "Диапазон: 0.0–1.0. По умолчанию: 0.0 (с самого начала). "
        "Позволяет применять CFG не на всех шагах. Должен быть < cfg_interval_end",
    )
    cfg_interval_end: Optional[float] = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Конец интервала применения CFG по шагам диффузии. "
        "Диапазон: 0.0–1.0. По умолчанию: 1.0 (до конца). "
        "Значения < 1.0 отключают CFG на последних шагах, делая звук естественнее",
    )
    shift: Optional[float] = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Фактор сдвига таймстепов диффузии (новое в v1.5). "
        "Диапазон: 0.1–10.0. По умолчанию: 1.0. "
        "Влияет на распределение шумоподавления. "
        "Значения > 1.0 сдвигают процесс к более чистым шагам",
    )
    infer_method: Optional[InferMethod] = Field(
        default=InferMethod.ODE,
        description="Метод диффузионного вывода (новое в v1.5). "
        "Допустимые значения: ode | sde. По умолчанию: ode. "
        "ode — обыкновенное дифф. уравнение (быстрее, стабильнее). "
        "sde — стохастическое дифф. уравнение (разнообразнее, но менее предсказуемо)",
    )
    batch_size: Optional[int] = Field(
        default=1,
        ge=1,
        le=8,
        description="Количество вариаций за один запрос. "
        "Диапазон: 1–8. По умолчанию: 1. "
        "Каждая вариация — уникальный трек. Больше = дольше и больше VRAM",
    )
    audio_format: Optional[AudioFormat] = Field(
        default=AudioFormat.WAV,
        description="Формат выходного аудиофайла. "
        "Допустимые значения: wav | mp3 | flac. По умолчанию: wav. "
        "wav — без сжатия, макс. качество. "
        "mp3 — компактный, с потерями. "
        "flac — без потерь, компактнее wav",
    )

    repainting_start: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=600.0,
        description="Начало фрагмента для перегенерации (в секундах). "
        "Диапазон: 0.0–600.0. По умолчанию: null. "
        "Используется с задачами repaint и lego. "
        "Пример: 10.0 — начать перегенерацию с 10-й секунды",
    )
    repainting_end: Optional[float] = Field(
        default=None,
        ge=-1.0,
        le=600.0,
        description="Конец фрагмента для перегенерации (в секундах). "
        "Диапазон: -1.0–600.0. По умолчанию: null. "
        "Значение -1 = до конца трека. "
        "Пример: 20.0 — перегенерировать до 20-й секунды",
    )
    audio_cover_strength: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Сила трансформации для задачи cover. "
        "Диапазон: 0.0–1.0. По умолчанию: null (1.0). "
        "0.0 = почти не менять оригинал, 1.0 = максимальная трансформация",
    )

    thinking: Optional[bool] = Field(
        default=False,
        description="Включить режим планирования LLM. "
        "Допустимые значения: true | false. По умолчанию: false. "
        "Модель автоматически генерирует метаданные "
        "(BPM, тональность, структуру) на основе промпта. Требует инициализации LLM-модуля (ACESTEP_INIT_LLM=true)",
    )
    lm_temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Температура сэмплирования LLM. "
        "Диапазон: 0.0–2.0. По умолчанию: null (1.0). "
        "Выше = более креативные, но менее предсказуемые метаданные",
    )
    lm_top_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling для LLM. "
        "Диапазон: 0.0–1.0. По умолчанию: null (0.95). "
        "Ограничивает набор токенов по суммарной вероятности",
    )
    lm_top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=500,
        description="Top-K сэмплирование для LLM. "
        "Диапазон: 1–500. По умолчанию: null (50). "
        "Ограничивает выбор K самыми вероятными токенами",
    )
    lm_max_tokens: Optional[int] = Field(
        default=None,
        ge=64,
        le=4096,
        description="Максимальное количество токенов LLM. "
        "Диапазон: 64–4096. По умолчанию: null (2048)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "chill lofi hip-hop beat with vinyl crackle and jazzy piano",
                    "task_type": "text2music",
                    "duration": 60,
                    "lyrics": "[verse]\nLate night vibes\nCity lights outside\n[chorus]\nJust relax and unwind",
                    "style": "lofi, hip-hop, chill, jazzy",
                    "bpm": 85,
                    "keyscale": "F minor",
                    "seed": 42,
                    "num_steps": 8,
                    "cfg_scale": 3.5,
                    "batch_size": 1,
                    "audio_format": "wav",
                },
                {
                    "prompt": "epic cinematic orchestral soundtrack",
                    "task_type": "text2music",
                    "duration": 180,
                    "instrumental": True,
                    "style": "orchestral, epic, cinematic",
                    "bpm": 110,
                    "keyscale": "D minor",
                    "timesignature": "4/4",
                    "num_steps": 16,
                    "cfg_scale": 5.0,
                    "audio_format": "flac",
                },
                {
                    "prompt": "acoustic guitar cover in bossa nova style",
                    "task_type": "cover",
                    "src_audio": "<base64-encoded audio>",
                    "audio_cover_strength": 0.7,
                    "style": "bossa nova, acoustic",
                },
            ]
        }
    }


class GenerationResponse(BaseModel):
    task_id: str = Field(description="Уникальный ID задачи. Используйте для проверки статуса через GET /status/{task_id}")
    status: str = Field(description="Текущий статус: pending (в очереди)")


class StatusResponse(BaseModel):
    task_id: str = Field(description="ID задачи")
    status: str = Field(description="Статус: pending | processing | success | failed")
    file_url: Optional[str] = Field(default=None, description="Ссылка для скачивания (только при status=success). Пример: /files/abc123.wav")
    error: Optional[str] = Field(default=None, description="Описание ошибки (только при status=failed)")


class LoraTrainResponse(BaseModel):
    task_id: str = Field(description="ID задачи обучения. Проверяйте статус через GET /status/{task_id}")
    status: str = Field(description="Текущий статус: pending (в очереди)")
    style_name: str = Field(description="Имя стиля. После обучения используйте как lora_id в /generate")


class HealthResponse(BaseModel):
    status: str = Field(description="Статус API: healthy")
    redis_connected: bool = Field(description="Подключение к Redis (брокер задач)")
    output_dir: str = Field(description="Директория для сгенерированных файлов")
    model_path: str = Field(description="Путь к модели ACE-Step")


def _extract_audio_from_zip(archive_content: bytes, tmp_dir: str) -> List[str]:
    zip_path = os.path.join(tmp_dir, "audio.zip")
    with open(zip_path, "wb") as f:
        f.write(archive_content)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.namelist():
            member_path = os.path.realpath(os.path.join(tmp_dir, member))
            if not member_path.startswith(os.path.realpath(tmp_dir) + os.sep):
                raise HTTPException(status_code=400, detail="Invalid archive: path traversal detected")
        zip_ref.extractall(tmp_dir)

    os.remove(zip_path)

    audio_files = []
    for root_dir, _, files in os.walk(tmp_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in AUDIO_EXTENSIONS:
                audio_files.append(os.path.join(root_dir, file))

    return audio_files


@app.get("/", tags=["Состояние"], summary="Информация об API",
         description="Возвращает общую информацию о сервисе и список доступных эндпоинтов.")
async def root():
    return {
        "service": "ACE-Step Music Generation API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "generate": "POST /generate — создать трек",
            "status": "GET /status/{task_id} — проверить статус",
            "files": "GET /files/{filename} — скачать файл",
            "train_lora": "POST /train/lora — обучить стиль",
            "health": "GET /health — состояние сервиса",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Состояние"], summary="Проверка здоровья",
         description="Проверяет работоспособность API, подключение к Redis и доступность модели. "
         "Используйте для мониторинга и health-check в Docker/Kubernetes.")
async def health_check():
    redis_ok = False
    try:
        r = redis_lib.Redis(connection_pool=_redis_pool)
        r.ping()
        redis_ok = True
    except Exception:
        pass

    return {
        "status": "healthy",
        "redis_connected": redis_ok,
        "output_dir": settings.OUTPUT_DIR,
        "model_path": settings.MODEL_PATH,
    }


@app.post("/generate", response_model=GenerationResponse, tags=["Генерация"],
          summary="Создать музыкальный трек",
          description="""Отправляет задачу на генерацию музыки. Возвращает `task_id` для отслеживания.

**Минимальный запрос** — только `prompt`:
```json
{"prompt": "jazz piano solo"}
```

**Генерация с метаданными** (BPM, тональность, размер):
```json
{
  "prompt": "energetic rock with electric guitar",
  "duration": 120,
  "bpm": 140,
  "keyscale": "E minor",
  "timesignature": "4/4",
  "style": "rock, energetic, guitar",
  "num_steps": 16,
  "cfg_scale": 5.0
}
```

**Генерация с вокалом** (lyrics + язык):
```json
{
  "prompt": "romantic pop ballad",
  "lyrics": "[verse]\\nПод звёздным небом тишина\\n[chorus]\\nТы и я — одна мечта",
  "vocal_language": "ru",
  "duration": 180
}
```

**Cover** — перенос стиля на аудио:
```json
{
  "task_type": "cover",
  "prompt": "jazz piano version",
  "src_audio": "<base64>",
  "audio_cover_strength": 0.7
}
```

**Repaint** — перегенерация фрагмента:
```json
{
  "task_type": "repaint",
  "prompt": "smooth piano solo transition",
  "src_audio": "<base64>",
  "repainting_start": 10.0,
  "repainting_end": 20.0
}
```

**С LLM-планированием** (автоопределение BPM, структуры):
```json
{
  "prompt": "ambient meditation music",
  "thinking": true,
  "lm_temperature": 0.8
}
```

**Время генерации:** ~2 сек на A100, ~10 сек на RTX 3090, 5–15 мин на CPU (turbo-режим).""")
async def generate(request: GenerationRequest):
    task_id = str(uuid.uuid4())

    from tasks.generation_tasks import generate_track

    generation_params = request.model_dump(exclude_none=True)
    generate_track.apply_async(args=[task_id, generation_params], task_id=task_id)

    return GenerationResponse(task_id=task_id, status="pending")


@app.get("/status/{task_id}", response_model=StatusResponse, tags=["Генерация"],
         summary="Проверить статус задачи",
         description="""Возвращает текущий статус задачи генерации или обучения LoRA.

**Возможные статусы:**
- `pending` — задача в очереди, ожидает выполнения
- `processing` — идёт генерация / обучение
- `success` — готово, поле `file_url` содержит ссылку на файл
- `failed` — ошибка, поле `error` содержит описание

**Рекомендация:** опрашивайте статус каждые 2–5 секунд до получения `success` или `failed`.""")
async def get_status(task_id: str):
    from celery.result import AsyncResult

    result = AsyncResult(task_id, app=celery_app)

    if result.state == "PENDING":
        return StatusResponse(task_id=task_id, status="pending")
    elif result.state == "PROGRESS":
        return StatusResponse(task_id=task_id, status="processing")
    elif result.state == "SUCCESS":
        task_result = result.result
        if isinstance(task_result, dict):
            if task_result.get("status") == "success":
                file_path = task_result.get("file_path", "")
                safe_name = os.path.basename(file_path)
                return StatusResponse(
                    task_id=task_id,
                    status="success",
                    file_url=f"/files/{safe_name}",
                )
            else:
                return StatusResponse(
                    task_id=task_id,
                    status="failed",
                    error=task_result.get("error", "Unknown error"),
                )
        return StatusResponse(task_id=task_id, status="success")
    elif result.state == "FAILURE":
        return StatusResponse(
            task_id=task_id,
            status="failed",
            error=str(result.result),
        )
    else:
        return StatusResponse(task_id=task_id, status=result.state.lower())


@app.get("/files/{filename}", tags=["Файлы"],
         summary="Скачать аудиофайл",
         description="""Скачивает сгенерированный аудиофайл по имени.

Имя файла берётся из поля `file_url` в ответе `/status/{task_id}`.

**Пример:**
```bash
curl -O http://localhost:5000/files/abc123.wav
```

Поддерживаемые форматы: `.wav`, `.mp3`, `.flac`, `.ogg`, `.opus`""")
async def get_file(filename: str):
    safe_filename = os.path.basename(filename)
    if safe_filename != filename or ".." in filename:
        raise HTTPException(status_code=403, detail="Access denied")

    output_real = os.path.realpath(settings.OUTPUT_DIR)
    file_path = os.path.realpath(os.path.join(output_real, safe_filename))

    if not file_path.startswith(output_real + os.sep):
        raise HTTPException(status_code=403, detail="Access denied")

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    content_type, _ = mimetypes.guess_type(safe_filename)
    if not content_type:
        content_type = "audio/wav"

    return FileResponse(
        path=file_path,
        media_type=content_type,
        filename=safe_filename,
        headers={"Cache-Control": "no-cache"},
    )


@app.post("/train/lora", response_model=LoraTrainResponse, tags=["Обучение LoRA"],
          summary="Обучить собственный стиль (LoRA)",
          description="""Загрузите ZIP-архив с аудиозаписями вашего стиля для обучения LoRA-адаптера.

**Требования к архиву:**
- Формат: `.zip`
- Количество аудиофайлов: **от 5 до 10**
- Допустимые форматы аудио: `.wav`, `.mp3`, `.flac`, `.ogg`, `.opus`
- Рекомендуемая длительность каждого файла: 30–180 сек
- Все файлы должны быть в одном стиле

**После обучения:**
Используйте имя стиля (`style_name`) как `lora_id` в запросе `/generate`:
```json
{"prompt": "upbeat track", "lora_id": "my_style_name"}
```

**Пример (curl):**
```bash
curl -X POST http://localhost:5000/train/lora \\
  -F "style_name=my_lofi" \\
  -F "audio_archive=@samples.zip"
```

**Время обучения:** 10–30 мин на GPU.""")
async def train_lora(
    style_name: str = Form(..., description="Имя стиля. Латиницей, без пробелов. Пример: my_lofi_style"),
    audio_archive: UploadFile = File(..., description="ZIP-архив с 5–10 аудиофайлами (.wav, .mp3, .flac, .ogg, .opus)"),
):
    filename = audio_archive.filename or ""
    if not filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Please upload a ZIP archive")

    tmp_dir = tempfile.mkdtemp(prefix="lora_train_")

    try:
        content = await audio_archive.read()
        audio_files = _extract_audio_from_zip(content, tmp_dir)

        if len(audio_files) < MIN_LORA_FILES:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least {MIN_LORA_FILES} audio files for LoRA training, got {len(audio_files)}",
            )
        if len(audio_files) > MAX_LORA_FILES:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum {MAX_LORA_FILES} audio files for LoRA training, got {len(audio_files)}",
            )

        task_id = str(uuid.uuid4())

        from tasks.generation_tasks import train_lora_task

        train_lora_task.apply_async(args=[style_name, audio_files], task_id=task_id)

        return LoraTrainResponse(
            task_id=task_id,
            status="pending",
            style_name=style_name,
        )
    except zipfile.BadZipFile:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Invalid ZIP archive")
    except HTTPException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.error(f"Failed to process training request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
