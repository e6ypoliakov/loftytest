import os
import uuid
import zipfile
import shutil
import tempfile
import logging
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
  -d '{"prompt": "energetic electronic dance music"}'
```

**2. Проверьте статус по task_id:**
```bash
curl http://localhost:5000/status/{task_id}
```

**3. Скачайте готовый файл:**
```bash
curl -O http://localhost:5000/files/{filename}
```

### Поддерживаемые аудиоформаты
`.wav` `.mp3` `.flac` `.ogg` `.opus`

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


class GenerationRequest(BaseModel):
    prompt: str = Field(
        ...,
        description="Описание стиля музыки. Чем подробнее — тем точнее результат",
        json_schema_extra={"examples": ["energetic electronic dance music with heavy bass and synth leads"]},
    )
    duration: int = Field(
        default=120,
        ge=10,
        le=600,
        description="Длительность трека в секундах (от 10 до 600). По умолчанию 120 сек (2 минуты)",
    )
    lyrics: Optional[str] = Field(
        default=None,
        description="Текст песни с маркерами структуры: [verse], [chorus], [bridge], [intro], [outro]. "
        "Если не указан — генерируется инструментал",
        json_schema_extra={"examples": ["[verse]\nFeel the rhythm in your soul\n[chorus]\nDance all night long"]},
    )
    style: Optional[str] = Field(
        default=None,
        description="Дополнительные теги стиля через запятую. Уточняют жанр и настроение",
        json_schema_extra={"examples": ["electronic, dance, upbeat, 128bpm"]},
    )
    reference_audio: Optional[str] = Field(
        default=None,
        description="Референсное аудио в формате Base64. Модель постарается создать трек похожего звучания",
    )
    lora_id: Optional[str] = Field(
        default=None,
        description="ID обученного LoRA-адаптера (имя стиля из /train/lora). "
        "Применяет ваш собственный стиль к генерации",
        json_schema_extra={"examples": ["my_lofi_style"]},
    )
    seed: Optional[int] = Field(
        default=-1,
        description="Сид генерации. -1 = случайный. Одинаковый сид + одинаковые параметры = одинаковый результат",
    )
    num_steps: Optional[int] = Field(
        default=8,
        description="Количество шагов диффузии. Больше шагов = выше качество, но дольше. Рекомендуется 8–16",
    )
    cfg_scale: Optional[float] = Field(
        default=3.5,
        description="Сила следования промпту (classifier-free guidance). "
        "Выше = точнее следует описанию, но может звучать менее естественно. Рекомендуется 2.0–7.0",
    )
    batch_size: Optional[int] = Field(
        default=1,
        ge=1,
        le=8,
        description="Количество вариаций за один запрос (1–8). Каждая вариация — уникальный трек",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "chill lofi hip-hop beat with vinyl crackle and jazzy piano",
                    "duration": 60,
                    "lyrics": "[verse]\nLate night vibes\nCity lights outside\n[chorus]\nJust relax and unwind",
                    "style": "lofi, hip-hop, chill, jazzy",
                    "seed": 42,
                    "num_steps": 8,
                    "cfg_scale": 3.5,
                    "batch_size": 1,
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

**Полный запрос** со всеми параметрами:
```json
{
  "prompt": "epic cinematic orchestral music",
  "duration": 180,
  "lyrics": "[intro]\\n(instrumental)\\n[verse]\\nRise above the clouds",
  "style": "orchestral, epic, cinematic, dramatic",
  "seed": 42,
  "num_steps": 16,
  "cfg_scale": 5.0,
  "batch_size": 2
}
```

**Время генерации:** ~30–60 сек на GPU, 5–15 мин на CPU.""")
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
