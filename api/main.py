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

app = FastAPI(
    title="ACE-Step Music Generation API",
    description="REST API for music generation using ACE-Step 1.5 model",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Music style/description prompt")
    duration: int = Field(default=120, ge=10, le=600, description="Duration in seconds")
    lyrics: Optional[str] = Field(default=None, description="Song lyrics with structure markers")
    style: Optional[str] = Field(default=None, description="Music style tags")
    reference_audio: Optional[str] = Field(default=None, description="Base64-encoded reference audio")
    lora_id: Optional[str] = Field(default=None, description="LoRA adapter ID for style customization")
    seed: Optional[int] = Field(default=-1, description="Random seed (-1 for random)")
    num_steps: Optional[int] = Field(default=8, description="Diffusion steps")
    cfg_scale: Optional[float] = Field(default=3.5, description="Classifier-free guidance scale")
    batch_size: Optional[int] = Field(default=1, ge=1, le=8, description="Number of variations")


class GenerationResponse(BaseModel):
    task_id: str
    status: str


class StatusResponse(BaseModel):
    task_id: str
    status: str
    file_url: Optional[str] = None
    error: Optional[str] = None


class LoraTrainResponse(BaseModel):
    task_id: str
    status: str
    style_name: str


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


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "ACE-Step Music Generation API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "generate": "POST /generate",
            "status": "GET /status/{task_id}",
            "files": "GET /files/{filename}",
            "train_lora": "POST /train/lora",
        },
    }


@app.get("/health", tags=["Health"])
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


@app.post("/generate", response_model=GenerationResponse, tags=["Generation"])
async def generate(request: GenerationRequest):
    task_id = str(uuid.uuid4())

    from tasks.generation_tasks import generate_track

    generation_params = request.model_dump(exclude_none=True)
    generate_track.apply_async(args=[task_id, generation_params], task_id=task_id)

    return GenerationResponse(task_id=task_id, status="pending")


@app.get("/status/{task_id}", response_model=StatusResponse, tags=["Generation"])
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


@app.get("/files/{filename}", tags=["Files"])
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


@app.post("/train/lora", response_model=LoraTrainResponse, tags=["Training"])
async def train_lora(
    style_name: str = Form(..., description="Name for the style/LoRA adapter"),
    audio_archive: UploadFile = File(..., description="ZIP archive with 5-10 audio files"),
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
