import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    REDIS_URL: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    MODEL_PATH: Optional[str] = os.environ.get("MODEL_PATH", "acestep-v15-turbo")
    OUTPUT_DIR: str = os.environ.get("OUTPUT_DIR", "generated_audio")
    HF_TOKEN: Optional[str] = os.environ.get("HF_TOKEN", None)
    LORA_DIR: str = os.environ.get("LORA_DIR", "lora_models")

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()

os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.LORA_DIR, exist_ok=True)
