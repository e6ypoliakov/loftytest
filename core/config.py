from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    REDIS_URL: str = "redis://localhost:6379/0"
    MODEL_PATH: str = "acestep-v15-turbo"
    OUTPUT_DIR: str = "generated_audio"
    HF_TOKEN: Optional[str] = None
    LORA_DIR: str = "lora_models"
    ACESTEP_INIT_LLM: bool = False

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()

import os
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.LORA_DIR, exist_ok=True)
