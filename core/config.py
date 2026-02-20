import os
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    REDIS_URL: str = "redis://localhost:6379/0"
    MODEL_PATH: str = "acestep-v15-turbo"
    OUTPUT_DIR: str = "generated_audio"
    HF_TOKEN: Optional[str] = None
    LORA_DIR: str = "lora_models"
    ACESTEP_INIT_LLM: bool = False


settings = Settings()

os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.LORA_DIR, exist_ok=True)
