import json

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    hf_token: str | None = None
    cors_origins: list[str] = ["http://localhost:5173"]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, list):
            return value
        if value.startswith("["):
            return json.loads(value)
        return [origin.strip() for origin in value.split(",")]

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
