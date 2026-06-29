import json

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    hf_token: str | None = None
    cors_origins: str = "http://localhost:5173"
    hf_adapter_id: str = "nasserbrid/whisper-large-v3-turbo-fr"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    def get_cors_list(self) -> list[str]:
        value = self.cors_origins.strip()
        if value.startswith("["):
            return json.loads(value)
        return [origin.strip() for origin in value.split(",")]


settings = Settings()
