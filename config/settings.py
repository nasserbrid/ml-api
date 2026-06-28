from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    default_voice: str = "af_heart"
    default_speed: float = 1.0
    lang_code: str = "a"
    hf_token: str | None = None
    cors_origins: list[str] = ["http://localhost:5173"]

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
