from typing import Literal

from pydantic import BaseModel

from config.settings import settings


class TTSRequest(BaseModel):
    text: str
    voice: str = settings.default_voice
    speed: float = settings.default_speed
    lang: Literal["a", "f"] = settings.lang_code
