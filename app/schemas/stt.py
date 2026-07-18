from pydantic import BaseModel


class DiarizedSegment(BaseModel):
    speaker: str
    text: str
    start: float
    end: float


class STTResponse(BaseModel):
    text: str
    segments: list[DiarizedSegment]
