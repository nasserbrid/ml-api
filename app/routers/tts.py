import io

import numpy as np
import soundfile as sf
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.logger import logger
from app.schemas.tts import TTSRequest

router = APIRouter(prefix="/tts", tags=["tts"])

KOKORO_SAMPLE_RATE = 24000


@router.post("")
def text_to_speech(body: TTSRequest, request: Request) -> StreamingResponse:
    pipelines = {
        "a": request.app.state.pipeline_en,
        "f": request.app.state.pipeline_fr,
    }
    pipeline = pipelines[body.lang]

    audio_chunks = []
    for result in pipeline(body.text, voice=body.voice, speed=body.speed):
        audio = result[2]
        logger.debug("Chunk généré : %s", audio.shape)
        audio_chunks.append(audio)

    if not audio_chunks:
        raise HTTPException(status_code=500, detail="No audio generated")

    full_audio = np.concatenate(audio_chunks)

    buffer = io.BytesIO()
    sf.write(buffer, full_audio, KOKORO_SAMPLE_RATE, format="WAV")
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=speech.wav"},
    )
