import io

import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydub import AudioSegment

from app.logger import logger
from app.schemas.stt import STTResponse

router = APIRouter(prefix="/stt", tags=["stt"])

WHISPER_SAMPLE_RATE = 16000
PRO_INITIAL_PROMPT = (
    "compte rendu réunion procès-verbal contrat devis facture rapport"
)


def _decode_audio(audio_bytes: bytes) -> np.ndarray:
    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Format audio invalide : {error}")

    audio_segment = audio_segment.set_channels(1).set_frame_rate(WHISPER_SAMPLE_RATE)
    raw_samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
    audio_float32 = raw_samples / (2 ** 15)
    return audio_float32


@router.post("", response_model=STTResponse)
async def speech_to_text(file: UploadFile = File(...), request: Request = None) -> STTResponse:
    audio_bytes = await file.read()
    audio_float32 = _decode_audio(audio_bytes)

    duration = len(audio_float32) / WHISPER_SAMPLE_RATE
    logger.info("Transcription (base) en cours — durée audio : %.1fs", duration)

    result = request.app.state.pipeline_stt(
        {"array": audio_float32, "sampling_rate": WHISPER_SAMPLE_RATE}
    )

    transcription = result["text"].strip()
    logger.debug("Transcription terminée : %d caractères", len(transcription))

    return STTResponse(text=transcription)


@router.post("/pro", response_model=STTResponse)
async def speech_to_text_pro(file: UploadFile = File(...), request: Request = None) -> STTResponse:
    audio_bytes = await file.read()
    audio_float32 = _decode_audio(audio_bytes)

    duration = len(audio_float32) / WHISPER_SAMPLE_RATE
    logger.info("Transcription (pro) en cours — durée audio : %.1fs", duration)

    result = request.app.state.pipeline_stt_pro(
        {"array": audio_float32, "sampling_rate": WHISPER_SAMPLE_RATE},
        generate_kwargs={"language": "fr", "initial_prompt": PRO_INITIAL_PROMPT},
    )

    transcription = result["text"].strip()
    logger.debug("Transcription pro terminée : %d caractères", len(transcription))

    return STTResponse(text=transcription)
