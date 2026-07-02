import io
import subprocess
import wave

import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.logger import logger
from app.schemas.stt import STTResponse

router = APIRouter(prefix="/stt", tags=["stt"])

WHISPER_SAMPLE_RATE = 16000


def _normalize_to_wav(audio_bytes: bytes) -> bytes:
    # -err_detect ignore_err tolère les paquets Opus corrompus produits par MediaRecorder (WebM sans durée)
    result = subprocess.run(
        [
            "ffmpeg",
            "-err_detect", "ignore_err",
            "-i", "pipe:0",
            "-f", "wav",
            "-ac", "1",
            "-ar", str(WHISPER_SAMPLE_RATE),
            "-acodec", "pcm_s16le",
            "-loglevel", "error",
            "pipe:1",
        ],
        input=audio_bytes,
        capture_output=True,
    )
    if not result.stdout:
        detail = result.stderr.decode(errors="replace")
        raise HTTPException(status_code=400, detail=f"Format audio invalide : {detail}")
    return result.stdout


def _decode_audio(audio_bytes: bytes) -> np.ndarray:
    wav_bytes = _normalize_to_wav(audio_bytes)
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        frames = wf.readframes(wf.getnframes())
    raw_samples = np.frombuffer(frames, dtype=np.int16)
    return raw_samples.astype(np.float32) / (2 ** 15)


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
        generate_kwargs={
            "language": "fr",
            "prompt_ids": request.app.state.pro_prompt_ids,
        },
    )

    transcription = result["text"].strip()
    logger.debug("Transcription pro terminée : %d caractères", len(transcription))

    return STTResponse(text=transcription)
