import io
import os
import subprocess
import tempfile
import wave

import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.logger import logger
from app.schemas.stt import STTResponse

router = APIRouter(prefix="/stt", tags=["stt"])

WHISPER_SAMPLE_RATE = 16000


def _normalize_to_wav(audio_bytes: bytes) -> bytes:
    # Écriture sur disque obligatoire : les conteneurs box-based (MP4/M4A) placent parfois
    # l'atome moov après les données audio (pas de "faststart") — un pipe:0 stdin est
    # séquentiel et non-seekable, ffmpeg ne peut alors pas relire la table des échantillons
    # et produit un WAV vide (header seul) sans lever d'erreur explicite.
    with tempfile.NamedTemporaryFile(delete=False) as tmp_input:
        tmp_input.write(audio_bytes)
        tmp_input_path = tmp_input.name

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-err_detect", "ignore_err",
                "-i", tmp_input_path,
                "-f", "wav",
                "-ac", "1",
                "-ar", str(WHISPER_SAMPLE_RATE),
                "-acodec", "pcm_s16le",
                "-loglevel", "error",
                "pipe:1",
            ],
            capture_output=True,
        )
    finally:
        os.unlink(tmp_input_path)

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
