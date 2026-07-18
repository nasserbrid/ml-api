import os
import subprocess
import tempfile

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.logger import logger
from app.schemas.stt import DiarizedSegment, STTResponse

router = APIRouter(prefix="/stt", tags=["stt"])

WHISPER_SAMPLE_RATE = 16000


def _normalize_to_wav(audio_bytes: bytes) -> str:
    # Écriture sur disque obligatoire : les conteneurs box-based (MP4/M4A) placent parfois
    # l'atome moov après les données audio (pas de "faststart") — un pipe:0 stdin est
    # séquentiel et non-seekable, ffmpeg ne peut alors pas relire la table des échantillons
    # et produit un WAV vide (header seul) sans lever d'erreur explicite.
    with tempfile.NamedTemporaryFile(delete=False) as tmp_input:
        tmp_input.write(audio_bytes)
        tmp_input_path = tmp_input.name

    tmp_output_path = f"{tmp_input_path}.out.wav"

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
                "-y",
                tmp_output_path,
            ],
            capture_output=True,
        )
    finally:
        os.unlink(tmp_input_path)

    if result.returncode != 0 or not os.path.exists(tmp_output_path) or os.path.getsize(tmp_output_path) == 0:
        detail = result.stderr.decode(errors="replace")
        if os.path.exists(tmp_output_path):
            os.unlink(tmp_output_path)
        raise HTTPException(status_code=400, detail=f"Format audio invalide : {detail}")

    return tmp_output_path


def _merge(diarization, fw_segments) -> list[dict]:
    result = []
    for seg in fw_segments:
        mid = (seg.start + seg.end) / 2
        speaker = "SPEAKER_00"
        for turn, _, label in diarization.itertracks(yield_label=True):
            if turn.start <= mid <= turn.end:
                speaker = label
                break
        result.append(
            {
                "speaker": speaker,
                "text": seg.text.strip(),
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
            }
        )

    merged = []
    for item in result:
        if merged and merged[-1]["speaker"] == item["speaker"]:
            merged[-1]["text"] += " " + item["text"]
            merged[-1]["end"] = item["end"]
        else:
            merged.append(dict(item))
    return merged


def _transcribe_and_diarize(request: Request, wav_path: str) -> STTResponse:
    segments, info = request.app.state.whisper_model.transcribe(
        wav_path, language="fr", beam_size=5, word_timestamps=True
    )
    fw_segments = list(segments)

    diarization = request.app.state.diarization_pipeline(wav_path)

    merged = _merge(diarization, fw_segments)

    text = " ".join(f"[{item['speaker']}] {item['text']}" for item in merged)

    logger.info("Transcription + diarisation terminées — durée audio : %.1fs", info.duration)
    logger.debug("Transcription terminée : %d caractères, %d segments", len(text), len(merged))

    return STTResponse(
        text=text,
        segments=[DiarizedSegment(**item) for item in merged],
    )


@router.post("", response_model=STTResponse)
async def speech_to_text(file: UploadFile = File(...), request: Request = None) -> STTResponse:
    audio_bytes = await file.read()
    wav_path = _normalize_to_wav(audio_bytes)

    try:
        return _transcribe_and_diarize(request, wav_path)
    finally:
        os.unlink(wav_path)


@router.post("/pro", response_model=STTResponse)
async def speech_to_text_pro(file: UploadFile = File(...), request: Request = None) -> STTResponse:
    audio_bytes = await file.read()
    wav_path = _normalize_to_wav(audio_bytes)

    try:
        return _transcribe_and_diarize(request, wav_path)
    finally:
        os.unlink(wav_path)
