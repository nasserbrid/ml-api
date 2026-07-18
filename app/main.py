import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

from app.logger import logger
from app.routers import stt
from config.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.hf_token:
        os.environ["HF_TOKEN"] = settings.hf_token

    logger.info("Chargement du modèle faster-whisper large-v3-turbo (int8, CPU)...")
    app.state.whisper_model = WhisperModel(
        "large-v3-turbo", device="cpu", compute_type="int8"
    )
    logger.info("Modèle faster-whisper chargé.")

    logger.info("Chargement du pipeline de diarisation pyannote/speaker-diarization-3.1...")
    app.state.diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=settings.hf_token
    )
    logger.info("Pipeline de diarisation chargé — API prête.")
    yield


app = FastAPI(title="voclaire ML API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_list(),
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

app.include_router(stt.router)


@app.get("/health", tags=["infra"])
def health():
    return {"status": "ok"}
