import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.logger import logger
from app.rate_limit import limiter
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
        "pyannote/speaker-diarization-3.1"
    )
    if app.state.diarization_pipeline is None:
        raise RuntimeError(
            "Échec du chargement du pipeline de diarisation pyannote/speaker-diarization-3.1 : "
            "vérifier HF_TOKEN et l'acceptation des conditions d'utilisation sur "
            "pyannote/speaker-diarization-3.1 ET pyannote/segmentation-3.0 (huggingface.co)."
        )
    logger.info("Pipeline de diarisation chargé — API prête.")
    yield


app = FastAPI(title="voclaire ML API", lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

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
