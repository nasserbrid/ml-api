import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.logger import logger
from app.rate_limit import limiter
from app.routers import stt
from config.settings import settings


class ContentLengthLimitMiddleware(BaseHTTPMiddleware):
    # Rejette une requête surdimensionnée via l'en-tête Content-Length AVANT que
    # FastAPI ne parse le corps multipart — _read_with_limit() (stt.py) reste la
    # vraie limite mais n'agit qu'après que Starlette a déjà bufferisé le corps.
    # Ne couvre pas Transfer-Encoding: chunked (sans Content-Length) : aucun client
    # de ce projet (backend_voclaire, démo publique) n'envoie de requête chunked.
    async def dispatch(self, request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > settings.max_upload_bytes:
            return JSONResponse(status_code=413, content={"detail": "Fichier audio trop volumineux."})
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    # X-Frame-Options/CSP omis volontairement : cette API ne renvoie que du JSON,
    # jamais rendue dans un navigateur comme une page — aucun effet protecteur réel
    # ici (contrairement à frontend_voclaire, le vrai correctif à valeur pour ce point).
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response


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
    # pyannote/speaker-diarization-3.1 est un checkpoint officiel HuggingFace, gated,
    # déjà authentifié via HF_TOKEN — même niveau de confiance qu'avant torch 2.6
    # (où weights_only valait False par défaut). On restaure ce comportement
    # UNIQUEMENT pour ce chargement précis, pas pour le reste de l'app — la
    # protection CVE-2025-32434 reste active pour tout autre usage futur de torch.load.
    _original_torch_load = torch.load

    def _load_trusted_pyannote_checkpoint(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _original_torch_load(*args, **kwargs)

    torch.load = _load_trusted_pyannote_checkpoint
    try:
        app.state.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1"
        )
    finally:
        torch.load = _original_torch_load

    if app.state.diarization_pipeline is None:
        raise RuntimeError(
            "Échec du chargement du pipeline de diarisation pyannote/speaker-diarization-3.1 : "
            "vérifier HF_TOKEN et l'acceptation des conditions d'utilisation sur "
            "pyannote/speaker-diarization-3.1 ET pyannote/segmentation-3.0 (huggingface.co)."
        )
    logger.info("Pipeline de diarisation chargé — API prête.")
    yield


app = FastAPI(title="voclaire ML API", lifespan=lifespan)

app.add_middleware(ContentLengthLimitMiddleware)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_list(),
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

app.add_middleware(SecurityHeadersMiddleware)

app.include_router(stt.router)


@app.get("/health", tags=["infra"])
def health():
    return {"status": "ok"}
