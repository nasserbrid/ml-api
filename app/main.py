import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from kokoro import KPipeline

from app.logger import logger
from app.routers import tts
from config.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.hf_token:
        os.environ["HF_TOKEN"] = settings.hf_token

    logger.info("Chargement des pipelines Kokoro...")
    app.state.pipeline_en = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    app.state.pipeline_fr = KPipeline(lang_code="f", repo_id="hexgrad/Kokoro-82M")
    logger.info("Pipelines chargés — API prête.")
    yield


app = FastAPI(title="voclaire ML API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

app.include_router(tts.router)


@app.get("/health", tags=["infra"])
def health():
    return {"status": "ok"}
