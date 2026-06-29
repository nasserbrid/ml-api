import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

from app.logger import logger
from app.routers import stt
from config.settings import settings

BASE_MODEL_ID = "openai/whisper-large-v3-turbo"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.hf_token:
        os.environ["HF_TOKEN"] = settings.hf_token

    logger.info("Chargement du modèle Whisper large-v3-turbo (base)...")
    app.state.pipeline_stt = pipeline(
        "automatic-speech-recognition",
        model=BASE_MODEL_ID,
        dtype=torch.float32,
        device="cpu",
    )
    logger.info("Modèle base chargé.")

    logger.info("Chargement de l'adapter LoRA %s...", settings.hf_adapter_id)
    base_model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float32
    )
    peft_model = PeftModel.from_pretrained(base_model, settings.hf_adapter_id)
    merged_model = peft_model.merge_and_unload()
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)
    app.state.pipeline_stt_pro = pipeline(
        "automatic-speech-recognition",
        model=merged_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=torch.float32,
        device="cpu",
    )
    logger.info("Modèle pro chargé — API prête.")
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
