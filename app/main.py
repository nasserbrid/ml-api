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

    logger.info("Chargement de l'adapter LoRA %s...", settings.hf_adapter_id)
    base_model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float32
    )
    peft_model = PeftModel.from_pretrained(base_model, settings.hf_adapter_id)
    merged_model = peft_model.merge_and_unload()
    merged_model.eval()

    logger.info("Quantisation dynamique int8 du modèle mergé...")
    quantized_model = torch.quantization.quantize_dynamic(
        merged_model, {torch.nn.Linear}, dtype=torch.qint8
    )

    processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)
    shared_pipeline = pipeline(
        "automatic-speech-recognition",
        model=quantized_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        device="cpu",
    )
    app.state.pipeline_stt = shared_pipeline
    app.state.pipeline_stt_pro = shared_pipeline
    app.state.pro_prompt_ids = processor.tokenizer.get_prompt_ids(
        settings.pro_initial_prompt, return_tensors="pt"
    )
    logger.info("Modèle unique (LoRA mergé + quantisé int8) chargé — API prête.")
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
