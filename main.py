# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🎵 Thaipodee Music API
# Copyright (c) 2026 Thaipodee Global Market
# เว็บไซต์: https://thaipodee.com
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torchaudio
from audiocraft.models import MusicGen
import uuid
import os

app = FastAPI(title="Thaipodee Music API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "generated"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class MusicRequest(BaseModel):
    prompt: str
    duration: int = 10

model = None

@app.on_event("startup")
async def load_model():
    global model
    model = MusicGen.get_pretrained("small")
    model.set_generation_params(duration=10)

@app.get("/")
def root():
    return {
        "name": "🎵 Thaipodee Music API",
        "version": "1.0.0",
        "copyright": "© 2026 Thaipodee Global Market"
    }

@app.post("/generate")
def generate_music(request: MusicRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    wav = model.generate([request.prompt])
    wav = wav[0].cpu()
    
    filename = f"{uuid.uuid4().hex}.wav"
    filepath = os.path.join(UPLOAD_DIR, filename)
    
    torchaudio.save(filepath, wav, 32000)
    
    return {
        "status": "success",
        "prompt": request.prompt,
        "audio_url": f"/download/{filename}"
    }

@app.get("/download/{filename}")
def download_music(filename: str):
    filepath = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(filepath, media_type="audio/wav", filename=filename)
