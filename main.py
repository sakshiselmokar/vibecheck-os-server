"""
VibeCheck OS — Middleware Server
================================
YOU host this once (free tier on Render or Railway).
YOUR HF token lives in an environment variable here.
Users install the `vibecheck-os` pip package and point it at your URL.
They never see your token.

Deploy steps (Render free tier):
  1. Push this folder to a GitHub repo
  2. New Web Service → connect repo → set env var HF_TOKEN=hf_xxx
  3. Start command: uvicorn main:app --host 0.0.0.0 --port $PORT
  4. Share the public URL (e.g. https://vibecheck.onrender.com)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os

app = FastAPI(title="VibeCheck OS API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_URL   = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
LABELS   = [
    "Deep Work",
    "Procrastination",
    "Debugging Panic",
    "Meeting Zombie",
    "Fake Productivity",
]


class TextInput(BaseModel):
    text: str


@app.get("/")
def root():
    return {"status": "ok", "service": "VibeCheck OS API", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"healthy": True, "token_set": bool(HF_TOKEN)}


@app.post("/classify")
async def classify(body: TextInput):
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HF_TOKEN not set on server")

    async with httpx.AsyncClient(timeout=25) as client:
        resp = await client.post(
            HF_URL,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={
                "inputs": body.text.strip()[:500],
                "parameters": {
                    "candidate_labels": LABELS,
                    "multi_label": False,
                },
            },
        )

    if resp.status_code == 503:
        raise HTTPException(status_code=503, detail="Model loading on HF, retry in ~20s")
    if resp.status_code == 429:
        raise HTTPException(status_code=429, detail="HF rate limit hit")
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"HF returned {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    result = data[0] if isinstance(data, list) else data

    return {
        "label":      result["labels"][0],
        "confidence": round(result["scores"][0] * 100),
    }