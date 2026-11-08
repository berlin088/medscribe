from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

load_dotenv()

from . import asr, llm  # noqa: E402  (after load_dotenv so env vars are set)

ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(title="MedScribe")


@app.on_event("startup")
def warm_up():
    # Load Parakeet at startup so the first utterance isn't slow.
    asr.load_model()
    print("Parakeet model ready.")


@app.post("/api/transcribe")
async def transcribe(request: Request):
    """Body: raw float32 little-endian mono PCM, 16 kHz (one VAD speech segment)."""
    raw = await request.body()
    if len(raw) < 4 or len(raw) % 4:
        raise HTTPException(400, "Expected raw float32 PCM")
    audio = np.frombuffer(raw, dtype=np.float32)
    if len(audio) < 16000 * 0.3:  # under 0.3 s: nothing worth recognising
        return {"text": ""}
    text = await run_in_threadpool(asr.transcribe, audio)
    return {"text": text}


class AnalyzeIn(BaseModel):
    transcript: str


@app.post("/api/analyze")
async def analyze(body: AnalyzeIn):
    if not body.transcript.strip():
        raise HTTPException(400, "Transcript is empty")
    try:
        return await run_in_threadpool(llm.analyze, body.transcript)
    except llm.LLMError as e:
        raise HTTPException(502, str(e))


# Must come last so it doesn't shadow /api routes
app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")