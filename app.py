"""FastAPI server for Hugging Face Inference Endpoints (custom container).

Listens on port 8080 by default (HF IE injects PORT=8080 into custom containers).
Uses EndpointHandler for multi-language CosyVoice3 TTS. Not a Gradio/Spaces app.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cosyvoice3-ie")

app = FastAPI(title="ALL Lab CosyVoice3 Inference Endpoint", version="1.0.0")

_HANDLER = None
_READY = False
_INIT_ERROR: Optional[str] = None


class TTSRequest(BaseModel):
    inputs: str = Field(..., min_length=1, description="Text to synthesize")
    language: str = Field("ha-NG", description="ha-NG | tw-GH | ig-NG | ee-GH (or hausa|twi|igbo|ewe)")
    prompt_text: Optional[str] = Field(None, description="Transcript of prompt_audio_base64, if overriding the bundled voice")
    prompt_audio_base64: Optional[str] = Field(None, description="Reference voice clip (WAV, base64) to clone instead of the bundled one")


def _get_handler():
    global _HANDLER, _READY, _INIT_ERROR
    if _HANDLER is not None:
        return _HANDLER
    try:
        from handler import EndpointHandler

        # IE mounts the selected Hub repo at /repository when configured.
        model_dir = os.environ.get("MODEL_DIR", "/repository")
        if not os.path.isdir(model_dir):
            model_dir = "."
        _HANDLER = EndpointHandler(model_dir)
        _READY = True
        _INIT_ERROR = None
        log.info("EndpointHandler ready")
        return _HANDLER
    except Exception as exc:
        _INIT_ERROR = f"{type(exc).__name__}: {exc}"
        log.exception("handler init failed")
        raise


@app.on_event("startup")
def _startup() -> None:
    # Load during startup so /health flips to 200 only when ready.
    try:
        _get_handler()
    except Exception:
        # Keep process alive; /health stays 503 until a later retry succeeds.
        pass


@app.get("/health")
def health():
    """HF probes this. Return 503 until the handler (and warmup model) is ready."""
    if _READY and _HANDLER is not None:
        return {"status": "ok"}
    if _INIT_ERROR:
        # Still 503 so the platform retries; include detail for logs/UI.
        return JSONResponse(status_code=503, content={"status": "error", "detail": _INIT_ERROR})
    try:
        _get_handler()
        return {"status": "ok"}
    except Exception as exc:
        return JSONResponse(status_code=503, content={"status": "loading", "detail": str(exc)})


@app.get("/")
def root():
    from handler import MODELS

    return {
        "service": "ALL Lab CosyVoice3 Inference Endpoint",
        "health": "/health",
        "tts": "POST /",
        "languages": sorted(MODELS.keys()),
        "ready": _READY,
    }


@app.post("/")
def tts(req: TTSRequest) -> Dict[str, Any]:
    try:
        handler = _get_handler()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model not ready: {exc}") from exc

    payload = req.model_dump()
    try:
        return handler(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("generate failed")
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc
