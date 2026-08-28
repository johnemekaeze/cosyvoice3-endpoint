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
    language: str = Field("hausa", description="Language name, e.g. hausa | swahili | zulu. See GET /languages")
    voice: Optional[str] = Field(None, description="'male' or 'female'. Ignored when uploading your own clip.")
    prompt_text: Optional[str] = Field(None, description="Transcript of prompt_audio_base64, required with it")
    prompt_audio_base64: Optional[str] = Field(None, description="Your own reference voice clip (WAV, base64) to clone")


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


# The replica has died mid-session several times (readyReplica drops to 0 while the platform
# still reports "running"), usually after rapid language switching. It recovers on its own,
# so every not-ready response says so explicitly rather than leaving a caller to guess whether
# the service is broken.
RETRY_NOTE = ("The endpoint is starting up or was restarting. This is expected after idle "
              "time or when switching language. Please TRY AGAIN in 1-2 minutes; a cold "
              "start can take up to 8 minutes. Use a client timeout of at least 10 minutes.")


@app.get("/health")
def health():
    """HF probes this. Return 503 until the handler (and warmup model) is ready."""
    if _READY and _HANDLER is not None:
        return {"status": "ok"}
    if _INIT_ERROR:
        # Still 503 so the platform retries; include detail for logs/UI.
        return JSONResponse(status_code=503, content={"status": "error", "detail": _INIT_ERROR,
                                                      "retry": True, "note": RETRY_NOTE})
    try:
        _get_handler()
        return {"status": "ok"}
    except Exception as exc:
        return JSONResponse(status_code=503, content={"status": "loading", "detail": str(exc),
                                                      "retry": True, "note": RETRY_NOTE})


@app.get("/")
def root():
    from handler import MODELS

    return {
        "service": "ALL Lab CosyVoice3 Inference Endpoint",
        "health": "/health",
        "languages": "GET /languages",
        "tts": "POST /",
        "languages_list": sorted(MODELS.keys()),
        "ready": _READY,
    }


@app.get("/languages")
def languages():
    """Everything a client needs to build a language/voice picker: the canonical name,
    a display label, and which preset voices that language actually has."""
    from handler import ALIASES, DEFAULT_LANGUAGE, MODELS, _load_bundled_prompts

    prompts = _load_bundled_prompts()
    out = []
    for code in sorted(MODELS):
        entry = prompts.get(code) or {}
        voices = sorted(k for k in entry if k in ("male", "female")) or ["default"]
        out.append({
            "language": code,
            "display": MODELS[code]["display"],
            "voices": voices,
            "aliases": sorted(a for a, t in ALIASES.items() if t == code),
        })
    return {
        "count": len(out),
        "default_language": DEFAULT_LANGUAGE,
        "languages": out,
    }


@app.post("/")
def tts(req: TTSRequest) -> Dict[str, Any]:
    try:
        handler = _get_handler()
    except Exception as exc:
        raise HTTPException(status_code=503,
                            detail={"error": f"Model not ready: {exc}", "retry": True,
                                    "note": RETRY_NOTE}) from exc

    payload = req.model_dump()
    try:
        return handler(payload)
    except ValueError as exc:
        # a bad request (unknown language/voice, malformed upload) -- retrying will not help
        raise HTTPException(status_code=400,
                            detail={"error": str(exc), "retry": False}) from exc
    except Exception as exc:
        log.exception("generate failed")
        raise HTTPException(status_code=500,
                            detail={"error": f"Generation failed: {exc}", "retry": True,
                                    "note": "Generation is stochastic and occasionally fails or "
                                            "collapses. TRY AGAIN -- the same request usually "
                                            "succeeds on a retry."}) from exc
