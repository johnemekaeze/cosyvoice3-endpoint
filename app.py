"""FastAPI server for Hugging Face Inference Endpoints (custom container).

Listens on port 8080 by default (HF IE injects PORT=8080 into custom containers).
Uses EndpointHandler for multi-language CosyVoice3 TTS. Not a Gradio/Spaces app.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
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
    voice: Optional[str] = Field(None, description="'male' or 'female' preset. Ignored if you supply your own clip.")
    voice_id: Optional[str] = Field(None, description="Id of a voice previously sent to POST /voice")
    prompt_text: Optional[str] = Field(None, description="Optional transcript of your clip. Omitted is fine and safer.")
    prompt_audio_base64: Optional[str] = Field(None, description="Your own reference voice clip (WAV, base64) to clone")


class VoiceUpload(BaseModel):
    audio_base64: str = Field(..., description="The recorded voice clip, WAV, base64. 5-15s works best.")
    prompt_text: Optional[str] = Field(None, description="Optional transcript. Leave empty unless it matches the audio exactly.")
    name: Optional[str] = Field(None, description="Your own label for this voice")


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


@app.post("/voice")
def upload_voice(req: VoiceUpload) -> Dict[str, Any]:
    """Register a recorded voice for cloning and get back a voice_id.

    Returns 200 with uploaded=true once the clip is decoded and confirmed to be a readable
    WAV of usable length -- that is the "did the upload work" signal. Supplying prompt_text
    is optional and usually better left out: without it the reference text is never fed to
    the model, so it cannot be spoken before the requested text.
    """
    try:
        from handler import register_voice

        return register_voice(req.audio_base64, req.prompt_text, req.name)
    except ValueError as exc:
        raise HTTPException(status_code=400,
                            detail={"error": str(exc), "uploaded": False, "retry": False}) from exc
    except Exception as exc:
        log.exception("voice upload failed")
        raise HTTPException(status_code=500,
                            detail={"error": f"Upload failed: {exc}", "uploaded": False,
                                    "retry": True}) from exc


@app.get("/voice/{voice_id}")
def voice_status(voice_id: str) -> Dict[str, Any]:
    """Whether a previously uploaded voice is still registered on this replica."""
    from handler import VOICE_STORE

    rec = VOICE_STORE.get(voice_id)
    if not rec:
        raise HTTPException(status_code=404, detail={
            "error": f"unknown voice_id '{voice_id}'", "uploaded": False,
            "note": "Voices are held per replica and are lost when it restarts. Upload again."})
    return {"voice_id": voice_id, "uploaded": True, "name": rec.get("name"),
            "duration_sec": rec.get("duration_sec"), "sample_rate": rec.get("sample_rate"),
            "has_transcript": bool(rec.get("prompt_text")),
            "mode": "zero_shot" if rec.get("prompt_text") else "cross_lingual"}


@app.get("/voices")
def list_voices() -> Dict[str, Any]:
    from handler import VOICE_STORE

    return {"count": len(VOICE_STORE),
            "voices": [{"voice_id": k, "name": v.get("name"),
                        "duration_sec": v.get("duration_sec"),
                        "has_transcript": bool(v.get("prompt_text"))}
                       for k, v in VOICE_STORE.items()]}


@app.get("/result/{request_id}")
def stream_result(request_id: str) -> Dict[str, Any]:
    """The verdict on a finished stream: the fields POST / returns in its JSON body.

    Response headers go out before the audio, so they cannot say whether the audio that
    followed was any good. Read the X-Request-Id header off the stream, then call this once
    the stream ends to get ok / audio_generated / duration_sec / peak / attempts.

    Verdicts are held per replica and only the most recent few hundred are kept, so read it
    shortly after the stream finishes rather than hours later.
    """
    from handler import STREAM_RESULTS

    rec = STREAM_RESULTS.get(request_id)
    if not rec:
        raise HTTPException(status_code=404, detail={
            "error": f"unknown request_id '{request_id}'",
            "note": "Either the stream has not finished yet, or the replica restarted. "
                    "Results are held in memory per replica."})
    return rec


@app.post("/stream")
def tts_stream(req: TTSRequest):
    """Same request body as POST /, but audio arrives as it is generated.

    Returns chunked audio/wav: a streaming RIFF header, then 16-bit PCM. First audio lands
    in roughly 1-2s regardless of how long the passage is, against 152s for a four-minute
    request on the batch route.

    Errors can only be reported before the first byte. Once audio is flowing the status
    fields POST / returns have nowhere to go, so a caller that needs `ok` and `status` should
    use POST / instead.
    """
    try:
        handler = _get_handler()
    except Exception as exc:
        raise HTTPException(status_code=503,
                            detail={"error": f"Model not ready: {exc}", "retry": True,
                                    "note": RETRY_NOTE}) from exc
    try:
        info, generate = handler.stream_prepare(req.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc), "retry": False}) from exc
    except Exception as exc:
        log.exception("stream setup failed")
        raise HTTPException(status_code=500,
                            detail={"error": f"Generation failed: {exc}", "retry": True}) from exc

    # A stream has no JSON body to carry the status fields, so they travel as headers, which
    # are sent BEFORE the first audio byte. voice_loaded and voice_cloned are both known by
    # now -- the reference was accepted and the model is committed to generating from it.
    # audio_generated has no header equivalent: audio arriving IS the signal, and a failure
    # before the first byte is an ordinary 4xx/5xx.
    return StreamingResponse(
        generate(), media_type="audio/wav",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-store",
            "X-Language": str(info["language"]),
            "X-Display": str(info["display"]),
            "X-Voice": str(info["voice"]),
            "X-Voice-Source": str(info["voice_source"]),
            "X-Mode": str(info["mode"]),
            "X-Sample-Rate": str(info["sampling_rate"]),
            "X-Voice-Loaded": "true",
            "X-Voice-Cloned": "true",
            "X-Request-Id": str(info["request_id"]),
            "Access-Control-Expose-Headers":
                "X-Language,X-Display,X-Voice,X-Voice-Source,X-Mode,X-Sample-Rate,"
                "X-Voice-Loaded,X-Voice-Cloned,X-Request-Id",
        })


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
