"""
Custom-container HTTP server for the CosyVoice3 multilang orchestrator, mirroring the
all-lab-tts-endpoint pattern: the cosyvoice/matcha SOURCE CODE is baked into this image
(COPY'd at build time, same as that project's alllab_tts/), so a missing Python dependency
fails the Docker build in ~1 minute instead of a live HF Inference Endpoint deploy (~10 min).
Only the model WEIGHTS (shared assets + per-language llm.pt/flow.pt) are loaded at runtime --
shared assets from the HF model repo HF auto-mounts at /repository, per-language weights
downloaded from their own repos via huggingface_hub.

Same routing/request contract as before: POST /generate with {"inputs": "...", "parameters":
{"language", "prompt_text", "prompt_audio_base64"}}, response {"language", "audio_base64",
"sample_rate", "duration_sec", "peak"}. GET /health for readiness.
"""
import base64
import os
import tempfile

import torch
import torchaudio
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download
from pydantic import BaseModel

from cosyvoice.cli.cosyvoice import CosyVoice3  # noqa: E402

MODEL_DIR = os.environ.get("MODEL_DIR", "/repository")
SHARED_ASSETS = ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "CosyVoice-BlankEN", "hift.pt"]
ORG = "all-lab"
LANGUAGE_REPOS = {
    "ha-NG": f"{ORG}/cosyvoice3-individual-ha-NG",
    "tw-GH": f"{ORG}/cosyvoice3-individual-tw-GH",
    "ig-NG": f"{ORG}/cosyvoice3-individual-ig-NG",
    "ee-GH": f"{ORG}/cosyvoice3-individual-ee-GH",
}

app = FastAPI()
MODELS = {}
READY = False


class GenerateRequest(BaseModel):
    inputs: str
    parameters: dict


@app.on_event("startup")
def load_models():
    token = os.environ.get("HF_TOKEN")
    workdir = tempfile.mkdtemp(prefix="cosyvoice_multilang_")

    for lang, repo_id in LANGUAGE_REPOS.items():
        lang_dir = os.path.join(workdir, lang)
        os.makedirs(lang_dir, exist_ok=True)
        for asset in SHARED_ASSETS:
            dst = os.path.join(lang_dir, asset)
            if not os.path.exists(dst):
                os.symlink(os.path.join(MODEL_DIR, asset), dst)

        try:
            print(f"[server] downloading '{lang}' weights from {repo_id}...", flush=True)
            for f in ("llm.pt", "flow.pt"):
                downloaded = hf_hub_download(repo_id=repo_id, filename=f, token=token)
                os.symlink(downloaded, os.path.join(lang_dir, f))
            print(f"[server] loading '{lang}'...", flush=True)
            MODELS[lang] = CosyVoice3(lang_dir, fp16=False)
        except Exception as e:
            print(f"[server] FAILED to load '{lang}' from {repo_id}: {e}", flush=True)

    global READY
    READY = len(MODELS) > 0
    print(f"[server] ready={READY}, languages={sorted(MODELS.keys())}", flush=True)


@app.get("/health")
def health():
    """HF probes this. 503 until at least one language actually loaded -- avoids the
    platform reporting 'running' while the container is really still failing every
    language load in the background."""
    if READY:
        return {"status": "ok", "languages": sorted(MODELS.keys())}
    return JSONResponse(status_code=503, content={"status": "loading", "languages": []})


@app.post("/generate")
def generate(req: GenerateRequest):
    params = req.parameters or {}
    language = params.get("language")
    if language not in MODELS:
        return {"error": f"unknown or missing 'language' parameter, available: {sorted(MODELS.keys())}"}

    target_text = req.inputs
    prompt_text = params.get("prompt_text")
    prompt_audio_b64 = params.get("prompt_audio_base64")
    if not target_text or not prompt_text or not prompt_audio_b64:
        return {"error": "requires inputs (target text), parameters.language, parameters.prompt_text, parameters.prompt_audio_base64"}

    if not prompt_text.endswith("<|endofprompt|>"):
        prompt_text = prompt_text + "<|endofprompt|>"

    model = MODELS[language]
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(base64.b64decode(prompt_audio_b64))
        tmp.flush()

        audio, last_err = None, None
        for _ in range(4):
            try:
                results = list(model.inference_zero_shot(target_text, prompt_text, tmp.name, stream=False))
                audio = results[0]["tts_speech"]
                break
            except RuntimeError as e:
                last_err = e
        if audio is None:
            return {"error": f"synthesis failed after retries: {last_err}"}

    out_path = tempfile.mktemp(suffix=".wav")
    torchaudio.save(out_path, audio, model.sample_rate)
    with open(out_path, "rb") as f:
        audio_bytes = f.read()
    os.remove(out_path)

    return {
        "language": language,
        "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
        "sample_rate": model.sample_rate,
        "duration_sec": audio.shape[1] / model.sample_rate,
        "peak": audio.abs().max().item(),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
