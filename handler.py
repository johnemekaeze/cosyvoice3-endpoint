"""Custom Hugging Face Inference Endpoint handler — African Languages Lab CosyVoice3 TTS.

Multi-language CosyVoice3 router: one endpoint, 4 private weight packs. Keeps one
language loaded in VRAM at a time; switching language unloads the current pack and
loads the requested one. Mirrors the all-lab-tts-endpoint (OmniVoice) handler.

Each all-lab/cosyvoice3-individual-* repo is a complete self-contained CosyVoice3
bundle (llm.pt, flow.pt, hift.pt, campplus.onnx, speech_tokenizer_v3.onnx,
cosyvoice3.yaml, CosyVoice-BlankEN/), so loading is a snapshot_download of the repo
followed by CosyVoice3(local_dir) -- the direct analogue of OmniVoice.from_pretrained.
The cosyvoice/ + matcha/ library source is baked into the image instead, so the repo
snapshot only needs to carry weights and config.

CosyVoice3 is a zero-shot voice-cloning model: it always synthesizes in the voice of a
reference clip. A validated reference prompt is bundled per language (assets/prompts/),
so a bare {"inputs": "...", "language": "ha-NG"} works with no extra arguments; callers
who want a different voice can pass their own prompt_audio_base64 + prompt_text.

Payload:
    {"inputs": "Sannu da safe", "language": "ha-NG"}
    optional: "prompt_text", "prompt_audio_base64" (override the bundled voice)

Response:
    {"language": "ha-NG", "sampling_rate": 24000, "duration_sec": 1.23,
     "audio_base64": "...", "format": "wav"}
"""

from __future__ import annotations

import base64
import gc
import io
import json
import logging
import os
import tempfile
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf
import torch

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cosyvoice3-endpoint")

# Every language here is backed by a published all-lab/cosyvoice3-individual-* repo holding
# the exact llm+flow checkpoint pair that produced a verified-clean sample in the
# best-checkpoint audit (full-length output, no high-frequency artefacts, healthy level).
# Deliberately excluded, having failed that audit: af-ZA / en-UG / ki-KE / nd-ZW / rw-RW /
# yo-NG (collapse to well under a second), sn-ZW (audible hiss), wo-SN (~30dB too quiet)
# and bem-ZM (generation errored outright).
_L = "all-lab/cosyvoice3-individual-{}"
MODELS: Dict[str, Dict[str, str]] = {
    "hausa": {"repo": _L.format("hausa"), "display": "Hausa"},
    "twi": {"repo": _L.format("twi"), "display": "Twi"},
    "igbo": {"repo": _L.format("igbo"), "display": "Igbo"},
    "ewe": {"repo": _L.format("ewe"), "display": "Ewe"},
    "berber": {"repo": _L.format("berber"), "display": "Berber (Tamazight)"},
    "umbundu": {"repo": _L.format("umbundu"), "display": "Umbundu"},
    "amharic": {"repo": _L.format("amharic"), "display": "Amharic"},
    "arabic": {"repo": _L.format("arabic"), "display": "Arabic"},
    "fula": {"repo": _L.format("fula"), "display": "Fula"},
    "luganda": {"repo": _L.format("luganda"), "display": "Luganda"},
    "lingala": {"repo": _L.format("lingala"), "display": "Lingala"},
    "malagasy": {"repo": _L.format("malagasy"), "display": "Malagasy"},
    "sepedi": {"repo": _L.format("sepedi"), "display": "Sepedi"},
    "chichewa": {"repo": _L.format("chichewa"), "display": "Chichewa"},
    "oromo": {"repo": _L.format("oromo"), "display": "Oromo"},
    "somali": {"repo": _L.format("somali"), "display": "Somali"},
    "sesotho": {"repo": _L.format("sesotho"), "display": "Sesotho"},
    "swahili": {"repo": _L.format("swahili"), "display": "Swahili"},
    "tigrinya": {"repo": _L.format("tigrinya"), "display": "Tigrinya"},
    "tswana": {"repo": _L.format("tswana"), "display": "Tswana"},
    "tsonga": {"repo": _L.format("tsonga"), "display": "Tsonga"},
    "venda": {"repo": _L.format("venda"), "display": "Venda"},
    "xhosa": {"repo": _L.format("xhosa"), "display": "Xhosa"},
    "zulu": {"repo": _L.format("zulu"), "display": "Zulu"},
}

# Legacy ISO-style codes (ha-NG, sw-KE, ...) remain accepted so anything already written
# against them keeps working; plain language names are the canonical form.
ALIASES: Dict[str, str] = {
    "afaan oromo": "oromo",
    "am": "amharic",
    "am-et": "amharic",
    "ar": "arabic",
    "ar-ar": "arabic",
    "ber": "berber",
    "ber-ma": "berber",
    "chewa": "chichewa",
    "ee": "ewe",
    "ee-gh": "ewe",
    "ff": "fula",
    "ff-sn": "fula",
    "fulani": "fula",
    "ganda": "luganda",
    "ha": "hausa",
    "ha-ng": "hausa",
    "ig": "igbo",
    "ig-ng": "igbo",
    "isixhosa": "xhosa",
    "isizulu": "zulu",
    "kiswahili": "swahili",
    "lg": "luganda",
    "lg-ug": "luganda",
    "ln": "lingala",
    "ln-cd": "lingala",
    "mg": "malagasy",
    "mg-mg": "malagasy",
    "northern sotho": "sepedi",
    "nso": "sepedi",
    "nso-za": "sepedi",
    "ny": "chichewa",
    "ny-mw": "chichewa",
    "nyanja": "chichewa",
    "or": "oromo",
    "or-ke": "oromo",
    "pedi": "sepedi",
    "pulaar": "fula",
    "setswana": "tswana",
    "so": "somali",
    "so-so": "somali",
    "sotho": "sesotho",
    "st": "sesotho",
    "st-za": "sesotho",
    "sw": "swahili",
    "sw-ke": "swahili",
    "tamazight": "berber",
    "ti": "tigrinya",
    "ti-er": "tigrinya",
    "tn": "tswana",
    "tn-bw": "tswana",
    "ts": "tsonga",
    "ts-za": "tsonga",
    "tshivenda": "venda",
    "tw": "twi",
    "tw-gh": "twi",
    "umb": "umbundu",
    "umb-ao": "umbundu",
    "ve": "venda",
    "ve-za": "venda",
    "xh": "xhosa",
    "xh-za": "xhosa",
    "xitsonga": "tsonga",
    "zu": "zulu",
    "zu-za": "zulu",
}

DEFAULT_LANGUAGE = os.environ.get("COSYVOICE_DEFAULT_LANGUAGE", "hausa").strip()
DEFAULT_VOICE = os.environ.get("COSYVOICE_DEFAULT_VOICE", "female").strip().lower()

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_DIR = os.path.join(APP_DIR, "assets", "prompts")

# The library source is baked into the image, so there is no reason to pull those
# files again inside every model snapshot -- only weights and config are needed.
_SNAPSHOT_IGNORE = ["cosyvoice/*", "third_party/*", "handler.py", "requirements.txt", ".gitattributes"]


def _hf_token() -> Optional[str]:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("huggingface_token")
    )


def _resolve_language(language: Optional[str]) -> str:
    """Accept the canonical plain name in any case ("Zulu", "zulu"), a friendly
    alias ("isizulu", "kiswahili"), or a legacy ISO-style code ("zu-ZA")."""
    key = (language or DEFAULT_LANGUAGE).strip()
    if key in MODELS:
        return key
    lowered = key.lower()
    if lowered in MODELS:
        return lowered
    if lowered in ALIASES:
        return ALIASES[lowered]
    raise ValueError(f"Unsupported language '{language}'. Choose: {sorted(MODELS)}")



def _load_bundled_prompts() -> Dict[str, Dict[str, str]]:
    path = os.path.join(PROMPT_DIR, "prompts.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        log.warning("no bundled prompts at %s: %s", path, exc)
        return {}


class EndpointHandler:
    """Loaded once per replica by HF's Inference Endpoint runtime."""

    def __init__(self, model_dir: str = "") -> None:
        token = _hf_token()
        if token:
            from huggingface_hub import login

            login(token=token, add_to_git_credential=False)

        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._prompts = _load_bundled_prompts()
        self._cache: Dict[str, Any] = {"name": None, "model": None}

        # Warm the default language at container start so the *first* real request
        # after a cold start isn't also paying the model-load cost on top of the
        # container boot cost. Only ONE language is loaded -- the rest load lazily.
        try:
            self._get_model(DEFAULT_LANGUAGE)
        except Exception as exc:
            log.warning("warmup load of %s failed (will retry on first request): %s", DEFAULT_LANGUAGE, exc)

    def _unload(self) -> None:
        if self._cache.get("model") is not None:
            del self._cache["model"]
        self._cache["model"] = None
        self._cache["name"] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_model(self, language: str):
        key = _resolve_language(language)
        if self._cache["name"] == key and self._cache["model"] is not None:
            return self._cache["model"], MODELS[key]

        from huggingface_hub import snapshot_download

        from cosyvoice.cli.cosyvoice import CosyVoice3

        self._unload()
        repo = MODELS[key]["repo"]
        log.info("downloading %s from %s", key, repo)
        local_dir = snapshot_download(repo_id=repo, token=_hf_token(), ignore_patterns=_SNAPSHOT_IGNORE)
        log.info("loading %s on %s", key, self._device)
        model = CosyVoice3(local_dir, fp16=False)
        self._cache["name"] = key
        self._cache["model"] = model
        log.info("loaded %s", key)
        return model, MODELS[key]

    def _prompt_for(self, key: str, data: Dict[str, Any]) -> tuple:
        """Decide which voice to clone.

        Order of preference:
          1. a voice the caller uploaded on this request (prompt_audio_base64 + prompt_text);
          2. the built-in preset for the requested gender ("voice": "male"/"female");
          3. the language's default preset.

        Returns (prompt_text, prompt_wav_path, cleanup_path_or_None, source, voice_id).
        """
        parameters = data.get("parameters") or {}
        audio_b64 = data.get("prompt_audio_base64") or parameters.get("prompt_audio_base64")
        text = data.get("prompt_text") or parameters.get("prompt_text")

        if audio_b64:
            if not text:
                raise ValueError("`prompt_text` is required when `prompt_audio_base64` is supplied")
            try:
                raw = base64.b64decode(audio_b64, validate=True)
            except Exception as exc:
                raise ValueError(f"prompt_audio_base64 is not valid base64: {exc}") from exc
            if len(raw) < 1024:
                raise ValueError("prompt_audio_base64 decoded to less than 1KB -- not a usable WAV")
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(raw)
            tmp.close()
            try:
                info = sf.info(tmp.name)
            except Exception as exc:
                os.remove(tmp.name)
                raise ValueError(f"uploaded audio could not be read as WAV: {exc}") from exc
            if info.duration < 1.0:
                os.remove(tmp.name)
                raise ValueError(f"uploaded audio is only {info.duration:.2f}s -- use 5-15s")
            return text, tmp.name, tmp.name, "uploaded", "uploaded"

        requested = (data.get("voice") or parameters.get("voice") or "").strip().lower()
        bundled = self._prompts.get(key)
        if not bundled:
            raise ValueError(
                f"no bundled reference voice for '{key}' -- supply prompt_audio_base64 and prompt_text"
            )

        # prompts.json is either {"prompt_text","prompt_wav"} (single preset) or
        # {"male": {...}, "female": {...}} once gendered presets exist.
        if "prompt_wav" in bundled:
            entry, voice_id = bundled, "default"
        else:
            available = sorted(k for k in bundled if k in ("male", "female"))
            if requested in bundled:
                entry, voice_id = bundled[requested], requested
            elif requested:
                raise ValueError(
                    f"voice '{requested}' not available for '{key}'; available: {available}"
                )
            else:
                pick = DEFAULT_VOICE if DEFAULT_VOICE in bundled else available[0]
                entry, voice_id = bundled[pick], pick
        return entry["prompt_text"], os.path.join(PROMPT_DIR, entry["prompt_wav"]), None, "preset", voice_id


    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        text = (data.get("inputs") or data.get("text") or "").strip()
        if not text:
            raise ValueError("`inputs` (text to synthesize) is required")

        key = _resolve_language(data.get("language"))
        model, meta = self._get_model(key)

        # stage 1 -- the reference voice was accepted (uploaded and readable, or a preset)
        prompt_text, prompt_wav, cleanup, source, voice_id = self._prompt_for(key, data)
        voice_loaded = True

        # CosyVoice3 uses this sentinel to mark where the reference transcript ends;
        # it must be appended explicitly, it is not implied.
        if not prompt_text.endswith("<|endofprompt|>"):
            prompt_text = prompt_text + "<|endofprompt|>"

        # stage 2 -- the voice was actually cloned and speech came back
        try:
            results = list(model.inference_zero_shot(text, prompt_text, prompt_wav, stream=False))
        finally:
            if cleanup:
                try:
                    os.remove(cleanup)
                except OSError:
                    pass
        if not results:
            raise RuntimeError("synthesis produced no audio")
        voice_cloned = True

        wav = results[0]["tts_speech"].squeeze(0).cpu().numpy().astype(np.float32)
        sr = int(model.sample_rate)
        duration = float(len(wav) / sr)
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0

        # stage 3 -- the audio is real rather than silence or a collapsed generation.
        # CosyVoice3 occasionally returns a fraction of a second regardless of input length,
        # so check against the text rather than against zero.
        expected_min = min(0.9, 0.06 * max(len(text.split()), 1))
        audio_ok = bool(duration >= expected_min and peak > 0.02)

        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV")
        raw = buf.getvalue()

        return {
            "language": key,
            "display": meta["display"],
            "voice": voice_id,
            "voice_source": source,
            "sampling_rate": sr,
            "duration_sec": duration,
            "peak": round(peak, 4),
            "audio_base64": base64.b64encode(raw).decode("ascii"),
            "format": "wav",
            "status": {
                "voice_loaded": voice_loaded,
                "voice_cloned": voice_cloned,
                "audio_generated": audio_ok,
            },
            "ok": bool(voice_loaded and voice_cloned and audio_ok),
        }
