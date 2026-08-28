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
# how many times to re-roll a collapsed generation before giving up
GEN_ATTEMPTS = int(os.environ.get("COSYVOICE_GEN_ATTEMPTS", "4"))

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


# Voices uploaded through POST /voice, kept for the life of the replica so a caller can
# upload once and then synthesize repeatedly by voice_id.
VOICE_STORE: Dict[str, Dict[str, Any]] = {}


def _decode_voice(audio_b64: str, prompt_text: Optional[str] = None) -> str:
    """Decode an uploaded clip to a temp wav, failing loudly if it is not usable."""
    try:
        raw = base64.b64decode(audio_b64, validate=True)
    except Exception as exc:
        raise ValueError(f"prompt_audio_base64 is not valid base64: {exc}") from exc
    if len(raw) < 1024:
        raise ValueError("uploaded audio decoded to under 1KB -- not a usable WAV")
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
        raise ValueError(f"uploaded audio is only {info.duration:.2f}s -- use a 5-15s clip")
    if info.duration > 30.0:
        os.remove(tmp.name)
        raise ValueError(f"uploaded audio is {info.duration:.1f}s -- the model accepts at most 30s")
    return tmp.name


def register_voice(audio_b64: str, prompt_text: Optional[str] = None,
                   name: Optional[str] = None) -> Dict[str, Any]:
    """Validate and store an uploaded voice, returning its id and measured properties."""
    import uuid
    path = _decode_voice(audio_b64)
    info = sf.info(path)
    vid = uuid.uuid4().hex[:12]
    VOICE_STORE[vid] = {"path": path, "prompt_text": (prompt_text or "").strip() or None,
                        "name": name, "duration_sec": round(info.duration, 2),
                        "sample_rate": info.samplerate}
    return {"voice_id": vid, "uploaded": True, "duration_sec": round(info.duration, 2),
            "sample_rate": info.samplerate, "channels": info.channels,
            "has_transcript": bool(prompt_text),
            "mode": "zero_shot" if prompt_text else "cross_lingual",
            "note": ("Voice stored. Use it by sending \"voice_id\": \"%s\" with your text. "
                     "Without a transcript the reference is never spoken." % vid)}


class EndpointHandler:
    """Loaded once per replica by HF's Inference Endpoint runtime."""

    def _preset_transcript(self, key: str, voice_id: str) -> str:
        """Reference transcript for a preset, needed so the <|endofprompt|> marker has
        something to delimit. An uploaded clip without a transcript falls back to this."""
        b = self._prompts.get(key) or {}
        if "prompt_text" in b:
            return b["prompt_text"]
        e = b.get(voice_id) or next((v for v in b.values() if isinstance(v, dict)), {})
        return e.get("prompt_text", "")

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
        # Each switch downloads ~5GB into the HF cache and nothing reclaims it, which is how
        # the replica kept hitting "Memory limit exceeded (15.0G)". An earlier attempt deleted
        # the snapshot here and produced something worse -- "Cannot copy out of meta tensor"
        # on the very next load, because CosyVoice3 still reads from those files after
        # construction. So the cache is left alone; the memory ceiling is handled by keeping
        # one model resident, and is being addressed at the instance level instead.
        self._cache.pop("snapshot", None)
        gc.collect()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _get_model(self, language: str):
        key = _resolve_language(language)
        if self._cache["name"] == key and self._cache["model"] is not None:
            return self._cache["model"], MODELS[key]

        from huggingface_hub import snapshot_download

        from cosyvoice.cli.cosyvoice import CosyVoice3

        self._unload()
        repo = MODELS[key]["repo"]
        log.info("downloading %s from %s", key, repo)
        try:
            local_dir = snapshot_download(repo_id=repo, token=_hf_token(), ignore_patterns=_SNAPSHOT_IGNORE)
            log.info("loading %s on %s", key, self._device)
            model = CosyVoice3(local_dir, fp16=False)
        except Exception as exc:
            # a failed switch must not leave a half-loaded model behind, and must not take
            # the replica down -- the caller gets a retryable error instead
            self._unload()
            log.exception("failed to load %s", key)
            raise RuntimeError(f"could not load the {key} model: {exc}") from exc
        self._cache["name"] = key
        self._cache["model"] = model
        self._cache["snapshot"] = local_dir
        log.info("loaded %s", key)
        return model, MODELS[key]

    def _prompt_for(self, key: str, data: Dict[str, Any]) -> tuple:
        """Decide which voice to clone, and whether a reference transcript is used.

        Order of preference:
          1. a voice uploaded on this request (prompt_audio_base64);
          2. the built-in preset for the requested gender ("voice": "male"/"female");
          3. the language's default preset.

        prompt_text comes back as None whenever we have no transcript we trust. That
        matters: with a transcript the model runs zero-shot, which conditions the LLM on
        the reference text and can end up SPEAKING it before the requested text. Without
        one it runs cross-lingual, which keeps the speaker identity but drops the
        reference text from the LLM entirely, so the reference can never be read aloud.

        Returns (prompt_text_or_None, wav_path, cleanup_path_or_None, source, voice_id).
        """
        parameters = data.get("parameters") or {}
        audio_b64 = data.get("prompt_audio_base64") or parameters.get("prompt_audio_base64")
        text = data.get("prompt_text") or parameters.get("prompt_text")
        voice_id = data.get("voice_id") or parameters.get("voice_id")

        # a previously uploaded voice, referenced by id
        if voice_id:
            rec = VOICE_STORE.get(voice_id)
            if not rec:
                raise ValueError(f"unknown voice_id '{voice_id}'. Upload one at POST /voice first.")
            return rec.get("prompt_text"), rec["path"], None, "uploaded", voice_id

        if audio_b64:
            path = _decode_voice(audio_b64)
            # transcript is optional: without it we simply use cross-lingual mode
            return (text.strip() if text else None), path, path, "uploaded", "uploaded"

        requested = (data.get("voice") or parameters.get("voice") or "").strip().lower()
        bundled = self._prompts.get(key)
        if not bundled:
            raise ValueError(
                f"no bundled reference voice for '{key}' -- supply prompt_audio_base64"
            )

        if "prompt_wav" in bundled:
            entry, vid = bundled, "default"
        else:
            available = sorted(k for k in bundled if k in ("male", "female"))
            if requested in bundled:
                entry, vid = bundled[requested], requested
            elif requested in ("male", "female"):
                other = [v for v in available if v != requested]
                alt = (f" A {other[0]} voice is available for {key}: send "
                       f'"voice": "{other[0]}".') if other else ""
                raise ValueError(
                    f"No {requested} voice is available for {key} -- no suitable {requested} "
                    f"recording was found in the source corpus for this language.{alt}"
                )
            elif requested:
                raise ValueError(
                    f"voice '{requested}' is not recognised; use 'male' or 'female'. "
                    f"Available for {key}: {available}"
                )
            else:
                pick = DEFAULT_VOICE if DEFAULT_VOICE in bundled else available[0]
                entry, vid = bundled[pick], pick
        return entry["prompt_text"], os.path.join(PROMPT_DIR, entry["prompt_wav"]), None, "preset", vid


    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        text = (data.get("inputs") or data.get("text") or "").strip()
        if not text:
            raise ValueError("`inputs` (text to synthesize) is required")

        key = _resolve_language(data.get("language"))
        model, meta = self._get_model(key)

        # stage 1 -- the reference voice was accepted (uploaded and readable, or a preset)
        prompt_text, prompt_wav, cleanup, source, voice_id = self._prompt_for(key, data)
        voice_loaded = True

        # stage 2 -- clone the voice and synthesize ONLY the requested text.
        #
        # CosyVoice3 requires the <|endofprompt|> marker: inference_cross_lingual omits
        # prompt_text entirely and asserts out ("<|endofprompt|> not detected"), so it is
        # not usable here. The marker is precisely what tells the LLM where the reference
        # ends, so supplying it correctly is what keeps the reference from being spoken.
        if not prompt_text:
            prompt_text = self._preset_transcript(key, voice_id)
        prompt_text = prompt_text.strip()
        if not prompt_text.endswith("<|endofprompt|>"):
            prompt_text = prompt_text + "<|endofprompt|>"
        mode = "zero_shot"
        sr = int(model.sample_rate)
        # Generation is stochastic and collapses to a fraction of a second every so often --
        # measured on a good reference clip, the same request gave 5.12s on one run and 0.08s
        # on the next. It is not the clip: the identical tswana file scored 0/3 in one sweep
        # and 2/3 in another. So retry a collapse here rather than handing the caller a
        # broken clip and asking them to notice.
        expected_min = max(0.5, min(1.2, 0.12 * len(text.split())))
        wav, attempts, last = None, 0, None
        try:
            for attempts in range(1, GEN_ATTEMPTS + 1):
                try:
                    results = list(model.inference_zero_shot(text, prompt_text, prompt_wav, stream=False))
                except Exception as exc:   # transient CUDA/shape faults also retry
                    last = exc
                    log.warning("attempt %d failed for %s: %s", attempts, key, exc)
                    continue
                if not results:
                    last = RuntimeError("no audio returned")
                    continue
                cand = results[0]["tts_speech"].squeeze(0).cpu().numpy().astype(np.float32)
                dur = len(cand) / sr
                pk = float(np.max(np.abs(cand))) if cand.size else 0.0
                if wav is None or dur > len(wav) / sr:
                    wav = cand
                if dur >= expected_min and pk > 0.02:
                    break
                log.warning("collapsed output %.2fs for %s (attempt %d), retrying", dur, key, attempts)
        finally:
            if cleanup:
                try:
                    os.remove(cleanup)
                except OSError:
                    pass
        if wav is None:
            raise RuntimeError(f"synthesis failed after {GEN_ATTEMPTS} attempts: {last}")
        voice_cloned = True

        duration = float(len(wav) / sr)
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        audio_ok = bool(duration >= expected_min and peak > 0.02)

        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV")
        raw = buf.getvalue()

        return {
            "language": key,
            "display": meta["display"],
            "voice": voice_id,
            "voice_source": source,
            "mode": mode,
            "attempts": attempts,
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
