# CosyVoice3 — HF Inference Endpoint (custom container)

Multi-language CosyVoice3 TTS for African Languages Lab: Hausa, Twi, Igbo, Ewe.
One endpoint, one model in VRAM at a time, switching on demand.

Structured after `ms0017/all-lab-tts-endpoint` (OmniVoice): the `cosyvoice/` +
`matcha/` library source is baked into the image, and only the per-language weight
packs are pulled from the Hub at runtime.

## Deploy

Inference Endpoints -> Custom Container:

- Image: `ghcr.io/johnemekaeze/cosyvoice3-endpoint:latest`
- Port: `8080`
- Health route: `/health`
- Secret: `HF_TOKEN` (read access to `all-lab/cosyvoice3-individual-*`)

## Request

```bash
curl https://<endpoint>.endpoints.huggingface.cloud/ \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"inputs": "Sannu da safe", "language": "ha-NG"}'
```

`language` accepts `ha-NG | tw-GH | ig-NG | ee-GH` (or `hausa|twi|igbo|ewe`).

CosyVoice3 is a zero-shot cloning model, so every request synthesizes in the voice of a
reference clip. A validated reference is bundled per language, so no extra arguments are
needed. To clone a different voice, pass `prompt_audio_base64` (WAV, base64) together with
`prompt_text` (its transcript).

Response: `{"language", "display", "sampling_rate", "duration_sec", "audio_base64", "format"}`.

## Notes

- Only the default language (`ha-NG`, override with `COSYVOICE_DEFAULT_LANGUAGE`) is warmed
  at startup; the others load on first request for that language.
- GHCR packages default to private. Make the package public via the web UI
  (Package settings -> Danger Zone -> Change visibility) or the endpoint cannot pull it.
