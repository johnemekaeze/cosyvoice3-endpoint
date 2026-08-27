# CosyVoice3 — HF Inference Endpoint (custom container)

Multi-language CosyVoice3 TTS for African Languages Lab: 24 African languages.
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
  -d '{"inputs": "Sannu da safe", "language": "hausa"}'
```

`language` takes the plain language name: `hausa`, `twi`, `igbo`, `ewe`, `berber`, `umbundu`,
`amharic`, `arabic`, `fula`, `luganda`, `lingala`, `malagasy`, `sepedi`, `chichewa`, `oromo`,
`somali`, `sesotho`, `swahili`, `tigrinya`, `tswana`, `tsonga`, `venda`, `xhosa`, `zulu`.
Case-insensitive; common alternatives (`isizulu`, `kiswahili`, ...) and legacy ISO-style codes
(`ha-NG`, `sw-KE`, ...) are also accepted.

| code | language | code | language | code | language |
|---|---|---|---|---|---|
| `ha-NG` | Hausa | `am-ET` | Amharic | `so-SO` | Somali |
| `tw-GH` | Twi | `ar-AR` | Arabic | `st-ZA` | Sesotho |
| `ig-NG` | Igbo | `ff-SN` | Fula | `sw-KE` | Swahili |
| `ee-GH` | Ewe | `lg-UG` | Luganda | `ti-ER` | Tigrinya |
| `ber-MA` | Berber | `ln-CD` | Lingala | `tn-BW` | Tswana |
| `umb-AO` | Umbundu | `mg-MG` | Malagasy | `ts-ZA` | Tsonga |
| `nso-ZA` | Sepedi | `ny-MW` | Chichewa | `ve-ZA` | Venda |
| `or-KE` | Oromo | `xh-ZA` | Xhosa | `zu-ZA` | Zulu |

Every one of these is the exact checkpoint pair that produced a verified-clean sample in the
best-checkpoint audit. Languages that failed that audit (af-ZA, en-UG, ki-KE, nd-ZW, rw-RW,
yo-NG, sn-ZW, wo-SN, bem-ZM) are deliberately not served.

CosyVoice3 is a zero-shot cloning model, so every request synthesizes in the voice of a
reference clip. A validated reference is bundled per language, so no extra arguments are
needed. To clone a different voice, pass `prompt_audio_base64` (WAV, base64) together with
`prompt_text` (its transcript).

Response: `{"language", "display", "sampling_rate", "duration_sec", "audio_base64", "format"}`.

## Notes

- Only the default language (`hausa`, override with `COSYVOICE_DEFAULT_LANGUAGE`) is warmed
  at startup; the others load on first request for that language. That first call to a new
  language downloads ~5GB of weights and takes ~2.5 minutes; later calls to it are ~11s.
- GHCR packages default to private. Make the package public via the web UI
  (Package settings -> Danger Zone -> Change visibility) or the endpoint cannot pull it.
