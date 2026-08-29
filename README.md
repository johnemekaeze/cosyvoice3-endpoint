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

## Behaviour worth knowing

**Generation is stochastic.** CosyVoice3 occasionally collapses to a fraction of a second
regardless of the input -- the same request measured 5.12s on one run and 0.08s on the next,
and an identical reference clip scored 0/3 in one sweep and 2/3 in another. The endpoint
re-rolls a collapsed generation up to `COSYVOICE_GEN_ATTEMPTS` (default 4) times, and reports
how many it took in `attempts`. If it still fails, `status.audio_generated` and `ok` are
false, so a caller can tell rather than shipping a broken clip.

**Very short input collapses more often.** CosyVoice warns when the requested text is under
half the length of the reference transcript, and that is the regime where collapse is most
frequent. A full sentence behaves far better than one or two words.

**The reference transcript is required and cannot be shortened.** Measured directly:
supplying only the `<|endofprompt|>` marker, or a truncated prompt, collapses generation at
every target length. The full transcript is what makes it work.

**Where `<|endofprompt|>` goes decides whether the reference is spoken aloud.** The marker
ends an *instruction preamble*; the reference transcript belongs after it, exactly as in
CosyVoice3's own examples:

```python
inference_zero_shot(tts_text,
                    "You are a helpful assistant.<|endofprompt|>" + reference_transcript,
                    prompt_wav)
```

`llm.py` concatenates the two halves into a single sequence (`text = cat[prompt_text, text]`),
so the marker is the only thing telling the model which span the reference audio already
covers. This endpoint previously *appended* the marker -- transcript first, marker last --
which left the transcript in the instruction slot with nothing after the marker matching the
audio, and the model spoke the transcript before the requested text. Both forms satisfy the
`EOP_TOKEN_ID` assert, so the bug was silent rather than an error.

Measured over all 48 deployed voices, with expected length calibrated from each voice's own
speaking rate: appending leaked on **11 of 48** (worst, chichewa/male: 2.7 reference-lengths
of unexplained audio); the correct ordering leaks on **4 of 48**. No property of the
reference clip predicted it -- duration, characters per second, target/prompt ratio and
punctuation all failed to separate leaking voices from clean ones. The cause is the
fine-tuning data: its `text` column holds bare transcripts with no marker and no preamble,
so the model was taught to speak every text token it is handed.

The endpoint now also rejects a leaked take and re-rolls it, using an upper duration bound
derived from the voice's own speaking rate. Note the earlier retry kept the *longest*
candidate, which actively selected leaked takes; it now keeps the one closest to the
expected length.

**`inference_cross_lingual` does work**, contrary to an earlier note here -- the marker must
travel on the `tts_text` argument instead, which is what the released examples do:
`inference_cross_lingual("You are a helpful assistant.<|endofprompt|>" + text, prompt_wav)`.
It passes no transcript at all, so nothing can leak; it scored 4 of 48 as well, with no
collapses but one worse outlier. Zero-shot with the corrected ordering is what ships.

**Reference audio must be at least 16 kHz.** CosyVoice3's `load_wav` asserts on anything
lower (its error message says 24000, but the check is against 16000).

**Language switching is expensive.** One model is held at a time; switching downloads ~5GB
and takes 1-2 minutes. The container has a 15GB memory ceiling and several switches in
quick succession can still exhaust it. Deleting the downloaded copy on unload was tried and
reverted -- CosyVoice3 keeps reading those files after construction, so removing them broke
the next load outright ("Cannot copy out of meta tensor"). A larger instance is the fix.
