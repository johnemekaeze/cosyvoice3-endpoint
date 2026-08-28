# Hugging Face Inference Endpoints — custom container (NOT a Space).
# Built automatically by .github/workflows/build-and-push.yml on every push
# to main -> ghcr.io/johnemekaeze/cosyvoice3-endpoint:latest
# Deploy: Inference Endpoints -> Custom Container -> that image URL, port 8080,
#         health route /health. Set secret HF_TOKEN (read access to the
#         all-lab/cosyvoice3-individual-* weight repos handler.py routes to).
#
# Structured after ms0017/all-lab-tts-endpoint (OmniVoice): the library source is
# COPY'd into the image, only weights come from the Hub at runtime.

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PORT=8080 \
    MODEL_DIR=/repository \
    HF_HOME=/hfcache \
    HUGGINGFACE_HUB_CACHE=/hfcache/hub

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      libsndfile1 \
      git \
      g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# openai-whisper's legacy setup.py needs pkg_resources at build time, and pip's isolated
# build env fetches its OWN fresh setuptools regardless of what is installed here --
# --no-build-isolation makes it use the pinned one instead.
RUN pip install --upgrade pip \
 && pip install "setuptools<81" \
 && pip install --no-build-isolation openai-whisper==20231117

COPY requirements.txt .
RUN pip install -r requirements.txt

# openai-whisper's dependency metadata resolves torch back down to 2.3.1, which leaves an
# ABI-mismatched torchaudio ("undefined symbol: torch::autograd::Node::name") at import
# time. Reinstall the matched pair LAST so nothing downstream can displace it.
RUN pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu128 \
      torch==2.8.0 torchaudio==2.8.0

COPY cosyvoice ./cosyvoice
COPY matcha ./matcha
COPY assets ./assets
COPY handler.py app.py ./

# cosyvoice3.yaml instantiates classes by string name (HyperPyYAML !name:/!new:), so a
# missing dependency would otherwise only surface deep inside model loading on the live
# endpoint. Resolve the whole graph here instead: a broken image fails the build.
RUN python -c "\
from cosyvoice.cli.cosyvoice import CosyVoice3; \
import handler, torch, torchaudio; \
print('import ok: torch', torch.__version__, 'torchaudio', torchaudio.__version__, 'cuda', torch.version.cuda); \
print('languages:', sorted(handler.MODELS)); \
print('prompts:', sorted(handler._load_bundled_prompts()))"

EXPOSE 8080

# Do not bake the language packs into the image — load from the Hub at runtime.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
