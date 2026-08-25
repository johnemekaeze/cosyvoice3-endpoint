# Hugging Face Inference Endpoints -- custom container (NOT a Space).
# Built automatically by .github/workflows/build-hf-endpoint-image.yml on every push to main
# -> ghcr.io/<owner>/cosyvoice3-endpoint:latest. Deploy: Inference Endpoints -> Custom
# Container -> that image URL, port 80, health route /health. Set secret HF_TOKEN (read
# access to the all-lab/cosyvoice3-individual-* repos handler.py routes to).
#
# Mirrors the proven all-lab-tts-endpoint pattern (same org, ms0017/all-lab-tts-endpoint) --
# GHCR + GITHUB_TOKEN instead of Docker Hub (no separate registry credentials, no local
# Docker/network dependency), port 80 (HF IE's default health/probe port), and this exact
# newer CUDA base -- that project reports zero CUDA/driver mismatch issues on the same HF T4
# infra, unlike torch==2.3.1+cu121 here which hit "libcudart.so.13 not found" (host wants a
# newer CUDA runtime than that build provides).
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PORT=80 \
    MODEL_DIR=/repository

RUN apt-get update -y && apt-get install -y --no-install-recommends ffmpeg git g++ libsndfile1 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "setuptools<81"

# openai-whisper's legacy setup.py needs pkg_resources at build time -- pip's isolated build
# env for it fetches its OWN fresh setuptools regardless of what's already installed above,
# so --no-build-isolation makes it use the pinned one instead.
RUN pip install --no-cache-dir --no-build-isolation openai-whisper==20231117

RUN pip install --no-cache-dir --timeout=180 --retries=10 onnxruntime-gpu==1.18.0

RUN pip install --no-cache-dir --timeout=180 --retries=10 \
    transformers==4.51.3 \
    diffusers==0.29.0 \
    lightning==2.2.4

RUN pip install --no-cache-dir --timeout=180 --retries=10 \
    fastapi==0.115.6 \
    "uvicorn[standard]"==0.30.0 \
    huggingface_hub \
    hf_transfer \
    rich \
    conformer==0.3.2 \
    x-transformers==2.11.24 \
    hydra-core==1.3.2 \
    HyperPyYAML==1.2.3 \
    omegaconf==2.3.0 \
    inflect==7.3.1 \
    librosa==0.10.2 \
    soundfile==0.12.1 \
    scipy \
    einops \
    tiktoken \
    wetext==0.0.4 \
    networkx==3.1 \
    onnx==1.16.0 \
    protobuf==4.25 \
    pydantic==2.7.0 \
    pyworld==0.3.4 \
    matplotlib==3.7.5

EXPOSE 80
ENTRYPOINT ["python", "/repository/hf_endpoint_server.py"]
