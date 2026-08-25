# Hugging Face Inference Endpoints -- custom container (NOT a Space).
# Built automatically by .github/workflows/build-hf-endpoint-image.yml on every push to main
# -> ghcr.io/<owner>/cosyvoice3-endpoint:latest. Deploy: Inference Endpoints -> Custom
# Container -> that image URL, port 8080, health route /health. Set secret HF_TOKEN (read
# access to the all-lab/cosyvoice3-individual-* repos handler.py routes to).
#
# Mirrors the proven all-lab-tts-endpoint pattern (same org, ms0017/all-lab-tts-endpoint) --
# GHCR + GITHUB_TOKEN instead of Docker Hub (no separate registry credentials, no local
# Docker/network dependency). Port 8080 -- HF's custom-container platform injects its own
# PORT=8080 env var into the running container regardless of what the image declares, so the
# server (and the endpoint's healthRoute port config) must match that, not 80. This newer CUDA
# base reports zero CUDA/driver mismatch issues on the same HF T4 infra, unlike
# torch==2.3.1+cu121 here which hit "libcudart.so.13 not found" (host wants a newer CUDA
# runtime than that build provides).
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PORT=8080 \
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
    matplotlib==3.7.5 \
    gdown \
    wget \
    pyarrow==18.1.0

# Installing torch+torchaudio as a matched pair EARLIER in the build still ended up broken at
# runtime ("undefined symbol: torch::autograd::Node::name") because one of the packages above
# silently pulls in its own unpinned, ABI-mismatched torchaudio as a transitive dependency,
# clobbering the earlier install. Force-reinstall the matched pair LAST, after every other pip
# install, so nothing downstream can overwrite it -- then verify the import actually works at
# build time so a broken image fails CI immediately instead of failing on the live endpoint.
RUN pip install --no-cache-dir --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.8.0 torchaudio==2.8.0 && \
    python -c "import torch, torchaudio; print('torch', torch.__version__, 'torchaudio', torchaudio.__version__, 'cuda', torch.version.cuda)"

EXPOSE 8080
ENTRYPOINT ["python", "/repository/hf_endpoint_server.py"]
