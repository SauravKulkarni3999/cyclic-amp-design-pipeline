# Node 02 — ESMFold local inference

FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# PyTorch with CUDA 12.4 support — falls back to CPU when no GPU is available
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CUDA_VERSION=12.4
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir \
    transformers \
    pandas \
    psutil

# Pre-download ESMFold model and tokenizer at build time
# Files: https://huggingface.co/facebook/esmfold_v1/tree/main
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
ENV LOGNAME=silva
ENV HOME=/tmp
ENV HF_HOME=/opt/hf_cache
ENV ESMFOLD_DIR=/opt/hf_cache/facebook/esmfold_v1
RUN mkdir -p $ESMFOLD_DIR && \
    for f in config.json pytorch_model.bin tokenizer_config.json special_tokens_map.json vocab.txt; do \
        echo "Downloading $f ..." && \
        curl -fSL -o $ESMFOLD_DIR/$f https://huggingface.co/facebook/esmfold_v1/resolve/main/$f; \
    done

# Keep container alive — Silva mounts node scripts and execs via docker exec
ENTRYPOINT ["tail", "-f", "/dev/null"]