# Slim Python base (~150MB vs ~6GB for CUDA PyTorch)
FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LOGNAME=silva
ENV HOME=/tmp

# CPU-only PyTorch (~200MB vs ~2GB+ for CUDA) – fine for Mac / CPU runs
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Only packages used in main.py (pandas, datasets, transformers, trl, accelerate)
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    datasets \
    transformers \
    trl \
    accelerate

# Pre-download ProtGPT2 model and tokenizer at build time
# Files: https://huggingface.co/nferruz/ProtGPT2/tree/main
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
ENV PROTGPT2_DIR=/opt/hf_cache/nferruz/ProtGPT2
RUN mkdir -p $PROTGPT2_DIR && \
    for f in config.json pytorch_model.bin merges.txt special_tokens_map.json tokenizer.json vocab.json; do \
        echo "Downloading $f ..." && \
        curl -fSL -o $PROTGPT2_DIR/$f https://huggingface.co/nferruz/ProtGPT2/resolve/main/$f; \
    done

# Keep container alive — Silva mounts node scripts and execs via docker exec
ENTRYPOINT ["tail", "-f", "/dev/null"]