# Node 02 — ESMFold local inference

FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# CPU-only PyTorch — sufficient for ESMFold inference on Mac
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    transformers \
    pandas \
    psutil

# Pre-download ESMFold model and tokenizer at build time
# Files: https://huggingface.co/facebook/esmfold_v1/tree/main
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
ENV HF_HOME=/tmp/hf_cache
ENV ESMFOLD_DIR=/tmp/hf_cache/facebook/esmfold_v1
RUN mkdir -p $ESMFOLD_DIR && \
    for f in config.json pytorch_model.bin tokenizer_config.json special_tokens_map.json vocab.txt; do \
        echo "Downloading $f ..." && \
        curl -fSL -o $ESMFOLD_DIR/$f https://huggingface.co/facebook/esmfold_v1/resolve/main/$f; \
    done

COPY predict_esmfold.py prepare_fasta.py run.sh pre_run.sh post_run.sh ./
RUN chmod +x run.sh pre_run.sh post_run.sh

# Keep container alive — Silva execs scripts via docker exec
ENTRYPOINT ["tail", "-f", "/dev/null"]