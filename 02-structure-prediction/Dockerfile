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

COPY predict_esmfold.py prepare_fasta.py run.sh pre_run.sh post_run.sh ./
RUN chmod +x run.sh pre_run.sh post_run.sh

# Keep container alive — Silva execs scripts via docker exec
ENTRYPOINT ["tail", "-f", "/dev/null"]