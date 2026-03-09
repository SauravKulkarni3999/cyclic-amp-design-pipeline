# Slim Python base (~150MB vs ~6GB for CUDA PyTorch)
FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

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

COPY main.py pre_run.sh post_run.sh run_dpo_training.sh .
RUN chmod +x pre_run.sh post_run.sh run_dpo_training.sh

ENTRYPOINT ["tail", "-f", "/dev/null"]