FROM python:3.10-slim-bullseye

WORKDIR /workspace


RUN pip install --no-cache-dir \
    openmm \
    pdbfixer \
    scipy \
    numpy \
    pandas

COPY main.py . 
COPY run.sh .
RUN chmod +x run.sh

CMD ["bash", "run.sh"]