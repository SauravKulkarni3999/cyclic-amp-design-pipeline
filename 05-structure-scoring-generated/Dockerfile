FROM esmfold:latest

WORKDIR /workspace

# Copy Node 05 scripts
COPY main.py .
COPY run.sh .

RUN chmod +x run.sh

# biopython for PDB parsing and SASA calculation
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    transformers \
    pandas \
    psutil \
    biopython
    
CMD ["bash", "run.sh"]

