# predict_esmfold.py — Local ESMFold inference
#
# Downloads and runs facebook/esmfold_v1 locally.
# Saves PDB files to outputs/prediction_results/seq_N.pdb

import glob
import os
import sys
import torch
from transformers import EsmForProteinFolding, AutoTokenizer

# ── Configuration ─────────────────────────────────────────────
FASTA_DIR = "fasta_inputs"
OUTPUT_DIR = "outputs/prediction_results"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading ESMFold model and tokenizer...")
    model_path = os.environ.get("ESMFOLD_DIR", "facebook/esmfold_v1")
    local = os.path.isdir(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local)
    model = EsmForProteinFolding.from_pretrained(model_path, local_files_only=local)

    # ESMFold FP32 requires ~13GB VRAM (8GB model + overhead during transfer); fall back to CPU if insufficient
    MIN_GPU_MEM_GB = 13
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        free_mem_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / (1024 ** 3)
        print(f"GPU: {torch.cuda.get_device_name(0)}, total: {gpu_mem_gb:.1f}GB, free: {free_mem_gb:.1f}GB")
        if free_mem_gb >= MIN_GPU_MEM_GB:
            device = torch.device("cuda")
        else:
            print(f"GPU memory ({free_mem_gb:.1f}GB free) < {MIN_GPU_MEM_GB}GB required, falling back to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model.to(device)
    model.eval()

    fasta_files = sorted(glob.glob(os.path.join(FASTA_DIR, "*.fasta")))
    if not fasta_files:
        print(f"Error: No FASTA files found in {FASTA_DIR}")
        sys.exit(1)

    print(f"Predicting structures for {len(fasta_files)} sequences...")

    for idx, fasta_file in enumerate(fasta_files):
        with open(fasta_file, "r") as f:
            lines = f.readlines()
        sequence = lines[1].strip() if len(lines) >= 2 else ""

        if not sequence:
            print(f"[{idx+1}/{len(fasta_files)}] WARNING: empty sequence in {fasta_file}, skipping")
            continue

        name = f"seq_{idx}"
        out_path = os.path.join(OUTPUT_DIR, f"{name}.pdb")

        # Skip already-predicted sequences — allows resuming interrupted runs
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"[{idx+1}/{len(fasta_files)}] {name}: already exists, skipping")
            continue

        print(f"[{idx+1}/{len(fasta_files)}] {name}: predicting ({len(sequence)} residues)...")

        inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        pdb_string = model.output_to_pdb(outputs)[0]

        with open(out_path, "w") as f:
            f.write(pdb_string)

        # pLDDT from model output tensor — correct source, not B-factor parsing
        plddt = outputs.plddt[0].mean().item()
        if (idx+1) % 100 == 0: 
            print(f"[{idx+1}/{len(fasta_files)}] {name}: pLDDT = {plddt:.3f}, saved to {out_path}")

    print("Prediction complete.")


if __name__ == "__main__":
    main()