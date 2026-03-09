# Node 05: Structure Prediction + Scoring for DPO-Generated Sequences
#
# Loads ESMFold from HuggingFace cache (populated by Node 02 — no re-download),
# predicts structures for the 50 generated sequences, then scores them using
# the exact same calculate_metrics() function and config as Node 03.
#
# Output: scored_generated.csv
# Columns: id, sequence, net_charge, muH, plddt, dist_nc, score,
#          cyclic_candidate, cys_count, rank
#
# Scores are directly comparable to Node 03's scoring_results.csv
# because calculate_metrics() is copied verbatim — same formula, same weights.

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import torch
from transformers import EsmForProteinFolding, AutoTokenizer
from Bio.PDB import PDBParser


# ── Scoring logic — verbatim from Node 03 main.py ────────────────────────────
# No changes. Same HYDROPHOBICITY dict, same calculate_metrics() function,
# same config keys. Scores will be on the same scale as Node 03 output.

HYDROPHOBICITY = {
    "ALA": 0.62, "ARG": -2.53, "ASN": -0.78, "ASP": -0.90, "CYS": 0.29,
    "GLN": -0.85, "GLU": -0.74, "GLY": 0.48, "HIS": -0.40, "ILE": 1.38,
    "LEU": 1.06, "LYS": -1.50, "MET": 0.64, "PHE": 1.19, "PRO": 0.12,
    "SER": -0.05, "THR": 0.05, "TRP": 0.81, "TYR": 0.26, "VAL": 1.08
}


def calculate_metrics(pdb_path, config):
    """
    Verbatim copy from Node 03 main.py.
    Calculate hydrophobic moment, net charge, and cyclic integrity.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("peptide", pdb_path)

    coords, plddts, residues = [], [], []

    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    ca = residue["CA"]
                    coords.append(ca.get_coord())
                    plddts.append(ca.get_bfactor())
                    residues.append(residue.get_resname())

    if not coords:
        return None

    coords = np.array(coords)
    com = np.mean(coords, axis=0)
    centered_coords = coords - com

    h_vals = np.array([HYDROPHOBICITY.get(res, 0) for res in residues])
    norms = np.linalg.norm(centered_coords, axis=1)
    norms[norms == 0] = 1.0
    unit_vectors = centered_coords / norms[:, np.newaxis]
    muH = np.linalg.norm(
        np.sum(unit_vectors * h_vals[:, np.newaxis], axis=0)
    ) / len(residues)

    charge = sum([
        1 if r in ["ARG", "LYS"]
        else -1 if r in ["ASP", "GLU"]
        else 0
        for r in residues
    ])
    net_charge = charge / len(residues)

    dist_nc = np.linalg.norm(coords[0] - coords[-1])
    cyclic_penalty = (
        1.0 if dist_nc > config['cyclicity_penalty_threshold'] else 0.0
    )

    score = (
        (net_charge * config['anionic_fraction'])
        + (muH * config['hydrophobic_moment_weight'])
        + np.mean(plddts)
        - (cyclic_penalty * 2.0)
    )

    return {
        "id":         os.path.basename(pdb_path).replace('.pdb', ''),
        "net_charge": net_charge,
        "muH":        muH,
        "plddt":      np.mean(plddts),
        "dist_nc":    dist_nc,
        "score":      score,
    }


# ── Step 1: Prepare FASTA files ───────────────────────────────────────────────

def prepare_fastas(sequences_csv: str, fasta_dir: str) -> list:
    """
    Convert generated sequences CSV to FASTA files.
    Identical format to Node 02's prepare_fasta.py:
        header  : >A|protein
        filenames: seq_0.fasta, seq_1.fasta, ...
    """
    if os.path.exists(fasta_dir):
        for f in os.listdir(fasta_dir):
            if f.endswith('.fasta'):
                os.remove(os.path.join(fasta_dir, f))
    os.makedirs(fasta_dir, exist_ok=True)

    df = pd.read_csv(sequences_csv)
    sequences = df['sequence'].tolist()

    for idx, seq in enumerate(sequences):
        fasta_path = os.path.join(fasta_dir, f"seq_{idx}.fasta")
        with open(fasta_path, 'w') as f:
            f.write(f">A|protein\n{str(seq).strip().upper()}\n")

    print(f"[Step 1] Prepared {len(sequences)} FASTA files → {fasta_dir}")
    return sequences


# ── Step 2: ESMFold structure prediction ──────────────────────────────────────

def predict_structures(fasta_dir: str, pdb_dir: str, device: str) -> None:
    """
    Load ESMFold from HuggingFace cache and predict structures.
    Cache was populated by Node 02 — no re-download required.
    Model is deleted after prediction to free memory before scoring.
    """
    os.makedirs(pdb_dir, exist_ok=True)

    fasta_files = sorted(glob.glob(os.path.join(fasta_dir, "*.fasta")))
    if not fasta_files:
        print(f"Error: No FASTA files found in {fasta_dir}")
        sys.exit(1)

    n = len(fasta_files)
    print(f"\n[Step 2] Loading ESMFold from HuggingFace cache...")
    print(f"         Sequences : {n}")
    print(f"         Device    : {device}")
    print(f"         Est. time : ~{n * 8 // 60}m {n * 8 % 60}s")

    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    model = model.to(device)
    model.eval()

    for idx, fasta_file in enumerate(fasta_files):
        with open(fasta_file, 'r') as f:
            lines = f.readlines()
            sequence = lines[1].strip()

        name = f"seq_{idx}"

        inputs = tokenizer(
            sequence,
            return_tensors="pt",
            add_special_tokens=False
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        pdb_string = model.output_to_pdb(outputs)[0]
        out_path = os.path.join(pdb_dir, f"{name}.pdb")
        with open(out_path, 'w') as f:
            f.write(pdb_string)

        plddt = outputs.plddt[0].mean().item()
        print(f"  [{idx+1}/{n}] {name}: pLDDT={plddt:.3f} → {out_path}")

    # Free ESMFold from memory before scoring phase
    del model, tokenizer
    if device == 'mps':
        torch.mps.empty_cache()
    elif device == 'cuda':
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    pdb_count = len(glob.glob(os.path.join(pdb_dir, "*.pdb")))
    print(f"[Step 2] Complete. {pdb_count} PDBs saved. "
          f"ESMFold freed from memory.")


# ── Step 3: Score all structures ──────────────────────────────────────────────

def score_structures(
    sequences: list,
    pdb_dir: str,
    config: dict
) -> pd.DataFrame:
    """
    Score each PDB with calculate_metrics() — verbatim from Node 03.
    Adds sequence, cyclic_candidate, cys_count, rank for Node 06 viz.
    """
    pdb_files = sorted(glob.glob(os.path.join(pdb_dir, "*.pdb")))
    print(f"\n[Step 3] Scoring {len(pdb_files)} structures...")

    results = []
    for pdb_path in pdb_files:
        metrics = calculate_metrics(pdb_path, config)
        if metrics is None:
            print(f"  Warning: no CA atoms in {pdb_path}, skipping")
            continue
        results.append(metrics)

    if not results:
        print("Error: no structures could be scored.")
        sys.exit(1)

    df = pd.DataFrame(results)

    # Map sequence back by index — seq_0 → sequences[0]
    def get_sequence(seq_id):
        try:
            idx = int(seq_id.replace('seq_', ''))
            return sequences[idx] if idx < len(sequences) else ''
        except (ValueError, IndexError):
            return ''

    df['sequence']         = df['id'].apply(get_sequence)
    df['cyclic_candidate'] = df['dist_nc'] < config['cyclicity_penalty_threshold']
    df['cys_count']        = df['sequence'].str.count('C')

    # Sort by composite score, assign rank
    df = df.sort_values('score', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)

    # Column order — Node 03 core columns first, then viz extras
    ordered = [
        'id', 'sequence', 'net_charge', 'muH', 'plddt',
        'dist_nc', 'score', 'cyclic_candidate', 'cys_count', 'rank'
    ]
    df = df[[c for c in ordered if c in df.columns]]

    print(f"[Step 3] Scoring complete.")
    return df


# ── Step 4: Print summary ─────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, config: dict):
    threshold = config['cyclicity_penalty_threshold']
    cyclic    = df['cyclic_candidate'].sum()
    multi_cys = (df['cys_count'] >= 2).sum()

    print(f"\n{'='*62}")
    print(f"  NODE 05 — SCORED GENERATED SEQUENCES")
    print(f"{'='*62}")
    print(f"  Sequences scored    : {len(df)}")
    print(f"  Cyclic candidates   : {cyclic}  (dist_nc < {threshold}Å)")
    print(f"  Multi-Cys (≥2)      : {multi_cys}  (disulfide bridge potential)")
    print(f"  Mean pLDDT          : {df['plddt'].mean():.3f}")
    print(f"  Mean score          : {df['score'].mean():.4f}")
    print(f"  Score range         : "
          f"{df['score'].min():.4f} – {df['score'].max():.4f}")

    print(f"\n  Top 10 candidates:")
    header = (f"  {'Rank':<5} {'Sequence':<30} {'Score':>8} "
              f"{'pLDDT':>7} {'dist_nc':>8} {'Cyclic':>7} {'Cys':>4}")
    print(header)
    print(f"  {'-'*68}")
    for _, row in df.head(10).iterrows():
        flag = '✓' if row['cyclic_candidate'] else ''
        print(
            f"  {int(row['rank']):<5} "
            f"{row['sequence']:<30} "
            f"{row['score']:>8.4f} "
            f"{row['plddt']:>7.3f} "
            f"{row['dist_nc']:>8.2f} "
            f"{flag:>7} "
            f"{int(row['cys_count']):>4}"
        )
    print(f"{'='*62}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Silva copies Node 04 output to ./inputs/
    generated_csv = "inputs/optimized_sequences.csv"
    fasta_dir     = "fasta_inputs"
    pdb_dir       = "outputs/generated_pdbs/*.pdb"
    scored_csv    = "outputs/scored_generated.csv"

    os.makedirs("outputs", exist_ok=True)

    # Config — matches Node 03 job.toml [params.scoring_config] exactly
    # cyclicity_penalty_threshold : 4.5Å  (not 8.0 — stricter cyclic filter)
    # anionic_fraction            : 0.25  (not 1.0 — charge weighted at 25%)
    # hydrophobic_moment_weight   : 5.0   (not 1.0 — muH is dominant signal)
    default_config = {
        "target_organism":              "E_coli",
        "anionic_fraction":             0.25,
        "hydrophobic_moment_weight":    5.0,
        "cyclicity_penalty_threshold":  4.5,
    }

    config_json = os.getenv('SCORING_CONFIG', json.dumps(default_config))
    try:
        config = json.loads(config_json)
    except json.JSONDecodeError:
        print("Warning: Could not parse SCORING_CONFIG, using defaults")
        config = default_config

    # Device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Node 05: Structure Prediction + Scoring (Generated Sequences)")
    print(f"Device : {device}")
    print(f"Config : {config}")
    print(f"Input  : {generated_csv}")

    if not os.path.exists(generated_csv):
        print(f"Error: {generated_csv} not found.")
        sys.exit(1)

    # Step 1 — FASTA prep
    sequences = prepare_fastas(generated_csv, fasta_dir)

    # Step 2 — structure prediction (ESMFold from cache)
    predict_structures(fasta_dir, pdb_dir, device)

    # Step 3 — score using Node 03's exact calculate_metrics()
    df = score_structures(sequences, pdb_dir, config)

    # Step 4 — save and summarise
    df.to_csv(scored_csv, index=False)
    print(f"\n[Step 4] Saved → {scored_csv}")
    print_summary(df, config)

    print(f"\nNode 05 complete.")
    print(f"Scored output for Node 06: {scored_csv}")


if __name__ == "__main__":
    main()