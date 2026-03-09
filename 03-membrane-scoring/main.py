# This script will score the predicted structures and generate DPO training triplets.

# Import necessary libraries
import os
import json
import argparse
import glob
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

# Custom function for calculating Eisenberg hydrophobic moment
# Define the hydrophobicities for the amino acids as per Eisenberg et al. 1984
HYDROPHOBICITY = {
    "ALA": 0.62, "ARG": -2.53, "ASN": -0.78, "ASP": -0.90, "CYS": 0.29,
    "GLN": -0.85, "GLU": -0.74, "GLY": 0.48, "HIS": -0.40, "ILE": 1.38,
    "LEU": 1.06, "LYS": -1.50, "MET": 0.64, "PHE": 1.19, "PRO": 0.12,
    "SER": -0.05, "THR": 0.05, "TRP": 0.81, "TYR": 0.26, "VAL": 1.08
}

# Defining the function to calculate the hydrophobic moment
def calculate_metrics(pdb_path, config):
    """
    Calculate the hydrophobic moment, net charge, and cyclic integrity of a peptide.
    
    Args:
        pdb_path (str): Path to the PDB file
        config (dict): Configuration dictionary

    Returns:
        dict: Dictionary containing the hydrophobic moment, net charge, and cyclic integrity of the peptide
    
    """
    
    # Parse the PDB file
    parser = PDBParser(QUIET=True)
    # Get the structure from the PDB file
    structure = parser.get_structure("peptide", pdb_path)

    # Initialize lists to store the coordinates, pLDDT scores, and residues
    coords, plddts, residues = [], [], []

    # Iterate over the models, chains, and residues in the structure
    for model in structure:
        for chain in model:
            for residue in chain:
                # Check if the residue has a CA atom
                if "CA" in residue:
                    # Get the CA atom
                    ca = residue["CA"]
                    # Append the coordinates, pLDDT scores, and residues to the lists
                    coords.append(ca.get_coord())
                    plddts.append(ca.get_bfactor())
                    residues.append(residue.get_resname())

    # If no CA (alpha carbon) atoms found return None
    if not coords: return None

    # Convert the coordinates to a numpy array and calculate the center of mass
    coords = np.array(coords)
    com = np.mean(coords, axis=0)
    # Center the coordinates around the center of mass
    centered_coords = coords - com

    # 3D Hydrophobic moment calculation (muH)
    h_vals = np.array([HYDROPHOBICITY.get(res, 0) for res in residues])

    # Calculate the norm of the centered coordinates
    norms = np.linalg.norm(centered_coords, axis=1)

    # Set any zero norms to 1.0 to avoid division by zero
    norms[norms == 0] = 1.0

    # Calculate the unit vectors
    unit_vectors = centered_coords / norms[:, np.newaxis]

    muH = np.linalg.norm(np.sum(unit_vectors * h_vals[:, np.newaxis], axis=0)) / len(residues)

    # Net charge
    charge = sum([1 if r in ["ARG", "LYS"] else -1 if r in ["ASP", "GLU"] else 0 for r in residues])
    net_charge = charge / len(residues)

    # Cyclic integrity
    dist_nc = np.linalg.norm(coords[0] - coords[-1])
    cyclic_penalty = 1.0 if dist_nc > config['cyclicity_penalty_threshold'] else 0.0

    # Composite membrane score
    # Score = (Charge * Anionic_Weight) + (muH * Weight) + Stability - Cyclic_Penalty
    score = (net_charge * config['anionic_fraction']) + \
            (muH * config['hydrophobic_moment_weight']) + \
            (np.mean(plddts)) - (cyclic_penalty * 2.0)

    return {
        "id": os.path.basename(pdb_path).replace('.pdb', ''), # ID of the peptide
        "net_charge": net_charge, # Net charge of the peptide
        "muH": muH, # Hydrophobic moment of the peptide
        "plddt": np.mean(plddts), # pLDDT score of the peptide
        "dist_nc": dist_nc, # Distance between the first and last residue
        "score": score # Composite membrane score
    }

def generate_dpo_pairs(df):
    """
    Generate DPO pairs from a dataframe of predicted structures.
    
    Args:
        df (pandas.DataFrame): Dataframe containing the predicted structures

    Returns:
        list: List of DPO pairs
    """

    # Initialize an empty list to store the DPO pairs
    dpo_pairs = []
    
    # Rank the peptides by "score" in descending order
    df = df.sort_values(by="score", ascending=False) # Sort the dataframe by "score" in descending order
    n= max(1, int(len(df) * 0.2)) # The number of winners is the top 20% of peptides
    winners = df.iloc[:n] # The top 20% of peptides are the winners
    losers = df.iloc[n:] # The remaining peptides are the losers
    
    if losers.empty:
        print("No losers found. Returning empty list.")
        return dpo_pairs
    
    for w_idx, winner in winners.iterrows():
        # Pair each winner with a random loser for contrast
        loser = losers.iloc[(losers['score'] - winner['score']).abs().argsort().iloc[0]]   # Sample a random loser from the losers dataframe
        dpo_pairs.append({"chosen": winner["id"], "rejected": loser["id"], "margin": winner["score"] - loser["score"]}) # Append the DPO pair to the list
    
    return dpo_pairs

def main():
    argparser = argparse.ArgumentParser() # Create an argument parser
    argparser.add_argument("--config", type=str, required=True) # Add the --config argument
    args = argparser.parse_args() # Parse the arguments
    config = json.loads(args.config) # Load the configuration

    # Path logic for silva
    pdb_files = glob.glob("prediction_results/*.pdb")
    if not pdb_files:
        pdb_files = glob.glob("*.pdb")
    if not pdb_files:
        print("WARNING: No PDB files found anywhere. Listing workspace:")
        for item in os.listdir("."):
            print(f"  {item}")

    results = [r for f in pdb_files if (r := calculate_metrics(f, config)) is not None]

    df = pd.DataFrame(results)
    df.to_csv("scoring_results.csv", index=False)

    if df.empty:
        print(f"WARNING: No structures were scored. Skipping DPO generation.")
        pairs=[]
    else:
        pairs = generate_dpo_pairs(df)
    
    with open('preferences.jsonl', 'w') as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    print(f"Node 03: Completed. Processed {len(df)} sequences and generated {len(pairs)} DPO pairs.")


if __name__ == "__main__":
    main()

# Free memory so Node 04 DPO has headroom
import gc
for _v in ['df', 'scored_df', 'structures', 'pdbs',
           'feature_matrix', 'dpo_pairs', 'sasa_results',
           'model', 'tokenizer']:
    if _v in globals():
        del globals()[_v]
gc.collect()
gc.collect()
import psutil
print(f"Node 03 EXIT memory: {psutil.virtual_memory().used/1e9:.1f}GB")

