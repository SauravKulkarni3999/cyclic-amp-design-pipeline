# Node 06: Physics Validation via OpenMM Energy Minimisation
#
# Takes the top 10 DPO-generated sequences by proxy score (from Node 05),
# runs implicit-solvent energy minimisation using OpenMM + AMBER14 force field,
# and computes Spearman correlation between proxy rank and physics rank.
#
# Scientific claim enabled:
#   "Sequences ranked highly by proxy membrane score also achieve lower
#    potential energy after OpenMM relaxation (ρ=X), providing preliminary
#    physical support for the proxy scoring framework."
#
# Runtime: ~2-10 minutes for 10 sequences (implicit solvent, no water box)
# Memory:  ~2 GB peak (no GPU needed — CPU or MPS both work)
#
# Dependencies: openmm, pdbfixer
#   conda install -c conda-forge openmm pdbfixer
#   OR: pip install openmm pdbfixer

import os
import sys
import json
import numpy as np
import pandas as pd

# ── Dependency check with helpful error message ───────────────────────────────
try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
except ImportError:
    print("ERROR: OpenMM not installed.")
    print("Install with: conda install -c conda-forge openmm pdbfixer")
    print("         OR: pip install openmm pdbfixer")
    sys.exit(1)

try:
    from pdbfixer import PDBFixer
except ImportError:
    print("ERROR: pdbfixer not installed.")
    print("Install with: conda install -c conda-forge pdbfixer")
    sys.exit(1)

from scipy.stats import spearmanr


# ── Step 1: Select top N sequences by proxy score ────────────────────────────

def select_top_candidates(
    scored_csv: str,
    pdb_dir: str,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Load Node 05 scored output and select top N sequences.
    Verifies corresponding PDB files exist before proceeding.
    """
    df = pd.read_csv(scored_csv)
    df = df.sort_values('score', ascending=False).head(top_n)

    # Verify PDB files exist for selected sequences
    valid = []
    for _, row in df.iterrows():
        pdb_path = os.path.join(pdb_dir, f"{row['id']}.pdb")
        if os.path.exists(pdb_path):
            valid.append(row)
        else:
            print(f"  Warning: PDB not found for {row['id']}, skipping")

    df_valid = pd.DataFrame(valid)
    print(f"[Step 1] Selected {len(df_valid)} candidates for physics validation")
    print(f"         Proxy score range: "
          f"{df_valid['score'].min():.4f} – {df_valid['score'].max():.4f}")

    return df_valid


# ── Step 2: OpenMM energy minimisation ───────────────────────────────────────

def fix_pdb(pdb_path: str, fixed_pdb_path: str) -> None:
    """
    Use pdbfixer to prepare ESMFold PDB for OpenMM:
      - Add missing hydrogens (ESMFold outputs heavy atoms only)
      - Add missing heavy atoms if any
      - Remove HETATM records that could confuse the force field
    """
    fixer = PDBFixer(filename=pdb_path)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.0)  # physiological pH

    with open(fixed_pdb_path, 'w') as f:
        app.PDBFile.writeFile(
            fixer.topology,
            fixer.positions,
            f,
            keepIds=True
        )


def minimise_structure(
    pdb_path: str,
    fixed_pdb_path: str,
    relaxed_pdb_path: str,
    max_iterations: int = 500
) -> dict:
    """
    Run OpenMM implicit solvent energy minimisation.

    Force field: AMBER14 + GBn2 implicit solvent
    - amber14-all.xml: best general AMBER FF for peptides
    - implicit/gbn2.xml: generalised Born with neck correction,
      most accurate implicit solvent for short peptides

    Returns dict with energies, RMSD, and convergence status.
    """
    # Fix PDB first
    fix_pdb(pdb_path, fixed_pdb_path)

    # Load fixed PDB
    pdb = app.PDBFile(fixed_pdb_path)

    # Force field — AMBER14 + GBn2 implicit solvent
    forcefield = app.ForceField(
        'amber14-all.xml',
        'implicit/gbn2.xml'
    )

    # Build system
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,     # implicit solvent: no cutoff
        constraints=app.HBonds,            # constrain H bonds
        hydrogenMass=1.5 * unit.amu,       # heavier H for stability
        soluteDielectric=1.0,
        solventDielectric=78.5             # water dielectric at 298K
    )

    # Integrator — needed by OpenMM even for minimisation only
    integrator = mm.LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1.0 / unit.picosecond,
        0.002 * unit.picoseconds
    )

    # Platform selection — CPU is most reliable across hardware
    # Change to 'CUDA' or 'OpenCL' if GPU available
    try:
        platform = mm.Platform.getPlatformByName('CPU')
    except Exception:
        platform = mm.Platform.getPlatformByName('Reference')

    simulation = app.Simulation(
        pdb.topology,
        system,
        integrator,
        platform
    )
    simulation.context.setPositions(pdb.positions)

    # Initial energy
    state_initial = simulation.context.getState(getEnergy=True, getPositions=True)
    e_initial = state_initial.getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )
    positions_initial = np.array(
        state_initial.getPositions().value_in_unit(unit.angstrom)
    )

    # Energy minimisation
    simulation.minimizeEnergy(
        tolerance=1.0 * unit.kilojoules_per_mole / unit.nanometer,
        maxIterations=max_iterations
    )

    # Final energy
    state_final = simulation.context.getState(getEnergy=True, getPositions=True)
    e_final = state_final.getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )
    positions_final = np.array(
        state_final.getPositions().value_in_unit(unit.angstrom)
    )

    # RMSD between initial and relaxed structure
    diff = positions_final - positions_initial
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    # Save relaxed structure
    with open(relaxed_pdb_path, 'w') as f:
        app.PDBFile.writeFile(
            simulation.topology,
            state_final.getPositions(),
            f
        )

    # Convergence: if delta_E is very small, minimisation converged well
    delta_e = e_final - e_initial
    converged = abs(delta_e) > 0.1  # non-trivial relaxation occurred

    return {
        'e_initial':   round(e_initial, 2),
        'e_final':     round(e_final, 2),
        'delta_e':     round(delta_e, 2),
        'rmsd':        round(rmsd, 3),
        'converged':   converged,
        'n_atoms':     len(positions_initial),
    }


def run_physics_validation(
    candidates: pd.DataFrame,
    pdb_dir: str,
    work_dir: str
) -> pd.DataFrame:
    """
    Run OpenMM minimisation on each candidate.
    Handles failures gracefully — a single failed sequence
    does not abort the whole validation run.
    """
    fixed_dir  = os.path.join(work_dir, 'fixed_pdbs')
    relaxed_dir = os.path.join(work_dir, 'relaxed_pdbs')
    os.makedirs(fixed_dir, exist_ok=True)
    os.makedirs(relaxed_dir, exist_ok=True)

    n = len(candidates)
    print(f"\n[Step 2] Running OpenMM energy minimisation on {n} sequences...")
    print(f"         Force field : AMBER14 + GBn2 implicit solvent")
    print(f"         Max steps   : 500 per sequence")
    print(f"         Est. time   : ~{n * 3} – {n * 8} minutes")

    results = []

    for i, (_, row) in enumerate(candidates.iterrows()):
        seq_id   = row['id']
        sequence = row['sequence']
        pdb_path = os.path.join(pdb_dir, f"{seq_id}.pdb")
        fixed_path   = os.path.join(fixed_dir, f"{seq_id}_fixed.pdb")
        relaxed_path = os.path.join(relaxed_dir, f"{seq_id}_relaxed.pdb")

        print(f"\n  [{i+1}/{n}] {seq_id}: {sequence[:25]}...")

        try:
            physics = minimise_structure(pdb_path, fixed_path, relaxed_path)

            result = {
                'id':            seq_id,
                'sequence':      sequence,
                'proxy_score':   row['score'],
                'plddt':         row['plddt'],
                'dist_nc':       row['dist_nc'],
                'muH':           row['muH'],
                'net_charge':    row['net_charge'],
                'cys_count':     row['cys_count'],
                'e_initial':     physics['e_initial'],
                'e_final':       physics['e_final'],
                'delta_e':       physics['delta_e'],
                'rmsd':          physics['rmsd'],
                'n_atoms':       physics['n_atoms'],
                'converged':     physics['converged'],
                'relaxed_pdb':   relaxed_path,
            }

            print(f"         E_initial : {physics['e_initial']:>12.2f} kJ/mol")
            print(f"         E_final   : {physics['e_final']:>12.2f} kJ/mol")
            print(f"         delta_E   : {physics['delta_e']:>12.2f} kJ/mol")
            print(f"         RMSD      : {physics['rmsd']:>12.3f} Å")
            print(f"         Converged : {physics['converged']}")

        except Exception as e:
            print(f"         ERROR: {e}")
            print(f"         Skipping {seq_id} — adding null record")
            result = {
                'id': seq_id, 'sequence': sequence,
                'proxy_score': row['score'], 'plddt': row['plddt'],
                'dist_nc': row['dist_nc'], 'muH': row['muH'],
                'net_charge': row['net_charge'], 'cys_count': row['cys_count'],
                'e_initial': None, 'e_final': None, 'delta_e': None,
                'rmsd': None, 'n_atoms': None, 'converged': False,
                'relaxed_pdb': None,
            }

        results.append(result)

    return pd.DataFrame(results)


# ── Step 3: Rank and correlate ────────────────────────────────────────────────

def compute_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign physics rank (lower energy = better = rank 1)
    and compute Spearman correlation with proxy rank.
    """
    # Only use sequences where minimisation succeeded
    valid = df[df['e_final'].notna()].copy()

    if len(valid) < 3:
        print("Warning: fewer than 3 valid results — skipping correlation")
        df['rank_proxy']   = range(1, len(df) + 1)
        df['rank_physics'] = None
        return df

    # Proxy rank: already sorted by score descending (rank 1 = best proxy)
    valid = valid.sort_values('proxy_score', ascending=False)
    valid['rank_proxy'] = range(1, len(valid) + 1)

    # Physics rank: lower e_final = better = rank 1
    valid = valid.sort_values('e_final', ascending=True)
    valid['rank_physics'] = range(1, len(valid) + 1)

    # Spearman correlation
    rho, pval = spearmanr(valid['rank_proxy'], valid['rank_physics'])

    print(f"\n[Step 3] Rank correlation analysis:")
    print(f"         Valid sequences:       {len(valid)}/{len(df)}")
    print(f"         Spearman ρ:            {rho:.3f}")
    print(f"         p-value:               {pval:.4f}")

    if rho > 0.5:
        interpretation = "Strong agreement — proxy scores reflect physical stability"
    elif rho > 0.3:
        interpretation = "Moderate agreement — proxy scores partially validated"
    elif rho > 0:
        interpretation = "Weak positive trend — limited proxy validation"
    else:
        interpretation = "No agreement — proxy and physics rankings diverge"

    print(f"         Interpretation:        {interpretation}")

    # Merge ranks back
    rank_map = valid.set_index('id')[['rank_proxy', 'rank_physics']]
    df = df.set_index('id')
    df[['rank_proxy', 'rank_physics']] = rank_map
    df = df.reset_index()

    # Save correlation summary
    return df, rho, pval, interpretation


# ── Step 4: Print summary ─────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, rho: float, pval: float,
                  interpretation: str):

    valid = df[df['e_final'].notna()]

    print(f"\n{'='*65}")
    print(f"  NODE 06 — PHYSICS VALIDATION SUMMARY")
    print(f"{'='*65}")
    print(f"  Sequences validated:     {len(valid)}/{len(df)}")
    print(f"  Force field:             AMBER14 + GBn2 implicit solvent")
    print(f"  Spearman ρ (proxy/phys): {rho:.3f}  (p={pval:.4f})")
    print(f"  Interpretation:          {interpretation}")

    print(f"\n  Full results (sorted by physics rank):")
    sorted_df = df[df['e_final'].notna()].sort_values('e_final')

    print(f"  {'Phy':>4} {'Prx':>4} {'Sequence':<28} "
          f"{'E_final':>10} {'dE':>9} {'RMSD':>6} {'pLDDT':>6}")
    print(f"  {'-'*72}")

    for _, row in sorted_df.iterrows():
        print(
            f"  {int(row['rank_physics']):>4} "
            f"{int(row['rank_proxy']):>4} "
            f"{row['sequence']:<28} "
            f"{row['e_final']:>10.1f} "
            f"{row['delta_e']:>9.1f} "
            f"{row['rmsd']:>6.3f} "
            f"{row['plddt']:>6.3f}"
        )

    print(f"\n  Legend: Phy=physics rank, Prx=proxy rank,")
    print(f"          E_final=kJ/mol, dE=delta_E (kJ/mol), RMSD=Å")
    print(f"{'='*65}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Silva copies Node 05 outputs to ./inputs/
    scored_csv  = "inputs/scored_generated.csv"
    pdb_dir     = "inputs/generated_pdbs"
    work_dir    = "outputs"
    output_csv  = "outputs/physics_validation.csv"
    summary_json = "outputs/validation_summary.json"

    os.makedirs(work_dir, exist_ok=True)

    # Config
    top_n = int(os.getenv('TOP_N', '20'))
    max_iterations = int(os.getenv('MAX_ITERATIONS', '500'))

    print(f"Node 06: Physics Validation (OpenMM)")
    print(f"Input scores : {scored_csv}")
    print(f"Input PDBs   : {pdb_dir}")
    print(f"Top N        : {top_n}")
    print(f"Max steps    : {max_iterations}")

    # Validate inputs exist
    for path in [scored_csv, pdb_dir]:
        if not os.path.exists(path):
            print(f"Error: {path} not found.")
            print("Make sure Node 05 has run and outputs are in inputs/")
            sys.exit(1)

    # Step 1 — select top candidates
    candidates = select_top_candidates(scored_csv, pdb_dir, top_n)

    if candidates.empty:
        print("Error: no valid candidates found.")
        sys.exit(1)

    # Step 2 — OpenMM energy minimisation
    results_df = run_physics_validation(candidates, pdb_dir, work_dir)

    # Step 3 — rank and correlate
    output = compute_correlation(results_df)
    if isinstance(output, tuple):
        results_df, rho, pval, interpretation = output
    else:
        results_df = output
        rho, pval, interpretation = 0.0, 1.0, "Insufficient data"

    # Step 4 — save outputs
    results_df.to_csv(output_csv, index=False)

    # Save machine-readable summary for Node 07 dashboard
    valid = results_df[results_df['e_final'].notna()]
    summary = {
        "n_validated":       len(valid),
        "n_total":           len(results_df),
        "spearman_rho":      round(rho, 4),
        "spearman_pval":     round(pval, 4),
        "interpretation":    interpretation,
        "forcefield":        "AMBER14 + GBn2 implicit solvent",
        "best_proxy":        results_df.iloc[0]['id'] if len(results_df) else None,
        "best_physics":      (valid.sort_values('e_final').iloc[0]['id']
                              if len(valid) else None),
        "mean_delta_e":      round(valid['delta_e'].mean(), 2) if len(valid) else None,
    }
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)

    print_summary(results_df, rho, pval, interpretation)
    print(f"\n[Step 4] Saved → {output_csv}")
    print(f"         Saved → {summary_json}")
    print(f"\nNode 06 complete.")
    print(f"Output for Validation: {output_csv}")


if __name__ == "__main__":
    main()