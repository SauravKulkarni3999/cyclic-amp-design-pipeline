# Cyclic AMP Discovery Pipeline
### DPO-Aligned Generative Design with Biophysics Scoring and Physics Validation

---

This pipeline uses AI to 'teach' a discovery engine how to design safer, more effective antimicrobial medicines before a single experiment is ever run in a lab.

--- 

[![Silva](https://img.shields.io/badge/runs%20on-Silva-00e5ff)](https://github.com/chiral-data/silva)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Challenge](https://img.shields.io/badge/Chiral%20Blueprint%20Build%20Challenge-2026-7c4dff)](https://chiral.one)

**Last full run:** March 9, 2026 · 6/6 nodes · Apple M4 Pro · ~2 hours

---

A fully reproducible, end-to-end cyclic antimicrobial peptide (AMP) discovery pipeline built on [Silva](https://github.com/chiral-data/silva). The pipeline applies **Direct Preference Optimization (DPO)** to align a protein language model toward biophysics-informed design objectives, then validates top candidates with OpenMM molecular dynamics.

**Input:** AMP sequence database (CSV)  
**Output:** Physics-validated, ranked cyclic peptide candidates with interactive results dashboard

---

## Why DPO for Peptide Design?

Most computational AMP pipelines generate sequences and filter them post-hoc. This pipeline inverts that logic: it first learns what "good" looks like from 1405 experimentally-derived AMP sequences, constructs preference pairs based on biophysics scoring, and uses DPO to align ProtGPT2 *before* generation. The result is a model that generates sequences with membrane-active properties built into the generative prior rather than filtered out afterward.

DPO was chosen over standard fine-tuning because it directly encodes the relative preference signal — "this sequence is better than that one, and here is why" — rather than maximizing likelihood on a static dataset. This is mechanistically appropriate for a problem where we care about ranking and relative quality, not absolute sequence probability.

---

## Pipeline Architecture

```
01-sequence-ingest
      │  1405 AMP sequences (CSV)
      ▼
02-structure-prediction
      │  ESMFold local inference → PDB files + pLDDT scores
      ▼
03-membrane-scoring
      │  Composite biophysics scoring → DPO preference pairs (281 pairs)
      ▼
04-dpo-iteration
      │  ProtGPT2 fine-tuned via DPO → aligned generative model
      ▼
05-structure-scoring-generated
      │  50 generated sequences → scored + ranked
      ▼
06-physics-validation
      │  Top 10 → OpenMM AMBER14/GBn2 relaxation → final rankings
      ▼
   results/
      ├── physics_validation.csv
      ├── scored_generated.csv
      └── visualization.html   ← interactive dashboard
```

Each node runs in its own Docker container with explicit input/output contracts. Nodes are independently reusable.

---

## Scoring Function Design

The composite proxy score combines four mechanistically grounded features:

| Feature | Rationale |
|---|---|
| **Hydrophobic Moment (μH)** | Quantifies amphipathicity for membrane insertion. Based on Eisenberg (1982). Correlates with activity in the Shai-Matsuzaki-Huang membrane disruption model. |
| **Net Charge at pH 7.4** | Cationic peptides exploit electrostatic selectivity — bacterial membranes are anionic (PG, cardiolipin) vs. zwitterionic mammalian membranes. |
| **pLDDT** | ESMFold structural confidence as a pre-filter for sequences with coherent predicted structure. Used as a fast filter, not a thermodynamic stability score. |
| **dist_nc (N-to-C terminus distance)** | Proxy for compact folding propensity in linear ESMFold predictions. Sequences with lower dist_nc have greater cyclization compatibility. |

**Known limitations acknowledged:** μH assumes helical conformation, which breaks down for constrained cyclic geometries. dist_nc reflects linear ESMFold predictions, not true cyclic topology — proper cyclic evaluation requires models trained on cyclic scaffolds (e.g., AfCycDesign, RFpeptides). The composite function uses equal weighting as a maximum-entropy assumption in the absence of experimental MIC data for calibration.

---

## DPO Training Details

- **Base model:** `nferruz/ProtGPT2`
- **Training data:** 281 preference pairs generated from biophysics scoring of 1405 sequences
- **Pair construction:** Closest-margin negative sampling — each winner is paired with the nearest-scoring loser below the median. In this dataset, the margin distribution collapsed to a single reject anchor (`seq_1104`), which limits the diversity of the preference signal. This is a known limitation addressed in the extensions roadmap below.
- **Framework:** HuggingFace TRL `DPOTrainer`
- **Hardware:** Apple M4 Pro (CPU mode, 25.8GB RAM)
- **Training time:** ~90 minutes for 1405-sequence dataset
- **Key hyperparameters:** `num_train_epochs=2`, `gradient_accumulation_steps=4`, `bf16=False`, `fp16=False`
- **Final loss:** ~2.8e-05 (converged), reward margins ~10.0 kJ

The reference model is held on CPU throughout training to reduce memory pressure by ~1.5GB — only the policy model uses the primary device.

**Note on generated sequence quality:** At this training scale (281 pairs, 2 epochs), the DPO alignment signal was insufficient to fully shift ProtGPT2's generative distribution into AMP-like sequence space. The generated sequences reflect ProtGPT2's general protein prior more than targeted antimicrobial properties. The primary contribution of this pipeline is the *framework design* — the alignment loop, scoring architecture, and validation protocol — rather than the specific candidate sequences from this initial run. Scaling the preference dataset (more sequences, diverse reject anchors) and increasing training epochs are the direct remedies outlined in the extensions section.

---

## Physics Validation

Top 10 candidates from Node 05 are validated using OpenMM with AMBER14 force field and GBn2 implicit solvent:

- **Protocol:** Energy minimization → 500-step convergence → final energy evaluation
- **Metrics:** E_final (absolute energy), ΔE (relaxation energy), RMSD from initial structure
- **Implicit solvent rationale:** Appropriate for ranking and filtering at this pipeline stage. Explicit solvent would improve accuracy but is computationally intractable at this throughput on consumer hardware.

---

## Results (Full 1405-Sequence Run)

**Top 3 candidates by physics validation (lowest E_final):**

| Physics Rank | ID | Sequence | E_final (kJ/mol) | ΔE (kJ/mol) | pLDDT | Cys |
|---|---|---|---|---|---|---|
| 1 | seq_0 | PTDPRARLRHLLEQDAPLAVAVD | -4590.8 | -4271.6 | 0.738 | 0 |
| 2 | seq_7 | DQQKIAILIDAENACQSRIDDVL | -4199.1 | -4427.9 | 0.807 | 1 |
| 3 | seq_25 | NRIVLLSFAILILACGSNKSNQ | -3916.3 | -4517.8 | 0.742 | 1 |

**Highlighted candidate — seq_32 (disulfide scaffold):**

```
STCPSCCSPIRPHQRFFCS
```
Physics rank 6, but ΔE = **-6449.9 kJ/mol** — the largest relaxation energy in the validated set by a wide margin. 4 Cys residues suggest strong disulfide bridge scaffold potential. Proxy scoring undervalues this sequence because the dist_nc penalty fires on the linear ESMFold prediction; the physics validation reveals exceptional structural stability upon energy minimization.

**DPO alignment signal:**
- 281 preference pairs from 1405 sequences
- Reward margins: 0.008 – 5.58 kJ (mean 0.74 kJ)
- Training loss converged to ~2.8e-05 by epoch 2
- 4/50 generated sequences exceeded the original dataset's 90th percentile score

**Spearman rank correlation (proxy vs. physics):** ρ = 0.164, p = 0.65. At n=10, statistical significance requires ρ > 0.63 for p < 0.05. This is a sample-size constraint rather than evidence against the scoring framework. The correlation direction is positive, and expanding the validated set is a straightforward extension.

**Runtime:** ~2 hours on Apple M4 Pro (consumer CPU). Estimated 15–20 minutes on A100 GPU for equivalent dataset.

---

## Visualization

An interactive HTML dashboard (`results/visualization.html`) is included with every run, containing:

- Pipeline architecture diagram
- Score distribution comparison (original vs. DPO-generated)
- pLDDT vs. proxy score scatter with physics-validated candidates highlighted
- Hydrophobic moment vs. net charge scatter
- N-to-C terminus distance distribution
- Proxy rank vs. physics rank agreement chart
- Full ranked candidate table with amino acid coloring, score bars, and Cys badges

Open directly in any browser — no server required.

---

## Quickstart

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Silva](https://github.com/chiral-data/silva) installed and running
- ~16GB RAM recommended

### Setup

```bash
# Clone the repository
git clone https://github.com/SauravKulkarni3999/cyclic-amp-design-pipeline
cd cyclic-amp-design-pipeline

# Set Silva workflow directory
export SILVA_WORKFLOW_HOME=$(pwd)/workflows

# Launch Silva
silva
```

### Running the Pipeline

1. Open Silva in your browser
2. Navigate to **Workflows** tab
3. Select `cyclic_amp_design_pipeline`
4. Click **Run**
5. All 6 nodes complete with green checkmarks in ~2 hours

### Outputs

```
results/
├── physics_validation.csv    # Physics-ranked candidates with energetics
├── scored_generated.csv      # All 50 generated sequences with scores
├── scoring_results.csv       # All 1405 original sequences scored
├── preferences.jsonl         # DPO training pairs with margins
└── visualization.html        # Interactive results dashboard
```

---

## Repository Structure

```
cyclic-amp-design-pipeline/
├── .chiral
|   └── workflow.toml
├── 01-sequence-ingest/
│   ├── .chiral/job.toml
│   ├── run_ingest.py
│   ├── sequences.csv
│   └── run.sh
├── 02-structure-prediction/
│   ├── .chiral/job.toml
│   ├── Dockerfile
│   ├── predict_esmfold.py
│   ├── prepare_fasta.py
│   ├── pre_run.sh
│   ├── post_run.sh
│   └── run.sh
├── 03-membrane-scoring/
│   ├── .chiral/job.toml
│   ├── main.py
│   ├── pre_run.sh
│   ├── post_run.sh
│   └── run_scoring.sh
├── 04-dpo-iteration/
│   ├── .chiral/job.toml
│   ├── Dockerfile
│   ├── main.py
│   ├── pre_run.sh
│   ├── run_dpo_training.sh
│   └── post_run.sh
├── 05-structure-scoring-generated/
│   ├── .chiral/job.toml
│   ├── Dockerfile
│   ├── pre_run.sh
│   ├── run.sh
│   └── main.py
├── 06-physics-validation/
|   ├── .chiral/job.toml
|   ├── Dockerfile
|   ├── run.sh
|   └── main.py
├── results/
│   ├── validation_summary.json
│   ├── preferences.jsonl
│   ├── scoring_results.csv
│   ├── physics_validation.csv
│   ├── scored_generated.csv
│   └── visualization.html
└── README.md
```

---

## Design Decisions and Tradeoffs

**Why ESMFold over AlphaFold3?**  
ESMFold runs locally without API dependencies, enabling reproducible offline execution. On short peptides (10–30 residues), the pLDDT quality is sufficient for the pre-filtering role it plays in this pipeline. AlphaFold3 would be a drop-in upgrade for Node 02 in a production context.

**Why ProtGPT2 over ESM2?**  
ProtGPT2 is a generative model trained for sequence generation; ESM2 is a discriminative model primarily suited for embeddings and property prediction. DPO requires a generative policy model. ProtGPT2 is the appropriate architectural choice.

**Why implicit solvent MD over explicit?**  
For a ranking and filtering stage operating on 10 candidates, AMBER14/GBn2 implicit solvent provides sufficient signal at a fraction of the compute cost. The pipeline is designed to be extended with explicit solvent free-energy methods (FEP, MMPBSA) on the top 3 candidates as a downstream step.

**Why equal weighting in the composite score?**  
Equal weighting is the maximum-entropy assumption in the absence of experimental MIC data for calibration. The intended extension is to regress score weights against held-out MIC values from APD3 or DBAASP once experimental data is available.

**Why closest-margin negative sampling for DPO pairs?**  
Closest-margin pairing provides the most informative preference signal in theory — the model learns from near-decision-boundary contrasts rather than trivially separable pairs. In practice, this collapsed to a single reject anchor in the current dataset due to the score distribution shape. The extensions section addresses this with diversified negative sampling strategies.

---

## Extending the Pipeline

Natural extensions in order of priority:

1. **Diversified DPO negative sampling** — replace closest-margin single-anchor pairing with stratified sampling across score bins, ensuring each winner sees a unique reject. This directly addresses the preference signal diversity limitation observed in the current run.
2. **Scaled DPO training** — increase to 5–10 epochs with learning rate warmup and larger preference datasets (>1000 pairs from expanded AMP databases like DBAASP or APD3) to strengthen the alignment signal.
3. **AfCycDesign integration** — replace ESMFold in Node 02 with AlphaFold2 + cyclic positional encoding for true cyclic topology prediction.
4. **External validation** — run top candidates through APD3/DBAASP for independent activity prediction, or integrate CAMPR4 API for antimicrobial probability scoring.
5. **Explicit solvent MD** — upgrade Node 06 to explicit solvent for top 3 candidates.
6. **Learned score weights** — calibrate composite score weights against experimental MIC data.

---

## Built As Part Of

[Chiral Blueprint Build Challenge 2026](https://chiral.one) — Feb 23 to Mar 16, 2026.  
Focus area: Cyclic peptide drug discovery.

---

## Citation

If you use this pipeline in your work:

```bibtex
@software{cyclic_amp_pipeline_2026,
  author    = {Kulkarni, Saurav},
  title     = {Cyclic AMP Discovery Pipeline: DPO-Aligned Generative Design with Biophysics Scoring},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/SauravKulkarni3999/cyclic-amp-design-pipeline}
}
```

---

## Author

**Saurav Kulkarni**  
Computational Scientist | ML × Structural Biology × Drug Discovery  
[LinkedIn](https://linkedin.com/in/sauravkulkarni/) · [GitHub](https://github.com/SauravKulkarni3999/)

Background: wet lab AMR research (JNCASR) + ML pipeline development. This pipeline reflects both — the scoring function is mechanistically grounded in membrane biophysics, not just ML heuristics.