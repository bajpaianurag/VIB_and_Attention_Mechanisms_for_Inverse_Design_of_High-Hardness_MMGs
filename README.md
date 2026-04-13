# VIB + Attention for Inverse Design of High‑Hardness Metallic Multi‑Component Glasses

This repository contains the code and data used for the article **Attention-Enhanced Variational Learning for Physically Informed Design of Exceptionally hard Multicomponent Metallic Glasses** for designing exceptionally hard multi-component bulk metallic glasses.

> **Reproducibility note**: The main ML scripts use a fixed random seed (see each script). For paper‑level statistics (mean ± std over multiple seeds), run the same pipeline over 10 seeds and aggregate the exported metrics.

---

## Repository structure

```
.
├── data/
│   └── H_v_dataset.csv                 # Composition + load + HV target
├── VIBANN/
│   └── vib_attention_nns.py            # Main VIB + Attention neural network pipeline
├── Final Model and Weights/            # Final VIBANN model with different cross-validation folds and weights
├── Baselines/
│   ├── other_regression_models.py      # Conventional regressors (RF/GB/Lasso/Ridge/KNN/MLP)
│   └── gp_bo_baseline.py               # Baseline GP surrogate + BO inverse design workflow
├── MD codes/
│   ├── script.py                       # LAMMPS input generator from comp.csv
│   ├── comp.csv                        # Final developed compositions for MD
│   ├── post-processing/                # RDF, Voronoi, coordination, bond metrics
│   ├── Final MD alloy configurations/  # Final atomic configurations of simulated alloys
│   └── REAMD.md                        # MD module documentation
├── requirements*.txt                   # Installation options
└── environment.yml                     # Conda environment (recommended for reviewers)
```

---

## Quick start

### 1) Create an environment

**Recommended (Conda)**

```bash
conda env create -f environment.yml
conda activate vib-attention-mmg
```

**Alternative (pip)**

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

---

## A. Main model: VIB + Attention

### Run

```bash
python VIBANN/vib_attention_nns.py
```

### Inputs

- `data/H_v_dataset.csv`
  - Composition columns: elemental fractions (the script auto-detects all columns except `Load` and `HV`)
  - `Load`: indentation load
  - `HV`: Vickers hardness target

### Outputs

The script generates diagnostic plots and model artifacts in the working directory. 


---

## B. Baseline regressors (“other regressors”)

This script provides conventional ML baselines for hardness prediction. It supports either:
- **Default training** (fast) using `--no-bayes`, or
- **Bayesian hyperparameter optimization** (slower; requires `scikit-optimize`).

### Run (fast defaults)

```bash
Baselines/other_regression_models.py --no-bayes
```

### Run (Bayesian search)

```bash
Baselines/other_regression_models.py --n-iter 25 --cv 5
```


---

## C. Baseline inverse design: GP‑BO

`Baselines/gp_bo_baseline.py` implements a GP surrogate model and a Bayesian optimization loop to enable **inverse design** (composition search) as a baseline comparator to the VIB+Attention inverse design workflow.

### Run

```bash
python baselines/gp_bo_baseline.py
```

### Notes

- The script loads `data/H_v_dataset.csv` and builds a composition‑plus‑load feature space.
- It performs a stratified split (train/calibration/test) and then fits the GP surrogate.
- Exported traces and candidate sets are written by the script in its configured output section.

---

## D. MD code (LAMMPS script generation + post‑processing)

The MD module lives in `md/md_codes/`. It includes:
- A LAMMPS input generator (`script.py`) driven by `comp.csv`
- Post‑processing scripts for RDF, pair RDF, coordination number, bond statistics, and Voronoi analysis

### Read the MD module documentation

See `md/md_codes/REAMD.md` for full details.

---

## Important Notes:

The repository contains the **primary pipelines** used in the project:
- VIB + Attention regression and uncertainty outputs
- Baseline regressors
- GP‑BO baseline inverse design
- MD structure analysis utilities

For reproduction (including seed averaging, selected hyperparameters, and final figure styling), run the pipelines with the 10 seeds and same settings reported in the manuscript and aggregate the exported results.

---

## License

See `LICENSE`.

---

## Contact
Please contact me: Dr. Anurag Bajpai (a.bajpai@mpi-susmat.de) if you encounter any issues or need clarifications.
