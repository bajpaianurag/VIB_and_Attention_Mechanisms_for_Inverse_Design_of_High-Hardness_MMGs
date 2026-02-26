# VIB + Attention for Inverse Design of High‑Hardness Metallic Multi‑Component Glasses

This repository contains the code and data used for the Nature Communications submission associated with **Variational Information Bottleneck (VIB) + Attention neural networks** for predicting Vickers hardness and for **inverse design** of high‑hardness metallic multi‑component glasses (MMGs).

The repository is organized to allow reviewers and readers to:
1. Reproduce the **supervised regression** results (hardness prediction).
2. Reproduce **baseline regressors** (“other regressors”) used for comparison.
3. Reproduce the **baseline inverse design** workflow using **Gaussian‑Process Bayesian Optimization (GP‑BO)**.
4. Run the **molecular dynamics (MD)** helper scripts used to generate LAMMPS inputs and to perform post‑processing structural analyses.

> **Reproducibility note**: The main ML scripts use a fixed random seed (see each script). For paper‑level statistics (mean ± std over multiple seeds), run the same pipeline over the desired seed list and aggregate the exported metrics.

---

## Repository structure

```
.
├── data/
│   └── H_v_dataset.csv                 # Composition + load + HV target
├── ml/
│   └── vib_attention_nns.py            # Main VIB + Attention neural network pipeline
├── baselines/
│   ├── other_regression_models.py      # Conventional regressors (RF/GB/Lasso/Ridge/KNN/MLP)
│   └── gp_bo_baseline.py               # Baseline GP surrogate + BO inverse design workflow
├── md/
│   └── md_codes/
│       ├── script.py                   # LAMMPS input generator from comp.csv
│       ├── comp.csv                    # Example compositions for MD
│       ├── post-processing/            # RDF, Voronoi, coordination, bond metrics
│       └── REAMD.md                    # MD module documentation
├── results/                            # Auto-created outputs (ignored by git)
├── requirements*.txt                   # Installation options
├── environment.yml                     # Conda environment (recommended for reviewers)
└── CITATION.cff                        # Citation metadata (fill DOI after acceptance)
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

## A. Main model: VIB + Attention (hardness regression)

### Run

```bash
python ml/vib_attention_nns.py
```

### Inputs

- `data/H_v_dataset.csv`
  - Composition columns: elemental fractions (the script auto-detects all columns except `Load` and `HV`)
  - `Load`: indentation load
  - `HV`: Vickers hardness target

### Outputs (typical)

The script generates diagnostic plots and model artifacts in the working directory. Common outputs include:
- Hardness distribution and load–hardness scatter plots
- Train/test composition distribution comparisons
- Saved figures (`*.jpg`) and exported results (`*.csv` / `*.json`) as produced by the script

If you prefer a clean repository, run from a dedicated output directory, e.g.

```bash
mkdir -p results/vib_attention
python ml/vib_attention_nns.py
```

---

## B. Baseline regressors (“other regressors”)

This script provides conventional ML baselines for hardness prediction. It supports either:
- **Default training** (fast) using `--no-bayes`, or
- **Bayesian hyperparameter optimization** (slower; requires `scikit-optimize`).

### Run (fast defaults)

```bash
python baselines/other_regression_models.py --no-bayes
```

### Run (Bayesian search)

```bash
python baselines/other_regression_models.py --n-iter 25 --cv 5
```

### Outputs

- `results/baselines/baselines_metrics.json` (RMSE/MAE/R² and best hyperparameters when used)
- Parity plots for each model in `results/baselines/`

---

## C. Baseline inverse design: GP‑BO

`baselines/gp_bo_baseline.py` implements a GP surrogate model and a Bayesian optimization loop to enable **inverse design** (composition search) as a baseline comparator to the VIB+Attention inverse design workflow.

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

## Reproducing paper figures and tables

The repository contains the **primary pipelines** used in the manuscript:
- VIB + Attention regression and uncertainty outputs
- Baseline regressors
- GP‑BO baseline inverse design
- MD structure analysis utilities

For paper‑exact reproduction (including seed averaging, selected hyperparameters, and final figure styling), run the pipelines with the same seeds and settings reported in the manuscript and aggregate the exported results.

---

## License

See `LICENSE`.

---

## Contact

If you are a reviewer and encounter any issue running the code, please open a GitHub Issue in this repository and include:
- OS, Python version
- The exact command you ran
- The complete traceback / error log
