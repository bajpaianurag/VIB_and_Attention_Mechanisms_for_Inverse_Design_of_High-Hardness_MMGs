
# VIB and Attention Mechanisms for Inverse Design of High-Hardness Multicomponent Metallic Glasses (MMGs)

This repository hosts the codebase, dataset, and pretrained models used in our study on **Variational Information Bottleneck (VIB)** and **Attention-based Neural Networks** for **inverse design of high-hardness multicomponent metallic glasses (MMGs)**. The proposed framework integrates probabilistic latent representations and attention mechanisms to learn composition–hardness relationships and to discover new glassy compositions with extreme mechanical performance.

---

## 🚀 Project Highlights

- **Forward Model**: Predicts Vickers hardness (HV) from alloy composition using:
  - Variational Information Bottleneck (VIB) networks
  - Attention over elemental fractions and latent features
  - Baseline regression models for benchmarking

- **Inverse Design Module**: Generates novel alloy compositions targeting a desired hardness using:
  - Latent space optimization and sampling
  - Composition vector normalization
  - Constraints on element fraction summation (∑xᵢ = 1)

- **Dataset**: Empirical and simulated data for known MMGs, including:
  - Elemental compositions
  - Experimental hardness values (HV)
  - Glass-forming ability indicators

---

## 🗂️ Repository Structure

```
.
├── H_v_dataset.csv                          # Main dataset: Elemental fractions and hardness values
├── VIB+Attention_NNs.py                     # Main script: VIB + Attention neural network training
├── Other Regression Models.py               # Baseline models: RF, SVR, XGBoost, etc.
├── vib_attention_model.weights.h5           # Pretrained VIB-Attention model weights
├── vib_attention_full_model.keras           # Full Keras model including encoder and decoder
├── Inverse_Design_Module.ipynb              # Latent space optimization for target HV
├── Inverse_Design_Helper_Functions.py       # Helper functions for inverse design
├── MD codes.zip                             # LAMMPS/MD simulation code for validation of structures
├── requirements.txt                         # Python dependencies
└── README.md                                # You're reading it!
```

---

## 📊 Dataset Description

**File:** `H_v_dataset.csv`

| Column         | Description                                      |
|----------------|--------------------------------------------------|
| Element_x      | Molar fraction of element x (e.g., Fe, Ni, Cu)   |
| HV             | Measured or predicted Vickers hardness (kg/mm²)  |
| GFA_flag       | Optional: Glass-forming ability indicator (0/1)  |

Total number of data points: **~400** MMG compositions.

---

## 🧠 Forward Model: VIB + Attention

The model architecture implements:

- **VIB Encoder**: Compresses input composition to latent Gaussian space
- **KL Annealing**: Gradually activates KL regularization during training
- **Multi-Head Attention**: Attends to element-specific latent features
- **Regression Head**: Outputs predicted hardness value

Training parameters (configurable in script):

- Batch size: 64  
- Optimizer: Adam  
- Epochs: 500  
- Learning rate scheduler  
- Early stopping based on validation loss

---

## 🔁 Baseline Models

Implemented in `Other Regression Models.py`:

- Random Forest Regressor  
- Support Vector Regression  
- Gradient Boosting Regressor  
- XGBoost  
- k-NN  
- MLP Regressor  

Performance comparison includes RMSE and R² metrics for each.

---

## 🎯 Inverse Design Pipeline

Implemented in: `Inverse_Design_Module.ipynb`

### Workflow:
1. **Latent sampling** around known high-HV clusters
2. **Target-guided optimization** of latent vector toward desired HV
3. **Decoding** into alloy compositions
4. **Post-processing** to ensure:
   - Non-negativity (xᵢ ≥ 0)
   - Normalization (∑xᵢ = 1)
   - Filter based on GFA constraints (optional)

---

## 🧪 MD Simulation Codes

**File:** `MD codes.zip`

Contains sample scripts and input decks for LAMMPS-based mechanical testing simulations (e.g., nanoindentation, tension tests) to validate predicted MMGs.

---

## 🛠️ Installation

### 1. Clone this repository
```bash
git clone https://github.com/bajpaianurag/VIB_and_Attention_Mechanisms_for_Inverse_Design_of_High-Hardness_MMGs.git
cd VIB_and_Attention_Mechanisms_for_Inverse_Design_of_High-Hardness_MMGs
```

### 2. Create a Python environment
```bash
conda create -n mmg_design python=3.9
conda activate mmg_design
pip install -r requirements.txt
```

---

## 📌 Requirements

See `requirements.txt` for exact versions. Main packages:
- numpy
- pandas
- scikit-learn
- tensorflow (>=2.10)
- matplotlib
- seaborn
- keras
- xgboost

---

## 📈 Sample Results

| Model                | RMSE (test) | R² (test) |
|---------------------|-------------|-----------|
| VIB + Attention      | **92.1**    | **0.92**  |
| Random Forest        | 134.5       | 0.85      |
| SVR (RBF)            | 150.2       | 0.80      |

---

## 📚 Reference

If you use this repository in your work, please cite:

> Bajpai, A., et al. *Inverse Design of High-Hardness Multicomponent Metallic Glasses using Variational Bottleneck and Attention Mechanisms*, (2025). [Preprint DOI / Journal]

---

## 🤝 Acknowledgments

This work was carried out at **IIT Indore** and is supported by:
- Indo-German Science & Technology Centre (IGSTC)
- [Other Funding Acknowledgements]

Collaborators:
- Dr. Jasmin Duarte Correa (MPIE, Germany)  
- Prof. Dierk Raabe (MPIE, Germany)  
- Dr. Laurine Choisez (UC Louvain, Belgium)

---

## 📬 Contact

For questions, collaborations, or discussions:

**Dr. Anurag Bajpai**  
Assistant Professor  
Department of Metallurgical Engineering and Materials Science  
Indian Institute of Technology Indore  
📧 anurag.bajpai@iiti.ac.in
