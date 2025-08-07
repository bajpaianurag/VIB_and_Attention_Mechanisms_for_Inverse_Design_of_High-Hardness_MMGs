### Project: LAMMPS Script Generator and Structural Analysis Toolkit

This repository provides a complete pipeline to:

1. Automatically generate **LAMMPS simulation input scripts** from compositional data.
2. Perform **post-simulation structure analysis** using a variety of geometry-based techniques.

---

### 📁 Directory Structure

```
.
├── script.py              # LAMMPS script generator based on comp.csv
├── comp.csv               # Input file: elemental compositions per system
└── post-processing/
    ├── bond.py            # Calculates bond lengths and angles from CFG files
    ├── comb_bond.py       # Calculates bond lengths grouped by atom pairs (B or Re involved)
    ├── coord.py           # Computes coordination numbers for all atoms, and separately for B and Re
    ├── pair_rdf.py        # Computes RDF (g(r)) for each atomic pair
    ├── rdf.py             # Computes total RDF (g(r)) for all atoms
    └── voronoi.py         # Voronoi analysis for local atomic environment, focused on B and Re
```

---

### 📄 script.py

**Purpose:**
Generates LAMMPS input scripts based on element fractions listed in `comp.csv`.

**Input:**

* `comp.csv`: Each row contains element fractions (B, Co, Cr, Re, W, Fe, Ni, V).

**Output:**

* LAMMPS script files located in `results/{row_index}_{rep}/lammps_script`.

**Usage:**
Run directly:

```bash
python script.py
```

---

### 📁 post-processing/

All scripts here are designed to process the output structures (e.g., `CFG/test_*.cfg`) from LAMMPS simulations.

---

#### 🔹 bond.py

* Calculates all bond lengths and bond angles between atoms.
* Outputs two CSV files:

  * `bond/all_bond_length_histogram.csv`
  * `bond/all_bond_angle_histogram.csv`

#### 🔹 comb\_bond.py

* Computes bond lengths **by element pairs** (involving B or Re).
* Outputs one histogram per pair to `bond/*_bond_length.csv`.

#### 🔹 coord.py

* Calculates the **coordination number** (number of neighbors within cutoff).
* Outputs:

  * `coordination_histogram.csv`
  * `B_coordination_histogram.csv`
  * `Re_coordination_histogram.csv`

#### 🔹 pair\_rdf.py

* Computes **pairwise radial distribution functions** (g(r)) for all unique element pairs.
* Outputs: one CSV per pair in the `RDF/` directory.

#### 🔹 rdf.py

* Computes **total RDF** across all atoms.
* Output: `RDF/all.csv`

#### 🔹 voronoi.py

* Performs **Voronoi analysis** for atoms of type B and Re.
* Outputs:

  * Index distribution in the console
  * Example `.xyz` clusters (Voronoi polyhedra) in `voronoi_clusters/`

---

### 🔧 Requirements

* Python 3.x
* Libraries:

  ```bash
  pip install numpy pandas matplotlib ase pymatgen
  ```

---

### 📦 How to Run (Typical Workflow)

1. **Generate LAMMPS input scripts:**

   ```bash
   python script.py
   ```

2. **Run LAMMPS simulations externally using the generated scripts.**

3. **Run post-processing analyses on `CFG/test_*.cfg`:**

   ```bash
   cd post-processing
   python bond.py
   python coord.py
   python voronoi.py
   # etc.
   ```

---

### 📞 Contact

* j.wang@mpi-susmat.de (Dr. Jaemin Wang)
* a.bajpai@mpi-susmat.de (Dr. Anurag Bajpai)
