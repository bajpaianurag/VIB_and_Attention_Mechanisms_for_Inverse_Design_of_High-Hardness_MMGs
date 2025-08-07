import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
from collections import defaultdict
import pandas as pd

# Reads atomic positions and species from a CFG file
def read_cfg(filename):
	with open(filename, 'r') as f:
		lines = f.readlines()
	
	# Parse number of atoms and scale factor
	num_atoms = int(lines[0].split('=')[1].strip())
	scale = float(lines[1].split()[2])  # A = 1 Angstrom

	# Construct scaled H matrix (simulation cell)
	h_matrix = np.array([
		[float(lines[2].split()[2]), float(lines[3].split()[2]), float(lines[4].split()[2])],
		[float(lines[5].split()[2]), float(lines[6].split()[2]), float(lines[7].split()[2])],
		[float(lines[8].split()[2]), float(lines[9].split()[2]), float(lines[10].split()[2])]
	]) * scale

	# Read atom types and positions
	atom_lines = lines[13:]
	species = []
	positions = []

	for i in range(0, len(atom_lines), 3):
		symbol = atom_lines[i+1].strip()
		frac_coords = list(map(float, atom_lines[i+2].split()))
		cart_coords = np.dot(frac_coords, h_matrix)  # Convert to Cartesian coordinates
		positions.append(cart_coords)
		species.append(symbol)

	return np.array(positions), np.array(species), h_matrix

# Computes pairwise RDF g(r) for all element pairs
def compute_rdf_pairs(positions, species, box, r_max=10.0, dr=0.1):
	r_values = np.linspace(0.5 * dr, r_max - 0.5 * dr, int(r_max / dr))  # Bin centers
	num_bins = len(r_values)
	volume = np.linalg.det(box)  # Simulation box volume
	species_types = np.unique(species)  # Unique element types
	rdf_data = {}

	# Loop over all combinations of element pairs (including A-A)
	for s1, s2 in combinations_with_replacement(species_types, 2):
		rdf = np.zeros(num_bins)
		indices1 = np.where(species == s1)[0]
		indices2 = np.where(species == s2)[0]

		for i in indices1:
			for j in indices2:
				if s1 == s2 and j <= i:
					continue  # Avoid double counting for same-species pairs

				# Apply periodic boundary condition
				delta = positions[i] - positions[j]
				delta -= np.round(np.dot(delta, np.linalg.inv(box))) @ box
				r = np.linalg.norm(delta)

				# Accumulate into RDF bins
				if r < r_max:
					bin_index = int(r / dr)
					rdf[bin_index] += 1

		# Normalize RDF
		normalization = (4/3) * np.pi * ((r_values + 0.5*dr)**3 - (r_values - 0.5*dr)**3)  # Shell volume
		n1 = len(indices1)
		n2 = len(indices2) if s1 != s2 else len(indices2) - 1  # Avoid self count
		density = n2 / volume
		ideal = n1 * density * normalization  # Ideal gas reference
		g_r = rdf / ideal
		rdf_data[(s1, s2)] = (r_values, g_r)

	return rdf_data

# Save RDF data for each element pair to individual CSV files
def save_rdf_to_csv(rdf_data, output_dir="rdf_csv"):
	for (s1, s2), (r, g_r) in rdf_data.items():
		df = pd.DataFrame({
			"r (Angstrom)": r,
			"g(r)": g_r
		})
		filename = f"RDF/{s1}_{s2}_rdf.csv"
		df.to_csv(filename, index=False)

# --- Main execution ---
if __name__ == "__main__":
	filename = "CFG/test_160000.cfg"  # Replace with your actual file path

	# Load atomic positions and types
	positions, species, box = read_cfg(filename)

	# Compute RDFs for all species pairs
	rdf_data = compute_rdf_pairs(positions, species, box, r_max=10.0, dr=0.1)
	
	# Save RDF results to CSV
	save_rdf_to_csv(rdf_data)
