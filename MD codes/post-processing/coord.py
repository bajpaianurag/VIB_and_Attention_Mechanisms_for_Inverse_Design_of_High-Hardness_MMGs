import numpy as np
from collections import Counter
import pandas as pd

# Reads atomic positions and species from a CFG file
def read_cfg(filename):
	with open(filename, 'r') as f:
		lines = f.readlines()
	
	# Parse number of atoms and box scaling factor
	num_atoms = int(lines[0].split('=')[1].strip())
	scale = float(lines[1].split()[2])  # A = 1 Angstrom

	# Construct and scale box (H) matrix
	h_matrix = np.array([
		[float(lines[2].split()[2]), float(lines[3].split()[2]), float(lines[4].split()[2])],
		[float(lines[5].split()[2]), float(lines[6].split()[2]), float(lines[7].split()[2])],
		[float(lines[8].split()[2]), float(lines[9].split()[2]), float(lines[10].split()[2])]
	]) * scale

	# Extract atomic symbols and fractional coordinates
	atom_lines = lines[13:]  # Skip header lines
	species = []
	positions = []

	for i in range(0, len(atom_lines), 3):
		symbol = atom_lines[i+1].strip()
		frac_coords = list(map(float, atom_lines[i+2].split()))
		cart_coords = np.dot(frac_coords, h_matrix)  # Convert to Cartesian
		positions.append(cart_coords)
		species.append(symbol)

	return np.array(positions), np.array(species), h_matrix

# Computes coordination numbers for all atoms within a cutoff radius
def compute_coordination_numbers(positions, box, cutoff=3.0):
	N = len(positions)
	cn = np.zeros(N, dtype=int)

	for i in range(N):
		for j in range(N):
			if i == j:
				continue
			# Apply periodic boundary condition (minimum image)
			delta = positions[i] - positions[j]
			delta -= np.round(np.dot(delta, np.linalg.inv(box))) @ box
			r = np.linalg.norm(delta)

			if r < cutoff:
				cn[i] += 1  # Count neighbor within cutoff

	return cn

# --- Main execution block ---
if __name__ == "__main__":
	filename = "CFG/test_160000.cfg"  # Replace with your file path
	cutoff = 3.0  # Cutoff radius in Å for coordination number

	# Load atomic data
	positions, species, box = read_cfg(filename)

	# Compute coordination numbers
	cn = compute_coordination_numbers(positions, box, cutoff=cutoff)

	# Create overall coordination number histogram
	cn_counts = Counter(cn)
	print("\nCoordination Number Histogram:")
	for cn_val in sorted(cn_counts):
		print(f"CN = {cn_val:2d} : {cn_counts[cn_val]} atoms")

	# Save total coordination number histogram to CSV
	df_cn_hist = pd.DataFrame({
		"coordination_number": list(cn_counts.keys()),
		"number_of_atoms": list(cn_counts.values())
	})
	df_cn_hist = df_cn_hist.sort_values(by="coordination_number")
	df_cn_hist.to_csv("coordination_histogram.csv", index=False)

	# Filter indices for B and Re atoms
	b_indices = np.where(species == "B")[0]
	re_indices = np.where(species == "Re")[0]

	# Extract their coordination numbers
	b_cn = cn[b_indices]
	re_cn = cn[re_indices]

	# Create and save histogram for B atoms
	b_cn_counts = Counter(b_cn)
	df_b_cn_hist = pd.DataFrame({
		"coordination_number": list(b_cn_counts.keys()),
		"number_of_atoms": list(b_cn_counts.values())
	})
	df_b_cn_hist = df_b_cn_hist.sort_values(by="coordination_number")
	df_b_cn_hist.to_csv("B_coordination_histogram.csv", index=False)

	# Create and save histogram for Re atoms
	re_cn_counts = Counter(re_cn)
	df_re_cn_hist = pd.DataFrame({
		"coordination_number": list(re_cn_counts.keys()),
		"number_of_atoms": list(re_cn_counts.values())
	})
	df_re_cn_hist = df_re_cn_hist.sort_values(by="coordination_number")
	df_re_cn_hist.to_csv("Re_coordination_histogram.csv", index=False)
