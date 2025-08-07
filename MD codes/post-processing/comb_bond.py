import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import os
import csv

# Reads atomic positions and types from a CFG file
def read_cfg(filename):
	with open(filename, 'r') as f:
		lines = f.readlines()

	# Read number of atoms and box scaling factor
	num_atoms = int(lines[0].split('=')[1].strip())
	scale = float(lines[1].split()[2])

	# Construct the box matrix (H matrix), scaled
	h_matrix = np.array([
		[float(lines[2].split()[2]), float(lines[3].split()[2]), float(lines[4].split()[2])],
		[float(lines[5].split()[2]), float(lines[6].split()[2]), float(lines[7].split()[2])],
		[float(lines[8].split()[2]), float(lines[9].split()[2]), float(lines[10].split()[2])]
	]) * scale

	# Read species and convert fractional coordinates to Cartesian
	atom_lines = lines[13:]
	species = []
	positions = []

	for i in range(0, len(atom_lines), 3):
		symbol = atom_lines[i+1].strip()
		frac_coords = list(map(float, atom_lines[i+2].split()))
		cart_coords = np.dot(frac_coords, h_matrix)
		positions.append(cart_coords)
		species.append(symbol)

	return np.array(positions), np.array(species), h_matrix

# Apply minimum image convention to account for periodic boundaries
def minimum_image(vec, box):
	return vec - np.round(np.dot(vec, np.linalg.inv(box))) @ box

# Computes bond angle between two vectors (not used in current run)
def compute_bond_angle(vec1, vec2):
	cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
	cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
	return np.arccos(cosine_angle) * 180 / np.pi

# Returns a sorted key for a triple of species (e.g., 'B-Co-Fe')
def bond_angle_key(sp1, sp2, sp3):
	return "-".join(sorted([sp1, sp2, sp3]))

# Computes pairwise bond lengths for atom pairs involving B or Re
def compute_pairwise_bond_and_angle(positions, species, box, cutoff=3.0):
	N = len(positions)
	bond_lengths = {}

	for i in range(N):
		for j in range(i + 1, N):
			# Only consider pairs where at least one atom is B or Re
			if not (species[i] == 'B' or species[i] == 'Re' or species[j] == 'B' or species[j] == 'Re'):
				continue

			delta = minimum_image(positions[i] - positions[j], box)
			dist = np.linalg.norm(delta)

			if dist < cutoff:
				pair = sorted([species[i], species[j]])
				key = f"{pair[0]}-{pair[1]}"
				if key not in bond_lengths:
					bond_lengths[key] = []
				bond_lengths[key].append(dist)

	return bond_lengths, None  # bond_angles placeholder (unused)

# Save raw bond length values per bond type to individual CSV files
def save_csv_per_category(data_dict, prefix, folder="bond"):
	os.makedirs(folder, exist_ok=True)
	for key, values in data_dict.items():
		if len(values) == 0:
			continue
		filename = os.path.join(folder, f"{prefix}_{key.replace('-', '_')}.csv")
		np.savetxt(filename, values, delimiter=",", header=prefix, comments="")
		print(f"✔ Saved: {filename}")

# Save histogram data per bond type as CSV (e.g., for plotting externally)
def save_histogram_csv(data_dict, prefix, bins, folder="bond"):
	os.makedirs(folder, exist_ok=True)

	for key, values in data_dict.items():
		if len(values) == 0:
			continue

		counts, edges = np.histogram(values, bins=bins)
		bin_centers = 0.5 * (edges[:-1] + edges[1:])

		filename = os.path.join(folder, f"{key.replace('-', '_')}_{prefix}.csv")
		data = np.column_stack((bin_centers, counts))
		np.savetxt(filename, data, delimiter=",", header=f"{prefix},count", comments="", fmt="%.2f,%d")


# --- Main execution block ---
if __name__ == "__main__":
	filename = "CFG/test_160000.cfg"
	cutoff = 3.0  # bond length cutoff in Å

	# Read atomic data from file
	positions, species, box = read_cfg(filename)

	# Compute bond lengths (only those involving B or Re)
	bond_lengths, bond_angles = compute_pairwise_bond_and_angle(positions, species, box, cutoff)

	# Print number of bonds per element pair
	print("Bond count per pair type:")
	for key in bond_lengths:
		print(f"{key:7s}: {len(bond_lengths[key])} bonds")

	# Save bond length histograms per bond type
	save_histogram_csv(bond_lengths, prefix="bond_length", bins=np.arange(0, 3.1, 0.02))
