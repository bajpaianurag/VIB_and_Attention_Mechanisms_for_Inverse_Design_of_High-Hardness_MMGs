import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import os
import csv

# Reads atomic configuration from a CFG file
def read_cfg(filename):
	with open(filename, 'r') as f:
		lines = f.readlines()

	# Parse number of atoms and box scale factor
	num_atoms = int(lines[0].split('=')[1].strip())
	scale = float(lines[1].split()[2])

	# Parse box matrix (H matrix), scaled by the scale factor
	h_matrix = np.array([
		[float(lines[2].split()[2]), float(lines[3].split()[2]), float(lines[4].split()[2])],
		[float(lines[5].split()[2]), float(lines[6].split()[2]), float(lines[7].split()[2])],
		[float(lines[8].split()[2]), float(lines[9].split()[2]), float(lines[10].split()[2])]
	]) * scale

	# Parse atom types and positions
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

# Apply minimum image convention for periodic boundary conditions
def minimum_image(vec, box):
	return vec - np.round(np.dot(vec, np.linalg.inv(box))) @ box

# Computes bond lengths and bond angles for atoms within a given cutoff
def compute_bonds_angles(positions, box, cutoff=3.0):
	N = len(positions)
	bond_lengths = []
	angles = []

	# Initialize neighbor list
	neighbor_list = [[] for _ in range(N)]

	# --- Compute pairwise distances and build neighbor lists ---
	for i in range(N):
		for j in range(i + 1, N):
			delta = minimum_image(positions[i] - positions[j], box)
			dist = np.linalg.norm(delta)
			if dist < cutoff:
				bond_lengths.append(dist)
				neighbor_list[i].append((j, delta))
				neighbor_list[j].append((i, -delta))

	# --- Compute bond angles using combinations of neighbors ---
	for i in range(N):
		neighbors = neighbor_list[i]
		if len(neighbors) < 2:
			continue
		for (j, vec1), (k, vec2) in combinations(neighbors, 2):
			# Compute angle between vec1 and vec2
			cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
			cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
			angle = np.arccos(cosine_angle) * 180 / np.pi
			angles.append(angle)

	return np.array(bond_lengths), np.array(angles)

# Saves histogram data (bin center and count) to a CSV file
def save_histogram_to_csv(data, bins, filename, xlabel):
	counts, bin_edges = np.histogram(data, bins=bins)
	bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

	with open(filename, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow([xlabel, "count"])
		for val, count in zip(bin_centers, counts):
			writer.writerow([val, count])

# --- Main execution block ---
if __name__ == "__main__":
	# Input CFG file and cutoff distance
	filename = "CFG/test_160000.cfg"
	cutoff = 3.0  # bond distance cutoff in Ångström

	# Read atomic positions and box
	positions, species, box = read_cfg(filename)

	# Compute bond lengths and angles
	bonds, angles = compute_bonds_angles(positions, box, cutoff=cutoff)

	# Save histogram data to CSV
	os.makedirs("bond", exist_ok=True)
	save_histogram_to_csv(bonds, bins=np.arange(0, 3.1, 0.02), filename="bond/all_bond_length_histogram.csv", xlabel="length(Å)")
	save_histogram_to_csv(angles, bins=np.arange(0, 181, 1), filename="bond/all_bond_angle_histogram.csv", xlabel="angle(degrees)")
