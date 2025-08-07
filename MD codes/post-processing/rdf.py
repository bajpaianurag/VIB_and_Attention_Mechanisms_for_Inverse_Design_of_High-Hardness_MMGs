import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

# Reads atomic positions and types from a CFG file
def read_cfg(filename):
	with open(filename, 'r') as f:
		lines = f.readlines()
	
	# Read number of atoms and box scale factor
	num_atoms = int(lines[0].split('=')[1].strip())
	scale = float(lines[1].split()[2])  # A = 1 Angstrom

	# Construct and scale H (simulation cell) matrix
	h_matrix = np.array([
		[float(lines[2].split()[2]), float(lines[3].split()[2]), float(lines[4].split()[2])],
		[float(lines[5].split()[2]), float(lines[6].split()[2]), float(lines[7].split()[2])],
		[float(lines[8].split()[2]), float(lines[9].split()[2]), float(lines[10].split()[2])]
	]) * scale

	# Read atomic species and convert fractional to Cartesian coordinates
	atom_lines = lines[13:]  # Skip header
	species = []
	positions = []

	for i in range(0, len(atom_lines), 3):
		symbol = atom_lines[i+1].strip()
		frac_coords = list(map(float, atom_lines[i+2].split()))
		cart_coords = np.dot(frac_coords, h_matrix)
		positions.append(cart_coords)
		species.append(symbol)

	return np.array(positions), np.array(species), h_matrix

# Computes total radial distribution function g(r) for all atoms
def compute_rdf(positions, box, r_max=10.0, dr=0.1):
	num_bins = int(r_max / dr)
	rdf = np.zeros(num_bins)
	N = len(positions)
	volume = np.linalg.det(box)
	density = N / volume  # average number density

	# Compute pair distances and accumulate counts in bins
	for i in range(N):
		for j in range(i + 1, N):
			delta = positions[i] - positions[j]
			# Apply periodic boundary conditions (minimum image convention)
			delta -= np.round(np.dot(delta, np.linalg.inv(box)) @ np.eye(3)) @ box
			r = np.linalg.norm(delta)
			if r < r_max:
				bin_index = int(r / dr)
				rdf[bin_index] += 2  # count both i-j and j-i

	# Bin centers
	r_values = np.linspace(0.5 * dr, r_max - 0.5 * dr, num_bins)

	# Compute shell volumes for normalization
	shell_volumes = 4 / 3 * np.pi * ((r_values + 0.5 * dr)**3 - (r_values - 0.5 * dr)**3)

	# Expected ideal gas counts in each shell
	ideal_counts = density * shell_volumes * N

	# Final RDF
	g_r = rdf / ideal_counts

	return r_values, g_r

# --- Main execution ---
if __name__ == "__main__":
	filename = "CFG/test_160000.cfg"  # Path to CFG file

	# Load structure
	positions, species, box = read_cfg(filename)

	# Compute RDF
	r, g_r = compute_rdf(positions, box, r_max=10.0, dr=0.1)

	# Save to CSV file
	df = pd.DataFrame({
		"r (Angstrom)": r,
		"g(r)": g_r
	})
	filename = "RDF/all.csv"
	df.to_csv(filename, index=False)
