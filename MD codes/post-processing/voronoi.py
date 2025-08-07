import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.local_env import VoronoiNN
from collections import defaultdict, Counter
from ase import Atoms
from ase.io import write
import os

# Reads a CFG file and converts it to a pymatgen Structure object
def read_cfg_to_structure(filename):
	with open(filename, 'r') as f:
		lines = f.readlines()

	# Extract number of atoms and box scale factor
	num_atoms = int(lines[0].split('=')[1].strip())
	scale = float(lines[1].split()[2])

	# Construct and scale the simulation box matrix (H matrix)
	h = np.array([
		[float(lines[2].split()[2]), float(lines[3].split()[2]), float(lines[4].split()[2])],
		[float(lines[5].split()[2]), float(lines[6].split()[2]), float(lines[7].split()[2])],
		[float(lines[8].split()[2]), float(lines[9].split()[2]), float(lines[10].split()[2])]
	]) * scale

	species = []
	frac_coords = []

	# Parse species and fractional coordinates
	atom_lines = lines[13:]
	for i in range(0, len(atom_lines), 3):
		symbol = atom_lines[i+1].strip()
		coord = list(map(float, atom_lines[i+2].split()))
		species.append(symbol)
		frac_coords.append(coord)

	# Return pymatgen Structure object
	lattice = Lattice(h)
	structure = Structure(lattice, species, frac_coords)
	return structure

# Analyze Voronoi polyhedra centered on atoms of a given species
def analyze_voronoi_by_species(structure, target_species):
	vnn = VoronoiNN(allow_pathological=True)
	index_map = defaultdict(list)
	index_counter = Counter()

	for i in range(len(structure)):
		# Only consider atoms of the target species
		if structure[i].specie.symbol != target_species:
			continue

		# Get Voronoi polyhedron for this atom
		poly_info = vnn.get_voronoi_polyhedra(structure, i)
		faces = list(poly_info.values())
		face_sides = [face["n_verts"] for face in faces]  # number of vertices per face

		# Compute Voronoi index: count of n-sided faces (n = 3, 4, 5, ...)
		max_sides = max(face_sides) if face_sides else 0
		index = [face_sides.count(k) for k in range(3, max_sides + 1)]
		index_str = "<" + ",".join(str(n) for n in index) + ">"

		# Save if the index is not too long (optional filter)
		if len(index) <= 4:
			neighbor_indices = [nbr['site'].index for nbr in poly_info.values()]
			index_map[index_str].append((i, neighbor_indices))
			index_counter[index_str] += 1

	return index_counter, index_map

# Export a selected atom and its neighbors as an .xyz file (for visualization)
def export_local_cluster(structure, atom_id, neighbor_ids, filename):
	site_indices = [atom_id] + neighbor_ids
	selected_sites = [structure[i] for i in site_indices]

	new_structure = Structure(
		lattice=structure.lattice,
		species=[site.specie for site in selected_sites],
		coords=[site.coords for site in selected_sites],
		coords_are_cartesian=True
	)

	# Convert to ASE Atoms object and write as XYZ
	atoms = Atoms(
		symbols=[s.specie.symbol for s in new_structure],
		positions=[s.coords for s in new_structure],
		cell=new_structure.lattice.matrix,
		pbc=True
	)
	write(filename, atoms)
	print(f"✔ Saved: {filename}")

# Check if an atom is well inside the simulation box (not near boundaries)
def is_well_inside(structure, index, frac_margin=0.1):
	fcoord = structure[index].frac_coords
	return all(frac_margin < fc < 1 - frac_margin for fc in fcoord)

# --- Main Execution ---
if __name__ == "__main__":
	filename = "CFG/test_160000.cfg"
	structure = read_cfg_to_structure(filename)
	os.makedirs("voronoi_clusters", exist_ok=True)

	# Loop through each target element (e.g., B, Re)
	for element in ['B', 'Re']:
		print(f"\n===== {element}-centered Voronoi index distribution =====")

		# Analyze Voronoi indices
		index_counter, index_map = analyze_voronoi_by_species(structure, element)

		# Print index distribution statistics
		total = sum(index_counter.values())
		for index_str, count in index_counter.most_common():
			percent = 100 * count / total
			print(f"{index_str:15s}: {count:5d} atoms ({percent:.2f}%)")

		print(f"\n--- Saving clusters for top 10 indices ({element}) ---")
		top10 = index_counter.most_common(10)

		# For each of top 10 Voronoi indices, export one example cluster
		for rank, (index_str, _) in enumerate(top10, start=1):
			examples = index_map[index_str]
			for atom_id, neighbors in examples:
				if is_well_inside(structure, atom_id):  # avoid atoms near box edge
					filename_out = f"voronoi_clusters/{element}_voro_{index_str.replace('<','').replace('>','').replace(',','-')}.xyz"
					export_local_cluster(structure, atom_id, neighbors, filename_out)
					break  # only export one example per index
