# This script reads composition data from a CSV file and generates LAMMPS input scripts
# for each composition. The scripts define atom types, assign masses, set up lattice 
# configurations, and run temperature-controlled simulations.

import pandas as pd
import os
import math
import random

# Load the uploaded CSV file containing element compositions
file_path = 'comp.csv'
df = pd.read_csv(file_path)

# List of elements in the dataset
elements = ['B', 'Co', 'Fe', 'Hf', 'Nb', 'Ru', 'W', 'Zr']

# Atomic masses for the elements (for LAMMPS 'mass' command)
atomic_masses = {
    "B": 10.81, "Co": 58.933, "Fe": 55.845, "Hf": 178.49,
    "Nb": 92.9063, "Ru": 101.07, "W": 183.84, "Zr": 91.224
}

# Function to generate a LAMMPS input script for each material composition
def generate_lammps_script_with_all_elements(row, index, rep):
	# Initialize simulation box and lattice parameters
	counter = 0
	box_x = 24
	box_y = 24
	box_z = 24
	lat_st = 'bcc'

	# Count how many element types are present in the current row
	for element in elements:
		if row[element] > 0.001:
			counter += 1

	# Begin constructing the LAMMPS script with fixed setup and box configuration
	script = f"""# LAMMPS script generated for composition index {index}, replicate {rep}
units            metal
boundary         p p p
atom_style       atomic
atom_modify      sort 0 0
neighbor         0.3 bin

lattice {lat_st} 2.8 
region mybox block 0 {box_x} 0 {box_y} 0 {box_z}
create_box {counter} mybox  

create_atoms 1 box
"""

	# Estimate total number of atoms based on lattice type
	if lat_st == 'bcc':
		num_atoms = box_x * box_y * box_z * 2
	elif lat_st == 'fcc':
		num_atoms = box_x * box_y * box_z * 4

	# Assign element IDs for LAMMPS types
	element_dict = {'B':1}
	present_elements = [element for element in elements if row[element] > 0.001]
	for key in element_dict.keys():
		if key in present_elements:
			present_elements.remove(key)
	element_dict |= {item: index+len(element_dict)+1 for index, item in enumerate(present_elements)}

	# Assign atomic types based on compositions using 'set type/subset'
	cor = 1
	for element in elements:
		composition = row[element]
		if composition > 0.001 and element_dict[element] != 1:
			script += f"set type 1 type/subset {element_dict[element]} {int(round(composition * num_atoms))} {round(random.random()*1e+5) + element_dict[element]}  # {composition*100:.1f}% {element}\n"
			cor /= (1-math.ceil(composition*1000)/1000)

	# Add atomic mass declarations
	script += "\n"
	for element in elements:
		if row[element] > 0.001:
			script += f"mass {element_dict[element]} {atomic_masses[element]}  # {element}\n"

	# Add pair style and coefficient line with element order
	present_elements = list(element_dict.keys())
	present_elements_str = " ".join(present_elements)
	script += f"\npair_style       e3gnn\n"
	script += f"pair_coeff       * * /home/onwer/2.sdb2/sevenn_potential/deployed_serial.pt {present_elements_str}\n"

	# Set temperature parameters and time steps
	start_temp = 50
	end_temp = 4000
	step_num = 40000

	# Add compute and minimization commands
	script += """
# Setup thermodynamic output and simulation parameters
compute         c1 all temp
compute_modify  c1 dynamic/dof yes extra/dof 0

thermo          100
thermo_style    custom step pe ke press temp vol enthalpy density etotal
thermo_modify   format float "% .6e"
thermo_modify   temp c1
timestep        0.001

neigh_modify    every 20 delay 0 check yes

# Setup output directory
shell           "rm -r CFG"
shell           mkdir CFG

# Dump configuration snapshots
dump            d1 all cfg 1000 CFG/test_*.cfg mass type xs ys zs
dump_modify     d1 element {elements_str} sort id pad 5

# Initial structure minimization
velocity        all create {start_temp} 492851 mom yes rot yes dist gaussian
min_style       fire
minimize        1e-10  1e-10  100  100
""".replace("{elements_str}", present_elements_str).replace("{start_temp}", str(start_temp))

	# Add RDF compute section based on number of present elements
	if counter == 6:
		script += """
# Radial distribution functions (RDF) setup for 6-element systems
compute         rdf_all all rdf 100
compute         rdf_pairs all rdf 100 1 1 1 2 1 3 1 4 1 5 1 6 2 2 2 3 2 4 2 5 2 6 3 3 3 4 3 5 3 6 4 4 4 5 4 6 5 5 5 6 6 6
"""
	elif counter == 5:
		script += """
# RDF setup for 5-element systems
compute         rdf_all all rdf 100
compute         rdf_pairs all rdf 100 1 1 1 2 1 3 1 4 1 5 2 2 2 3 2 4 2 5 3 3 3 4 3 5 4 4 4 5 5 5
"""

	# Equilibration at high temp, then quenching, then low temp equilibration
	script += f"""
# High temperature equilibration
fix             f1 all npt temp {start_temp} {end_temp} 1 iso 0 0 5 
variable        end equal ($(step)+{step_num})
variable        inter equal ({step_num})/10
variable        start_step equal $(step)
label           lab1
"""
	script += """
run             ${inter} start ${start_step} stop ${end} post no
write_restart   restart.melt
if              "$(step)<${end}" then "jump lammps_script lab1"
unfix           f1
"""

	script += f"""
# Constant high temperature equilibration
fix             f1 all npt temp {end_temp} {end_temp} 1 iso 0 0 5
variable        end equal ($(step)+{step_num})
variable        inter equal ({step_num})/10
variable        start_step equal $(step)
label           lab12
"""
	script += """
run             ${inter} start ${start_step} stop ${end} post no
write_restart   restart.equilib
if              "$(step)<${end}" then "jump lammps_script lab12"
unfix           f1
"""

	script += f"""
# Quenching to low temperature
fix             f2 all npt temp {end_temp} {start_temp} 1 iso 0 0 5
variable        end equal ($(step)+{step_num})
variable        quench_inter equal {step_num/10}
variable        start_step equal $(step)
label           lab2
"""
	script += """
run             ${quench_inter} start ${start_step} stop ${end} post no
write_restart   restart.quench
if              "$(step)<${end}" then "jump lammps_script lab2"
unfix           f2
"""

	script += f"""
# Final low temperature equilibration
fix             f3 all npt temp {start_temp} {start_temp} 1 iso 0 0 5
variable        end equal ($(step)+{step_num})
variable        inter equal ({step_num})/10
variable        start_step equal $(step)
label           lab3
"""
	script += """
run             ${inter} start ${start_step} stop ${end} post no
write_restart   restart.equilib2
if              "$(step)<${end}" then "jump lammps_script lab3"
unfix           f3

# Output final RDF and per-atom energy
fix             rdf_all_out all ave/time 1 1 1 c_rdf_all[*] file rdf_all.out mode vector
fix             rdf_pair_out all ave/time 1 1 1 c_rdf_pairs[*] file rdf_pairs.out mode vector
run             0  pre yes post no
write_dump      all xyz sys_0GPa.xyz modify sort id element {elements_str}
unfix           rdf_all_out
unfix           rdf_pair_out

# Output final per-atom energy and restart files
fix             fpe all ave/atom 1 1 1 c_c2
run             0 pre yes post no
write_data      data.msr
write_restart   restart.msr
write_dump      all custom dump.msr id type x y z vx vy vz f_fpe
""".replace("{elements_str}", present_elements_str)

	# Save the final LAMMPS script to file
	filename = f'results/{index}_{rep}/lammps_script'
	with open(filename, 'w') as file:
		file.write(script)

	return filename

# Create output directories and generate scripts for each row in the CSV
file_names = []
directory = "results"
if not os.path.exists(directory):
	os.makedirs(directory)

for idx, row in df.iterrows():
	for rep in range(1):
		directory = f"results/{idx}_{rep}"
		if not os.path.exists(directory):
			os.makedirs(directory)
		file_name = generate_lammps_script_with_all_elements(row, idx, rep)
		file_names.append(file_name)
