#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 15:32:24 2023

@author: ppxmg2
"""
# Write and run parameter files for BlackHawk and Isatis.

import numpy as np
import os

# Astrophysical parameter values
r_0 = 8.5                           # distance Sun-GC in kpc
rho_c_halo = 8.5e-25 	            # characteristic halo density in g/cm^3
r_c_halo = 17						# characteristic halo radius in kpc
gamma_halo = 1						# density profile inner slope

LN_bool = False
SLN_bool = False
CC3_bool = True

# Choose minimum energy as the lower range constrained by the Galactic Centre photon flux measured by INTEGRAL, COMPTEL, EGRET and Fermi-LAT (see e.g. Fig. 2 of 2201.01265)
E_min = 1e-6
E_max = 5  # maximum energy available in Hazma tables

# Number of energies to use
E_number = 500

# Number of black holes to include per run
BH_number = 1

# Range of PBH masses
m_pbh_values = np.logspace(11, 21, 1000)

# Path to BlackHawk and Isatis
BlackHawk_path = os.path.expanduser('~') + "/Downloads/version_finale/"
Isatis_path = BlackHawk_path + "scripts/Isatis/"

# Load default Isatis parameters file
params_Isatis = np.genfromtxt(Isatis_path + "parameters.txt", dtype=str, delimiter=" = ")
# Load default BlackHawk parameters file
params_BlackHawk = np.genfromtxt(BlackHawk_path + "parameters.txt", dtype=str, delimiter=" = ")
                
# Write your input when running BlackHawk to a file
with open(BlackHawk_path + "input.txt", "w") as f:
    f.write("y\ny")

# Update astrophysical parameter values used in Isatis
params_Isatis[2][1] = "0"        
params_Isatis[5][1] = "{:.1f}".format(r_0)     
params_Isatis[7][1] = "{:.1e}".format(rho_c_halo)     
params_Isatis[8][1] = "{:.0f}".format(r_c_halo)
params_Isatis[9][1] = "{:.0f}".format(gamma_halo)     
params_Isatis[14][1] = "1"   

E_min_full_range, E_max_full_range = 1e-6, 105.874        
                
energies = np.logspace(np.log10(E_min), np.log10(E_max), E_number)

fname_base = "mono_E500"
os.chdir(os.path.expanduser('~') + "/Asteroid_mass_gap/v2")
# Save Isatis parameters file.
params_Isatis[0][1] = fname_base
filename_params_Isatis = "params_%s.txt" % fname_base
np.savetxt(Isatis_path + filename_params_Isatis, params_Isatis, fmt="%s", delimiter = " = ")

# Boolean controls whether to use PYTHIA or Hazma tables
PYTHIA = False
Hazma = not PYTHIA
    
if PYTHIA:
    case = 2
elif Hazma:
    case = 3
                                                                                    
# Create runs file
runs_filename = "runs_%s.txt" % fname_base
runs_file_content = []
runs_file_content.append("nb_runs = {:.0f}".format(len(m_pbh_values)))
runs_file_content.append("")
 
for j, m in enumerate(m_pbh_values):
    
    destination_folder = fname_base + "_{:.0f}".format(j)
        
    filename_BlackHawk = "/BH_launcher/" + destination_folder + ".txt"
    # Add run name to runs file
    runs_file_content.append(destination_folder)
                        
    params_BlackHawk[0][1] = destination_folder
    params_BlackHawk[4][1] = "{:.0f}".format(BH_number)
    params_BlackHawk[5][1] = "{:.5e}".format(m)
    params_BlackHawk[6][1] = "{:.5e}".format(m)
    params_BlackHawk[15][1] = "1"
    params_BlackHawk[20][1] = "{:.5e}".format(m)
    params_BlackHawk[34][1] = "{:.0f}".format(E_number)
    params_BlackHawk[35][1] = "{:.5e}".format(E_min)
    params_BlackHawk[36][1] = "{:.5e}".format(E_max)
    params_BlackHawk[-1][1] = "{:.0f}".format(case)
                                        
    # Save BlackHawk parameters file.
    np.savetxt(Isatis_path + filename_BlackHawk, params_BlackHawk, fmt="%s", delimiter = " = ")
    
    # Run BlackHawk
    os.chdir(BlackHawk_path)
    command = "./BlackHawk_inst.x " + "scripts/Isatis/BH_launcher/" + destination_folder + ".txt<input.txt"
    os.system(command) 
   
# Save runs file
np.savetxt(BlackHawk_path + "scripts/Isatis/BH_launcher/%s" % runs_filename, runs_file_content, fmt="%s")

# Run Isatis
os.chdir(BlackHawk_path + "./scripts/Isatis")
command = "./Isatis.x %s ./BH_launcher/%s" % (filename_params_Isatis, runs_filename)
os.system(command)