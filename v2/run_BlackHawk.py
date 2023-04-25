#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 15:32:24 2023

@author: ppxmg2
"""

# Run BlackHawk, using the results from BH_launcher.
import numpy as np
import os

# Path to BlackHawk
BlackHawk_path = os.path.expanduser('~') + "/Downloads/version_finale/"
os.chdir(BlackHawk_path)

# If True, calculate total time-evolved BlackHawk spectrum.
tot = True    
# If True, calculate instantaneous BlackHawk spectrum.
inst = not tot

# Write your input when running BlackHawk to a file
with open(BlackHawk_path + "input.txt", "w") as f:
    f.write("y\ny")

m_pbh_values = np.logspace(np.log10(4e14), 16, 50)
fname_base = "mass_evolution"

for j in range(len(m_pbh_values)):
    
    destination_folder = fname_base + "_{:.0f}".format(j+1)
    filename_BlackHawk = "/BH_launcher/" + destination_folder + ".txt"
                            
    # Run BlackHawk
    os.chdir(BlackHawk_path)
    if tot:
        command = "./BlackHawk_tot.x " + "scripts/Isatis/BH_launcher/" + destination_folder + ".txt<input.txt"
    elif inst:
        command = "./BlackHawk_tot.x " + "scripts/Isatis/BH_launcher/" + destination_folder + ".txt<input.txt"
    os.system(command)