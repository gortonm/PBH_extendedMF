#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 15:32:24 2023

@author: ppxmg2
"""
# Write and run parameter files for BlackHawk and Isatis.

import numpy as np
import os
from preliminaries import SLN, CC3

# Astrophysical parameter values
local_DM_GeV = 0.47685              # DM local density in GeV/cm^3
r_0 = 8.5                           # distance Sun-GC in kpc
rho_c_halo = 8.5e-25 	            # characteristic halo density in g/cm^3
r_c_halo = 17						# characteristic halo radius in kpc
gamma_halo = 1						# density profile inner slope

log_normal = True
SLN_bool = False
CC3_bool = False

# Minimum and maximum central masses
mc_min = 1e14
mc_max = 1e19

# Load mass function parameters.
[Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)

# Load minimum and maximum masses 
[Deltas, m_lower_LN, m_upper_LN, m_lower_SLN, m_upper_SLN, m_lower_CC3, m_upper_CC3] = np.genfromtxt("MF_scaled_mass_ranges.txt", delimiter="\t\t ", skip_header=2, unpack=True)

# Path to BlackHawk and Isatis
BlackHawk_path = os.path.expanduser('~') + "/Downloads/version_finale/"
Isatis_path = BlackHawk_path + "scripts/Isatis/"

# Load default Isatis parameters file
parameters_Isatis = np.genfromtxt(Isatis_path + "parameters.txt", dtype=str, delimiter=" = ")
# Load default BlackHawk parameters file
parameters_BlackHawk = np.genfromtxt(BlackHawk_path + "parameters.txt", dtype=str, delimiter=" = ")

mc_values = np.logspace(np.log10(mc_min), np.log10(mc_max), 100)
BH_number = 10000

# Choose minimum energy as the lower range constrained by the Galactic Centre photon flux measured by INTEGRAL, COMPTEL, EGRET and Fermi-LAT (see e.g. Fig. 2 of 2201.01265)
E_min = 1e-5
E_max = 5   # maximum energy available in Hazma tables
E_number = 1000

spec_file_initial_line = "mass/spin \t 0.00000e+00"

# Write your input when running BlackHawk to a file
with open(BlackHawk_path + "input.txt", "w") as f:
    f.write("y\ny")
    
# Update parameter file appearing in Isatis
parameters_Isatis[2][1] = "0\t\t\t\t\t\t\t\t\t\t"        
parameters_Isatis[3][1] = "{:.2f}\t\t\t\t\t\t\t\t\t\t".format(local_DM_GeV)     
parameters_Isatis[5][1] = "{:.2f}\t\t\t\t\t\t\t\t\t\t".format(r_0)     
parameters_Isatis[7][1] = "{:.2e}\t\t\t\t\t\t\t\t\t\t".format(rho_c_halo)     
parameters_Isatis[8][1] = "{:.1f}\t\t\t\t\t\t\t\t\t\t".format(r_c_halo)     
parameters_Isatis[9][1] = "{:.2f}\t\t\t\t\t\t\t\t\t\t".format(gamma_halo)     
parameters_Isatis[14][1] = "1\t\t\t\t\t\t\t\t\t\t"   
    
for i in range(len(Deltas)):
      
    if log_normal:
        append = "GC_LN_Delta={:.1f}".format(Deltas[i])
    elif SLN_bool:
        append = "GC_SLN_Delta={:.1f}".format(Deltas[i])
    elif CC3_bool:
        append = "GC_CC3_Delta={:.1f}".format(Deltas[i])
    
    # Save Isatis parameters file.
    parameters_Isatis[0][1] = append + "\t\t\t\t\t\t\t\t\t\t"
    filename_parameters_Isatis = "parameters_GC_LN_Delta={:.1f}.txt".format(Deltas[i])
    np.savetxt(Isatis_path + filename_parameters_Isatis, parameters_Isatis, fmt="%s", delimiter = " = ")

    for j, m_c in enumerate(mc_values):
        
        spec_file = []
        spec_file.append(spec_file_initial_line)
        filename_BH_spec = BlackHawk_path + "/src/tables/users_spectra/" + append + "_{:.0f}.txt".format(j)

        if log_normal:
            append = "GC_LN_Delta={:.1f}".format(Deltas[i])

        if SLN_bool:
            append = "GC_SLN_Delta={:.1f}".format(Deltas[i])
            # Create and save file for PBH mass and spin distribution
            m_pbh_values = np.logspace(np.log10(m_lower_SLN[i] * m_c), m_upper_SLN[i] * m_c, BH_number)
            spec_values = SLN(m_pbh_values, m_c=m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i])
            for k in range(len(m_pbh_values)):
                spec_file.append("{:.5e}\t{:.5e}".format(m_pbh_values[k], spec_values[k]))
            np.savetxt(filename_BH_spec, spec_file, fmt="%s", delimiter = " = ")
                
        if CC3_bool:
            append = "GC_CC3_Delta={:.1f}".format(Deltas[i])
            # Create and save file for PBH mass and spin distribution
            m_pbh_values = np.logspace(np.log10(m_lower_CC3[i] * m_c), m_upper_CC3[i] * m_c, BH_number)
            spec_values = CC3(m_pbh_values, m_c, alphas_CC3[i], betas[i])
            for k in range(len(m_pbh_values)):
                spec_file.append("{:.5e}\t{:.5e}".format(m_pbh_values[k], spec_values[k]))
            np.savetxt(filename_BH_spec, spec_file, fmt="%s", delimiter = " = ")


        destination_folder = append + "_{:.0f}".format(j)
        filename_BlackHawk = "/BH_launcher/" + destination_folder + ".txt"
        
        filepath_Isatis = BlackHawk_path + "results/" + destination_folder                            
        if not os.path.exists(filepath_Isatis):
            os.makedirs(filepath_Isatis)
    
        parameters_BlackHawk[0][1] = destination_folder + "\t\t\t\t\t\t\t\t\t\t"
        parameters_BlackHawk[4][1] = "{:.0f}\t\t\t\t\t\t\t\t\t\t".format(BH_number)
        parameters_BlackHawk[15][1] = "1\t\t\t\t\t\t\t\t\t\t"
        if log_normal:
            parameters_BlackHawk[5][1] = "{:.5e}\t\t\t\t\t\t\t\t\t\t".format(m_lower_LN[i] * m_c)
            parameters_BlackHawk[6][1] = "{:.5e}\t\t\t\t\t\t\t\t\t\t".format(m_upper_LN[i] * m_c)
            parameters_BlackHawk[19][1] = "{:.3f}\t\t\t\t\t\t\t\t\t\t".format(sigmas_LN[i])
            parameters_BlackHawk[20][1] = "{:.5e}\t\t\t\t\t\t\t\t\t\t".format(m_c)
        if SLN_bool:
            parameters_BlackHawk[5][1] = "{:.5e}\t\t\t\t\t\t\t\t\t\t".format(m_lower_SLN[i] * m_c)
            parameters_BlackHawk[6][1] = "{:.5e}\t\t\t\t\t\t\t\t\t\t".format(m_upper_SLN[i] * m_c)
            parameters_BlackHawk[28][1] = append + "_{:.0f}.txt".format(j)
        if CC3_bool:
            parameters_BlackHawk[5][1] = "{:.5e}\t\t\t\t\t\t\t\t\t\t".format(m_lower_CC3[i] * m_c)
            parameters_BlackHawk[6][1] = "{:.5e}\t\t\t\t\t\t\t\t\t\t".format(m_upper_CC3[i] * m_c)
            parameters_BlackHawk[28][1] = append + "_{:.0f}.txt".format(j)
        parameters_BlackHawk[34][1] = "{:.0f}\t\t\t\t\t\t\t\t\t\t".format(E_number)
        parameters_BlackHawk[35][1] = "{:.5e}\t\t\t\t\t\t\t\t\t\t".format(E_min)
        parameters_BlackHawk[36][1] = "{:.5e}\t\t\t\t\t\t\t\t\t\t".format(E_max)
        parameters_BlackHawk[-1][1] = "3\t\t\t\t\t\t\t\t\t\t"
        
        # Save BlackHawk parameters file.
        np.savetxt(Isatis_path + filename_BlackHawk, parameters_BlackHawk, fmt="%s", delimiter = " = ")
        
        # Run BlackHawk
        os.chdir(BlackHawk_path)
        command = "./BlackHawk_inst.x " + "scripts/Isatis/BH_launcher/" + destination_folder + ".txt<input.txt"
        os.system(command)
            
    # Run Isatis
    os.chdir("./scripts/Isatis")
    command = "./Isatis.x %s %s" % (filename_parameters_Isatis, filename_BlackHawk)
    os.system(command)