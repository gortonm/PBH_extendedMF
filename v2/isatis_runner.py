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
r_0 = 8.5                           # distance Sun-GC in kpc
rho_c_halo = 8.5e-25 	            # characteristic halo density in g/cm^3
r_c_halo = 17						# characteristic halo radius in kpc
gamma_halo = 1						# density profile inner slope

LN_bool = True
SLN_bool = False
CC3_bool = False


# Load mass function parameters.
[Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)


# Controls whether to use a range of PBH masses that matches those used
# in isatis_reproduction.py. Use for comparing to the results obtained using 
# the method from 1705.05567.
test_mass_range = False
if test_mass_range:
    m_lower_test, m_upper_test = 1e11, 1e21


# Load minimum and maximum scaled masses for which the MF is above a cutoff.
# Select which range of masses to use (for convergence tests).

# If True, use cutoff in terms of the mass function scaled to its peak value.
MF_cutoff = True
# If True, use cutoff in te\wrms of the integrand appearing in Galactic Centre photon constraints.
integrand_cutoff = False
# If True, use cutoff in terms of the integrand appearing in Galactic Centre photon constraints, with the mass function evolved to the present day.
integrand_cutoff_present = False


# Mass range

cutoff = 1e-4
scaled_masses_filename = "MF_scaled_mass_ranges_c={:.0f}.txt".format(-np.log10(cutoff))
[Deltas, m_lower_LN, m_upper_LN, m_lower_SLN, m_upper_SLN, m_lower_CC3, m_upper_CC3] = np.genfromtxt(scaled_masses_filename, delimiter="\t\t ", skip_header=2, unpack=True)

if LN_bool:
    Deltas = Deltas[:-1]

# Minimum and maximum central masses.

# If True, use a single characteristic PBH mass. 
single_mass = True
mc_min = 1e14
mc_max = 1e19
if single_mass:
    mc_values = [mc_max]
else:
    mc_values = np.logspace(np.log10(mc_min), np.log10(mc_max), 100)

# PBH mass spacing, in log10(PBH mass / grams)
delta_log_m = 1e-2

# Path to BlackHawk and Isatis
BlackHawk_path = os.path.expanduser('~') + "/Downloads/version_finale/"
Isatis_path = BlackHawk_path + "scripts/Isatis/"

# Load default Isatis parameters file
params_Isatis = np.genfromtxt(Isatis_path + "parameters.txt", dtype=str, delimiter=" = ")
# Load default BlackHawk parameters file
params_BlackHawk = np.genfromtxt(BlackHawk_path + "parameters.txt", dtype=str, delimiter=" = ")

# Choose minimum energy as the lower range constrained by the Galactic Centre photon flux measured by INTEGRAL, COMPTEL, EGRET and Fermi-LAT (see e.g. Fig. 2 of 2201.01265)
E_min = 1e-5
E_max = 5   # maximum energy available in Hazma tables
E_number = 500

spec_file_initial_line = "mass/spin \t 0.00000e+00"

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
    
for i in range(len(Deltas)):
    
    if LN_bool:
        fname_base = "LN_D={:.1f}_dm={:.0f}".format(Deltas[i], -np.log10(delta_log_m))
    elif SLN_bool:
        fname_base = "SL_D={:.1f}_dm={:.0f}".format(Deltas[i], -np.log10(delta_log_m))
    elif CC3_bool:
        fname_base = "CC_D={:.1f}_dm={:.0f}".format(Deltas[i], -np.log10(delta_log_m))
    
    # Indicates which range of masses are being used (for convergence tests).
    if test_mass_range:
        fname_base += "_test_range"
    elif MF_cutoff:
        fname_base += "_c={:.0f}".format(-np.log10(cutoff))
        
    if single_mass:
        fname_base += "_mc={:.0f}".format(np.log10(mc_max))
                
    # Create runs file
    runs_filename = "runs_%s.txt" % fname_base
    runs_file_content = []
    runs_file_content.append("nb_runs = {:.0f}".format(len(mc_values)))
    runs_file_content.append("")

    # Save Isatis parameters file.
    params_Isatis[0][1] = fname_base
    filename_params_Isatis = "params_%s.txt" % fname_base
    np.savetxt(Isatis_path + filename_params_Isatis, params_Isatis, fmt="%s", delimiter = " = ")
    
    for j, m_c in enumerate(mc_values):
        
        if single_mass:
            destination_folder = fname_base
        else:
            destination_folder = fname_base + "_{:.0f}".format(j)
            
        filename_BlackHawk = "/BH_launcher/" + destination_folder + ".txt"
        filepath_Isatis = BlackHawk_path + "results/" + destination_folder
        # Add run name to runs file
        runs_file_content.append(destination_folder)
                            
        if not os.path.exists(filepath_Isatis):
            os.makedirs(filepath_Isatis)
    
        params_BlackHawk[0][1] = destination_folder
        
        if LN_bool:
            BH_number = int((np.log10(m_c*m_upper_LN[i])-np.log10(m_c*m_lower_LN[i])) / delta_log_m)
            params_BlackHawk[4][1] = "{:.0f}".format(BH_number)
            params_BlackHawk[5][1] = "{:.5e}".format(m_lower_LN[i] * m_c)
            params_BlackHawk[6][1] = "{:.5e}".format(m_upper_LN[i] * m_c)
            params_BlackHawk[15][1] = "1"
            params_BlackHawk[19][1] = "{:.3f}".format(sigmas_LN[i])
            params_BlackHawk[20][1] = "{:.5e}".format(m_c)
            
        if SLN_bool:
            BH_number = int((np.log10(m_c*m_upper_SLN[i])-np.log10(m_c*m_lower_SLN[i])) / delta_log_m)
            params_BlackHawk[4][1] = "{:.0f}".format(BH_number)
            params_BlackHawk[5][1] = "{:.5e}".format(m_lower_SLN[i] * m_c)
            params_BlackHawk[6][1] = "{:.5e}".format(m_upper_SLN[i] * m_c)
            params_BlackHawk[15][1] = "-1"
            params_BlackHawk[28][1] = destination_folder
            
            # Create and save file for PBH mass and spin distribution
            spec_file = []
            spec_file.append(spec_file_initial_line)
            filename_BH_spec = BlackHawk_path + "/src/tables/users_spectra/" + destination_folder

            m_pbh_values = np.logspace(np.log10(m_lower_SLN[i] * m_c), np.log10(m_upper_SLN[i] * m_c), BH_number)
            spec_values = SLN(m_pbh_values, m_c=m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i])         
            for k in range(len(m_pbh_values)):
                spec_file.append("{:.5e}\t{:.5e}".format(m_pbh_values[k], spec_values[k]))
            np.savetxt(filename_BH_spec, spec_file, fmt="%s", delimiter = " = ")            
            
        if CC3_bool:
            BH_number = int((np.log10(m_c*m_upper_CC3[i])-np.log10(m_c*m_lower_CC3[i])) / delta_log_m)
            params_BlackHawk[4][1] = "{:.0f}".format(BH_number)
            params_BlackHawk[5][1] = "{:.5e}".format(m_lower_CC3[i] * m_c)
            params_BlackHawk[6][1] = "{:.5e}".format(m_upper_CC3[i] * m_c)
            params_BlackHawk[15][1] = "-1"
            params_BlackHawk[28][1] = destination_folder
            
            # Create and save file for PBH mass and spin distribution
            spec_file = []
            spec_file.append(spec_file_initial_line)
            filename_BH_spec = BlackHawk_path + "/src/tables/users_spectra/" + destination_folder

            m_pbh_values = np.logspace(np.log10(m_lower_CC3[i] * m_c), np.log10(m_upper_CC3[i] * m_c), BH_number)
            spec_values = CC3(m_pbh_values, m_c, alphas_CC3[i], betas[i])
            for k in range(len(m_pbh_values)):
                spec_file.append("{:.5e}\t{:.5e}".format(m_pbh_values[k], spec_values[k]))
            np.savetxt(filename_BH_spec, spec_file, fmt="%s", delimiter = " = ")    
            
        print("BH_number = {:.0f}".format(BH_number))
            
        params_BlackHawk[34][1] = "{:.0f}".format(E_number)
        params_BlackHawk[35][1] = "{:.5e}".format(E_min)
        params_BlackHawk[36][1] = "{:.5e}".format(E_max)
        params_BlackHawk[-1][1] = "3"
        
        if test_mass_range:
            params_BlackHawk[5][1] = "{:.5e}".format(m_lower_test)
            params_BlackHawk[6][1] = "{:.5e}".format(m_upper_test)
                    
        # Save BlackHawk parameters file.
        np.savetxt(Isatis_path + filename_BlackHawk, params_BlackHawk, fmt="%s", delimiter = " = ")
        
        # Run BlackHawk
        os.chdir(BlackHawk_path)
        command = "./BlackHawk_inst.x " + "scripts/Isatis/BH_launcher/" + destination_folder + ".txt<input.txt"
        os.system(command)
            
    # Save runs file
    np.savetxt(BlackHawk_path + "scripts/Isatis/BH_launcher/%s" % runs_filename, runs_file_content, fmt="%s")
    
    # Run Isatis
    os.chdir("./scripts/Isatis")
    command = "./Isatis.x %s ./BH_launcher/%s" % (filename_params_Isatis, runs_filename)
    os.system(command)