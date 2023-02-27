#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 15:32:24 2023

@author: ppxmg2
"""
# Write a parameters files for Isatis

import numpy as np
import os

log_normal = False
SLN_bool = True
CC3_bool = False

# minimum and maximum central masses
mc_min = 1e14
mc_max = 1e19

# path to BlackHawk
BlackHawk_path = os.path.expanduser('~') + "/Downloads/version_finale/"

# load default Isatis parameters file
parameters = np.genfromtxt(BlackHawk_path + "parameters.txt", dtype=str, delimiter=" = ")

m_pbh_values = 10**np.linspace(11, 21.05, 10000)
mc_values = 10**np.arange(np.log10(mc_min), np.log10(mc_max), 0.1)

BH_number = len(m_pbh_values)
E_number = 1000
E_min = 1e-5
E_max = 5   # maximum energy available in Hazma tables


# Write your input to a file
with open(BlackHawk_path + "input.txt", "w") as f:
    f.write("y\ny")

#%% Log-normal mass function.
if log_normal:
    sigmas_LN = np.array([0.374, 0.377, 0.395, 0.430, 0.553, 0.864, 0.01])

    for sigma in sigmas_LN:
        number_of_sd = 4   # number of standard deviations away from median when calculating minimum and maximum PBH mass
        
        # width of log-normal mass function
        append = "LN_sigma={:.3f}".format(sigma)
                
        # name of runs file
        runs_filename = BlackHawk_path + "scripts/Isatis/BH_launcher/runs_GC_" + append
            
        # choose minimum and maximum energies to cover the range constrained by the
        # Galactic Centre photon flux measured by INTEGRAL, COMPTEL, EGRET and Fermi-
        # LAT (see e.g. Fig. 2 of 2201.01265)
        
        for i in range(len(parameters)):
            print(i, parameters[i])
            
        runs_file_content = []
        runs_file_content.append("nb_runs = {:.0f}".format(len(mc_values)))
        runs_file_content.append("")
    
        
        for i, mc in enumerate(mc_values):
                
            print(mc)
        
            M_min = 10**(np.log10(mc) - number_of_sd * sigma)  # n standard deviations below log_10 of the median value
            M_max = 10**(np.log10(mc) + number_of_sd * sigma)  # n standard deviations above log_10 of the median value
            
            if log_normal:
                destination_folder = "GC_LN_sigma={:.3f}_{:.0f}".format(sigma, i)
                # name of parameters file for running BlackHawk
                filename_BlackHawk = BlackHawk_path + "scripts/Isatis/BH_launcher/GC_LN_sigma={:.3f}_{:.0f}.txt".format(sigma, i)
                # name parameters name for running Isatis (in results folder)
                            
            filepath_Isatis = BlackHawk_path + "results/" + destination_folder
            if not os.path.exists(filepath_Isatis):
                os.makedirs(filepath_Isatis)
            filename_Isatis = filepath_Isatis + "/" + destination_folder + ".txt"
        
            parameters[0][1] = destination_folder + "\t\t\t\t\t\t\t\t\t\t"
            parameters[4][1] = "{:.0f}\t\t\t\t\t\t\t\t\t\t".format(BH_number)
            parameters[5][1] = "{:.5e}\t\t\t\t\t\t\t\t\t\t".format(M_min)
            parameters[6][1] = "{:.5e}\t\t\t\t\t\t\t\t\t\t".format(M_max)
            parameters[15][1] = "1\t\t\t\t\t\t\t\t\t\t"
            parameters[19][1] = "{:.3f}\t\t\t\t\t\t\t\t\t\t".format(sigma)
            parameters[20][1] = "{:.5e}\t\t\t\t\t\t\t\t\t\t".format(mc)
            parameters[34][1] = "{:.0f}\t\t\t\t\t\t\t\t\t\t".format(E_number)
            parameters[35][1] = "{:.5e}\t\t\t\t\t\t\t\t\t\t".format(E_min)
            parameters[36][1] = "{:.5e}\t\t\t\t\t\t\t\t\t\t".format(E_max)
            parameters[-1][1] = "3\t\t\t\t\t\t\t\t\t\t"
        
            runs_file_content.append("GC_LN_sigma={:.3f}_{:.0f}".format(sigma, i))
            np.savetxt(filename_BlackHawk, parameters, fmt="%s", delimiter = " = ")
            np.savetxt(filename_Isatis, parameters, fmt="%s", delimiter = " = ")
            
            # run BlackHawk
            os.chdir(BlackHawk_path)
            command = "./BlackHawk_inst.x " + "scripts/Isatis/BH_launcher/GC_LN_sigma={:.3f}_{:.0f}.txt<input.txt".format(sigma, i)
            os.system(command)
            
        np.savetxt(runs_filename, runs_file_content, fmt="%s")
        
        # run Isatis
        os.chdir("./scripts/Isatis")
        command = "./Isatis.x parameters_GC_LN_sigma={:.3f}.txt ./BH_launcher/runs_GC_LN_sigma={:.3f}".format(sigma, sigma)
        os.system(command)


#%%  Skew-lognormal and CC3 mass functions.

from constraints_extended_MF import skew_LN, CC3

# Mass function parameter values, from 2009.03204.
Deltas = np.array([0., 0.1, 0.3, 0.5, 1.0, 2.0, 5.0])
sigmas = np.array([0.55, 0.55, 0.57, 0.60, 0.71, 0.97, 2.77])
alphas_SL = np.array([-2.27, -2.24, -2.07, -1.82, -1.31, -0.66, 1.39])
 
alphas_CC = np.array([3.06, 3.09, 3.34, 3.82, 5.76, 18.9, 13.9])
betas = np.array([2.12, 2.08, 1.72, 1.27, 0.51, 0.0669, 0.0206])

file_initial_line = "mass/spin \t 0.00000e+00"
os.chdir(BlackHawk_path)

for i, Delta in enumerate(Deltas):
    
    runs_file_content = []
    runs_file_content.append("nb_runs = {:.0f}".format(len(mc_values)))
    runs_file_content.append("")

    if SLN_bool:
        append = "GC_SLN_Delta={:.1f}".format(Delta)

    elif CC3_bool:
        append = "GC_CC3_Delta={:.1f}".format(Delta)
        
    # name of runs file
    runs_filename = BlackHawk_path + "scripts/Isatis/BH_launcher/runs_" + append
        
    for j, m_c in enumerate(mc_values):
        
        file = []
        file.append(file_initial_line)
        filename_BH_spec = BlackHawk_path + "/src/tables/users_spectra/" + append + "_{:.0f}.txt".format(j)
        print(filename_BH_spec)
        
        if SLN_bool:
            spec_values = skew_LN(m_pbh_values, m_c=m_c, sigma=sigmas[i], alpha=alphas_SL[i])
        elif CC3_bool:
            spec_values = CC3(m_pbh_values, m_c, alphas_CC[i], betas[i])

        for k in range(len(m_pbh_values)):
            file.append("{:.5e}\t{:.5e}".format(m_pbh_values[k], spec_values[k]))
            
        np.savetxt(filename_BH_spec, file, fmt="%s", delimiter = " = ")
            
        destination_folder = append + "_{:.0f}".format(j)
        filename_BlackHawk = BlackHawk_path + "scripts/Isatis/BH_launcher/" + append + "_{:.0f}.txt".format(j)
        
        filepath_Isatis = BlackHawk_path + "results/" + destination_folder
        if not os.path.exists(filepath_Isatis):
            os.makedirs(filepath_Isatis)
        filename_Isatis = filepath_Isatis + "/" + destination_folder + ".txt"
    
        parameters[0][1] = destination_folder + "\t\t\t\t\t\t\t\t\t\t"
        parameters[4][1] = "{:.0f}\t\t\t\t\t\t\t\t\t\t".format(BH_number)
        parameters[5][1] = "{:.5e}\t\t\t\t\t\t\t\t\t\t".format(min(m_pbh_values))
        parameters[6][1] = "{:.5e}\t\t\t\t\t\t\t\t\t\t".format(max(m_pbh_values))
        parameters[15][1] = "-1\t\t\t\t\t\t\t\t\t\t"
        parameters[28][1] = append + "_{:.0f}.txt".format(j)
        parameters[34][1] = "{:.0f}\t\t\t\t\t\t\t\t\t\t".format(E_number)
        parameters[35][1] = "{:.5e}\t\t\t\t\t\t\t\t\t\t".format(E_min)
        parameters[36][1] = "{:.5e}\t\t\t\t\t\t\t\t\t\t".format(E_max)
        parameters[-1][1] = "3\t\t\t\t\t\t\t\t\t\t"
        
        runs_file_content.append(append +"_{:.0f}".format(j))
        np.savetxt(filename_BlackHawk, parameters, fmt="%s", delimiter = " = ")
        np.savetxt(filename_Isatis, parameters, fmt="%s", delimiter = " = ")
                
        # run BlackHawk
        os.chdir(BlackHawk_path)
        command = "./BlackHawk_inst.x " + "scripts/Isatis/BH_launcher/" + append + "_{:.0f}.txt<input.txt".format(j)
        os.system(command)
        
    np.savetxt(runs_filename, runs_file_content, fmt="%s")
    
    # run Isatis
    os.chdir("./scripts/Isatis")
    command = "./Isatis.x parameters_%s.txt ./BH_launcher/runs_%s" % (append, append)
    os.system(command)
