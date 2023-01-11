#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 15:32:24 2023

@author: ppxmg2
"""
# Write a parameters files for Isatis

import numpy as np
import os

log_normal = True

if log_normal:
    # minimum and maximum central masses for a log-normal mass function
    mu_min = 1e14
    mu_max = 1e19

    mu_pbh_values = 10**np.arange(np.log10(mu_min), np.log10(mu_max), 1)
    number_of_sd = 4   # number of standard deviations away from median when calculating minimum and maximum PBH mass

    # width of log-normal mass function
    sigma = 0.5

BH_number = 10000
E_number = 1000
E_min = 1e-5
E_max = 5   # maximum energy available in Hazma tables

# path to BlackHawk
BlackHawk_path = os.path.expanduser('~') + "/Downloads/version_finale/"

# load default Isatis parameters file
parameters = np.genfromtxt(BlackHawk_path + "parameters.txt", dtype=str, delimiter=" = ")

# name of runs file
runs_filename = BlackHawk_path + "scripts/Isatis/BH_launcher/runs_GC_LN_sigma={:.1f}".format(sigma)
# contents of runs file
runs_file_content = []
runs_file_content.append("nb_runs = {:.0f}".format(len(mu_pbh_values)))
runs_file_content.append("")

# choose minimum and maximum energies to cover the range constrained by the
# Galactic Centre photon flux measured by INTEGRAL, COMPTEL, EGRET and Fermi-
# LAT (see e.g. Fig. 2 of 2201.01265)

for i in range(len(parameters)):
    print(i, parameters[i])

for i, mu in enumerate(mu_pbh_values):

    print(mu)

    if log_normal:
        M_min = 10**(np.log10(mu) - number_of_sd * sigma)  # n standard deviations below log_10 of the median value
        M_max = 10**(np.log10(mu) + number_of_sd * sigma)  # n standard deviations above log_10 of the median value

    destination_folder = "GC_LN_sigma={:.1f}_{:.0f}".format(sigma, i)
    # name of parameters file for running BlackHawk
    filename_BlackHawk = BlackHawk_path + "scripts/Isatis/BH_launcher/GC_LN_sigma={:.1f}_{:.0f}.txt".format(sigma, i)
    # name parameters name for running Isatis (in results folder)
    filepath_Isatis = BlackHawk_path + "results/" + destination_folder
    if not os.path.exists(filepath_Isatis):
        os.makedirs(filepath_Isatis)
    filename_Isatis = filepath_Isatis + "/" + destination_folder + ".txt"

    parameters[0][1] = destination_folder + "\t\t\t\t\t\t\t\t\t\t"
    parameters[4][1] = "{:.0f}\t\t\t\t\t\t\t\t\t\t".format(BH_number)

    if log_normal:
        parameters[5][1] = "{:.5e}\t\t\t\t\t\t\t\t\t\t".format(M_min)
        parameters[6][1] = "{:.5e}\t\t\t\t\t\t\t\t\t\t".format(M_max)
        parameters[15][1] = "1\t\t\t\t\t\t\t\t\t\t"
        parameters[19][1] = "{:.1f}\t\t\t\t\t\t\t\t\t\t".format(sigma)
        parameters[20][1] = "{:.5e}\t\t\t\t\t\t\t\t\t\t".format(mu)

    parameters[34][1] = "{:.0f}\t\t\t\t\t\t\t\t\t\t".format(E_number)
    parameters[35][1] = "{:.5e}\t\t\t\t\t\t\t\t\t\t".format(E_min)
    parameters[36][1] = "{:.5e}\t\t\t\t\t\t\t\t\t\t".format(E_max)
    parameters[-1][1] = "3\t\t\t\t\t\t\t\t\t\t"

    runs_file_content.append("GC_LN_sigma={:.1f}_{:.0f}".format(sigma, i))
    np.savetxt(filename_BlackHawk, parameters, fmt="%s", delimiter = " = ")
    np.savetxt(filename_Isatis, parameters, fmt="%s", delimiter = " = ")
    
    # run BlackHawk
    os.chdir(BlackHawk_path)
    command = "./BlackHawk_inst.x " + "scripts/Isatis/BH_launcher/GC_LN_sigma={:.1f}_{:.0f}.txt".format(sigma, i)
    os.system(command)

np.savetxt(runs_filename, runs_file_content, fmt="%s")