#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 13:22:02 2023

@author: ppxmg2
"""
# Program for sanity checks on convergence test results. Includes plots 
# comparing mass function used by BlackHawk to the expected result, given a
# cutoff.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from preliminaries import load_data
from isatis_reproduction import read_blackhawk_spectra
import os

# Specify the plot style
mpl.rcParams.update({'font.size': 24, 'font.family':'serif'})
mpl.rcParams['xtick.major.size'] = 7
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.major.size'] = 7
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['ytick.minor.width'] = 1
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['xtick.top'] = False
mpl.rcParams['ytick.right'] = False
mpl.rcParams['font.family'] = 'serif'
mpl.rc('text', usetex=True)
mpl.rcParams['legend.edgecolor'] = 'lightgrey'

plt.style.use('tableau-colorblind10')


#%% Compare spectrum from a BH of temperature = 1 GeV (mass = 1.06e13 g). from
# BlackHawk to that shown in Fig. 1 of 0912.5297.

# Path to BlackHawk
BlackHawk_path = os.path.expanduser('~') + "/Downloads/version_finale/"
file_path_BlackHawk_data = BlackHawk_path + "results/"
        
fig, ax = plt.subplots(figsize=(7, 6))

# Results from Fig. 1 of 0912.5297
energies_primary, spectrum_primary = load_data("./0912.5297/0912.5297_Fig1_1GeV_primary.csv")
energies_tot, spectrum_tot = load_data("./0912.5297/0912.5297_Fig1_1GeV_secondary.csv")
ax.plot(energies_primary, spectrum_primary, color="r")
ax.plot(energies_tot, spectrum_tot, linestyle="dotted", color="orange", linewidth=5)

# Results from BlackHawk
energies_primary, spectrum_primary = read_blackhawk_spectra(file_path_BlackHawk_data + "Carr+10_test/instantaneous_primary_spectra.txt")
energies_tot, spectrum_tot = read_blackhawk_spectra(file_path_BlackHawk_data + "Carr+10_test/instantaneous_secondary_spectra.txt")
ax.plot(energies_primary, spectrum_primary, linestyle="dotted", color="k")
ax.plot(energies_tot, spectrum_tot, linestyle="dotted", color="k")

ax.set_xlabel("$E~[\mathrm{GeV}]$")
ax.set_ylabel("$\mathrm{d}^2N_\gamma / (\mathrm{d}t \mathrm{d}E_\gamma)~[\mathrm{GeV}^{-1}\mathrm{s}^{-1}]$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(5e-1, 12.5)
ax.set_ylim(1e19, 1e25)
ax.legend(fontsize="small")
fig.tight_layout()

