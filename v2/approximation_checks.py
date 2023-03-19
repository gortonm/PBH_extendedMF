#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 11:55:35 2023

@author: ppxmg2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from preliminaries import LN, SLN, CC3, load_results_Isatis
from isatis_reproduction import read_blackhawk_spectra

# Produce plots of the Subaru-HSC microlensing constraints on PBHs, for
# extended mass functions, using the method from 1705.05567.

# Specify the plot style
mpl.rcParams.update({'font.size': 16,'font.family':'serif'})
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
filepath = './Extracted_files/'

#%% Understand at which mass range secondary emission of photons from PBHs becomes significant.

m_pbh_values = np.logspace(11, 21, 1000)

# Path to BlackHawk data files
file_path_BlackHawk_data = "./../../Downloads/version_finale/results/"

# Plot the primary and total photon spectrum at different BH masses.

# Choose indices to plot
indices = [201, 301, 401, 501, 601]
colors = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200']

fig, ax = plt.subplots(figsize=(8, 6.5))

for j, i in enumerate(indices):
    m_pbh = m_pbh_values[i]
    energies_primary, spectrum_primary = read_blackhawk_spectra(file_path_BlackHawk_data + "GC_mono_{:.0f}/".format(i) + "instantaneous_primary_spectra.txt", col=1)
    energies_tot, spectrum_tot = read_blackhawk_spectra(file_path_BlackHawk_data + "GC_mono_{:.0f}/".format(i) + "instantaneous_secondary_spectra.txt", col=1)
    
    ax.plot(energies_primary, spectrum_primary, linestyle="dotted", color=colors[j])
    ax.plot(energies_tot, spectrum_tot, color=colors[j], label="{:.0e}".format(m_pbh))
    
ax.legend(title="$m~[\mathrm{g}]$", fontsize="small")
ax.set_xlabel("$E~[\mathrm{GeV}]$")
ax.set_ylabel("$\mathrm{d}^2 N_\gamma/(\mathrm{d}t\mathrm{d}E)~[\mathrm{GeV^{-1} \cdot \mathrm{cm}^{-2} \cdot \mathrm{s}^{-1} \cdot \mathrm{sr}^{-1}}]$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e-5, 5)
ax.set_ylim(1e10, 1e30)


# Plot the integral of primary and total photon spectrum over energy.
integral_primary = []
integral_secondary = []

for i in range(len(m_pbh_values)):
    energies_primary, spectrum_primary = read_blackhawk_spectra(file_path_BlackHawk_data + "GC_mono_{:.0f}/".format(i+1) + "instantaneous_primary_spectra.txt", col=1)
    energies_tot, spectrum_tot = read_blackhawk_spectra(file_path_BlackHawk_data + "GC_mono_{:.0f}/".format(i+1) + "instantaneous_secondary_spectra.txt", col=1)
            
    integral_primary.append(np.trapz(spectrum_primary, energies_primary))
    integral_secondary.append(np.trapz(spectrum_tot, energies_tot))

fit_m_square = integral_primary[500] * np.power(m_pbh_values/m_pbh_values[500], -1)

fig, ax = plt.subplots(figsize=(8, 6.5))
ax.plot(m_pbh_values, integral_secondary, color=colors[0], label="Total")
ax.plot(m_pbh_values, integral_primary, linestyle="dashed", color=colors[0], label="Primary emission only")
ax.plot(m_pbh_values, fit_m_square, color=colors[1], linestyle="dotted", label="$m^{-1}$ fit")
ax.set_xlabel("$m~[\mathrm{g}]$")
ax.set_ylabel("$\mathrm{d} N_\gamma/\mathrm{d}t~[\mathrm{cm}^{-2} \cdot \mathrm{s}^{-1} \cdot \mathrm{sr}^{-1}]$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(fontsize="small")
ax.set_xlim(1e16, 1e19)
ax.set_ylim(2e15, 1e19)


# Plot the integral of energy * primary and total photon spectrum over energy.
integral_primary = []
integral_secondary = []

for i in range(len(m_pbh_values)):
    energies_primary, spectrum_primary = read_blackhawk_spectra(file_path_BlackHawk_data + "GC_mono_{:.0f}/".format(i+1) + "instantaneous_primary_spectra.txt", col=1)
    energies_tot, spectrum_tot = read_blackhawk_spectra(file_path_BlackHawk_data + "GC_mono_{:.0f}/".format(i+1) + "instantaneous_secondary_spectra.txt", col=1)
            
    integral_primary.append(np.trapz(spectrum_primary*energies_primary, energies_primary))
    integral_secondary.append(np.trapz(spectrum_tot*energies_tot, energies_tot))

fit_m_square = integral_primary[500] * np.power(m_pbh_values/m_pbh_values[500], -2)

fig, ax = plt.subplots(figsize=(8, 6.5))
ax.plot(m_pbh_values, integral_secondary, color=colors[0], label="Total")
ax.plot(m_pbh_values, integral_primary, linestyle="dashed", color=colors[0], label="Primary emission only")
ax.plot(m_pbh_values, fit_m_square, color=colors[1], linestyle="dotted", label="$m^{-2}$ fit")
ax.set_xlabel("$m~[\mathrm{g}]$")
ax.set_ylabel("$\int E \mathrm{d}^2 N_\gamma/(\mathrm{d}t\mathrm{d}E) \mathrm{d}E~[\mathrm{GeV} \cdot \mathrm{cm}^{-2} \cdot \mathrm{s}^{-1} \cdot \mathrm{sr}^{-1}]$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(fontsize="small")
ax.set_xlim(2e13, 1e19)
ax.set_ylim(1e10, 1e23)

