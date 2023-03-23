#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 09:23:57 2023

@author: ppxmg2
"""

# Program for sanity checks on convergence test results. Includes plots 
# comparing mass function used by BlackHawk to the expected result, given a
# cutoff.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from preliminaries import SLN, CC3
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


#%% Compare mass functions used by BlackHawk calculations to that expected.

# Path to BlackHawk
BlackHawk_path = os.path.expanduser('~') + "/Downloads/version_finale/"

cutoff = 1e-7
m_c = 1e19
delta_log_m = 1e-4

scaled_masses_filename = "MF_scaled_mass_ranges_c={:.0f}.txt".format(-np.log10(cutoff))
[Deltas, m_lower_LN, m_upper_LN, m_lower_SLN, m_upper_SLN, m_lower_CC3, m_upper_CC3] = np.genfromtxt(scaled_masses_filename, delimiter="\t\t ", skip_header=2, unpack=True)

# Load mass function parameters.
[Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)

SLN_bool = True
CC3_bool = False

for i in range(len(Deltas)):
    
    if SLN_bool:
        fname_base = "SL_D={:.1f}_dm={:.0f}".format(Deltas[i], -np.log10(delta_log_m))
    elif CC3_bool:
        fname_base = "CC_D={:.1f}_dm={:.0f}".format(Deltas[i], -np.log10(delta_log_m))
    
    # Indicates which range of masses are being used (for convergence tests).
    fname_base += "_c={:.0f}".format(-np.log10(cutoff))
    fname_base += "_mc={:.0f}".format(np.log10(m_c))
        
    destination_folder = fname_base
    filename_BH_spec = BlackHawk_path + "/src/tables/users_spectra/" + destination_folder
    
    if SLN_bool: 
        m_min, m_max = m_lower_SLN[i] * m_c, m_upper_SLN[i] * m_c
        BH_number = int((np.log10(m_c*m_upper_SLN[i])-np.log10(m_c*m_lower_SLN[i])) / delta_log_m)
        m_range = np.logspace(np.log10(m_min), np.log10(m_max), BH_number)
        mf_values = SLN(m_range, m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i])
        title = "SLN, $\Delta={:.1f}$".format(Deltas[i])

    elif CC3_bool: 
        m_min, m_max = m_lower_CC3[i] * m_c, m_upper_CC3[i] * m_c
        BH_number = int((np.log10(m_c*m_upper_CC3[i])-np.log10(m_c*m_lower_CC3[i])) / delta_log_m)
        m_range = np.logspace(np.log10(m_min), np.log10(m_max), BH_number)
        mf_values = CC3(m_range, m_c, alpha=alphas_CC3[i], beta=betas[i])
        title = "CC3, $\Delta={:.1f}$".format(Deltas[i])
    
    m_BH, mf_BH = np.genfromtxt(filename_BH_spec, skip_header=1, delimiter="\t", unpack=True)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(m_range, mf_values, label="Calculated", linewidth=3)
    ax.plot(m_BH, mf_BH, label="From BlackHawk", linestyle="dotted", linewidth=3)
    ax.hlines(cutoff*max(mf_values), xmin=min(m_range), xmax=max(m_range), linestyle="dashed", color="k", alpha=0.5)
    ax.set_xlabel("$m~[\mathrm{g}]$")
    ax.set_ylabel("$\psi(m)$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(min(m_range), max(m_range))
    ax.set_title(title)
    ax.legend(fontsize="small")
    fig.tight_layout()
    
#%% Compare photon spectrum for cases where the constraint is calculated
# to be zero to cases where it has a finite value.

Delta = 5.0
cutoff = 1e-3
m_c = 1e19
dm_values = [1e-3, 1e-4]

colors = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200']

# Path to BlackHawk
BlackHawk_path = os.path.expanduser('~') + "/Downloads/version_finale/"
file_path_BlackHawk_data = BlackHawk_path + "results/"

fig, ax = plt.subplots(figsize=(7, 6))

for i, delta_log_m in enumerate(dm_values):
    energies_primary, spectrum_primary = read_blackhawk_spectra(file_path_BlackHawk_data + "SL_D={:.1f}_dm={:.0f}_c={:.0f}_mc={:.0f}/".format(Delta, -np.log10(delta_log_m), -np.log10(cutoff), np.log10(m_c)) + "instantaneous_primary_spectra.txt", col=1)
    energies_tot, spectrum_tot = read_blackhawk_spectra(file_path_BlackHawk_data + "SL_D={:.1f}_dm={:.0f}_c={:.0f}_mc={:.0f}/".format(Delta, -np.log10(delta_log_m), -np.log10(cutoff), np.log10(m_c)) + "instantaneous_secondary_spectra.txt", col=1)
    ax.plot(energies_primary, spectrum_primary, linestyle="dotted", color=colors[i])
    ax.plot(energies_tot, spectrum_tot, color=colors[i], label="{:.0e}".format(delta_log_m))
    
ax.legend(title="$\delta \log_{10}(m / \mathrm{g})$", fontsize="small")
ax.set_xlabel("$E~[\mathrm{GeV}]$")
ax.set_ylabel(r"$\tilde{Q}_\gamma(E)~[\mathrm{GeV^{-1} \cdot \mathrm{cm}^{-2} \cdot \mathrm{s}^{-1} \cdot \mathrm{sr}^{-1}}]$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title("$m_c = {:.0e}".format(m_c) + "~\mathrm{g}$" + ", cutoff={:.0e}".format(cutoff))
ax.set_xlim(1e-5, 5)
ax.legend(title="$\delta \log_{10}(m / \mathrm{g})$", fontsize="small")
fig.tight_layout()
