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


#%%

# Path to BlackHawk
BlackHawk_path = os.path.expanduser('~') + "/Downloads/version_finale/"

cutoff = 1e-7
m_c = 1e19
delta_log_m = 1e-5

scaled_masses_filename = "MF_scaled_mass_ranges_c={:.0f}.txt".format(-np.log10(cutoff))
[Deltas, m_lower_LN, m_upper_LN, m_lower_SLN, m_upper_SLN, m_lower_CC3, m_upper_CC3] = np.genfromtxt(scaled_masses_filename, delimiter="\t\t ", skip_header=2, unpack=True)

# Load mass function parameters.
[Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)

SLN_bool = False
CC3_bool = True

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