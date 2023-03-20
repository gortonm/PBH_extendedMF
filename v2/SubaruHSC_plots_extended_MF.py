#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 11:38:16 2023

@author: ppxmg2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from preliminaries import *
from extended_MF_checks import constraint_Carr

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

filepath = './Extracted_files/'

#%%
# Test: calculate constraints from Fig. 20 of 2002.12778.
# The monochromatic MF constraints used match Fig. 6 of 1910.01285.

mc_subaru = 10**np.linspace(-15, -4, 100)

# Load data files
m_subaru_mono, f_max_subaru_mono = load_data("2002.12778/Subaru-HSC_2002.12778_mono.csv")
m_subaru_mono_Smyth, f_max_subaru_mono_Smyth = load_data("1910.01285/Subaru-HSC_1910.01285.csv")
mc_subaru_LN, f_pbh_subaru_LN = load_data("2002.12778/Subaru-HSC_2002.12778_LN.csv")

# Convert 1910.01285 constraint masses from grams to solar masses
m_subaru_mono_Smyth /= 1.989e33

sigma = 2

if "__main__" == __name__:

    # Compare constraints from Subaru-HSC for a monochromatic MF.
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(m_subaru_mono, f_max_subaru_mono, label='Extracted (2002.12778)')
    ax.plot(m_subaru_mono_Smyth, f_max_subaru_mono_Smyth, linewidth=7, alpha=0.5, linestyle='dotted', label='Extracted (1910.01285)')
    ax.set_xlabel('$M_\mathrm{PBH}~[M_\odot]$')
    ax.set_ylabel('$f_\mathrm{PBH}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title("Monochromatic MF")
    ax.legend()
    fig.tight_layout()

    # Calculate constraints from Subaru-HSC for a log-normal mass function.
    # Compare to results extracted from RH panel of Fig. 20 of 2002.12778.
    f_pbh_subaru = []
    params = [sigma]

    f_pbh_subaru = constraint_Carr(mc_subaru, m_subaru_mono_Smyth, f_max_subaru_mono_Smyth, LN, params)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(mc_subaru_LN, f_pbh_subaru_LN, label='Extracted (2002.12778)')
    ax.plot(mc_subaru, f_pbh_subaru, label='Calculated', linestyle='dashed')
    ax.set_xlabel('$M_\mathrm{c}~[M_\odot]$')
    ax.set_ylabel('$f_\mathrm{PBH}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.set_xlim(10**(-12), 1e-4)
    ax.set_ylim(1e-3, 1)
    ax.set_title("Log-normal MF ($\sigma = {:.1f}$)".format(sigma))
    fig.tight_layout()

#%%
# Calculate Subaru-HSC constraint for an extended MF, using the monochromatic
# MF constraint from 2007.12697.
# Sanity check for a very small sigma to compare to the monochromatic MF
# constraint.
# Also compare to the PBHbounds constraint

mc_subaru = 10**np.linspace(20, 29, 1000)

# Load data files
m_subaru_mono_Croon, f_max_mono_Croon = load_data("2007.12697/Subaru-HSC_2007.12697.csv")

# Load comparison constraint from PBHbounds
PBHbounds = np.transpose(np.genfromtxt("./../../PBHbounds/PBHbounds/bounds/HSC.txt", delimiter=" ", skip_header=1))
m_subaru_mono_PBHbounds, f_max_subaru_mono_PBHbounds = np.transpose(np.genfromtxt("./../../PBHbounds/PBHbounds/bounds/HSC.txt", delimiter=" ", skip_header=1))

sigma = 0.1

if "__main__" == __name__:
    # Calculate constraints for extended MF from microlensing.
    params = [sigma]
    
    f_pbh_subaru = constraint_Carr(mc_subaru, m_subaru_mono_Croon, f_max_mono_Croon, LN, params)
    
    fig, ax = plt.subplots(figsize=(6,6.5))
    ax.plot(mc_subaru, f_pbh_subaru, label="Calculated")
    ax.plot(m_subaru_mono_Croon, f_max_mono_Croon, linestyle='dotted', label="Monochromatic (Croon et al. (2020))")
    ax.plot(m_subaru_mono_PBHbounds * 1.989e33, f_max_subaru_mono_PBHbounds, linestyle='dotted', linewidth=2, label="Monochromatic (PBHbounds)")
    ax.set_xlabel(r'$M_\mathrm{PBH}~[\mathrm{g}]$')
    ax.set_ylabel(r'$f_\mathrm{PBH}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.set_xlim(1e20, 1e29)
    ax.set_ylim(1e-3, 1)
    ax.set_title(r"Log-normal MF ($\sigma = {:.2f}$)".format(sigma))
    fig.tight_layout()
    

#%%
# Use skew-lognormal mass function from 2009.03204.
# Include test case with alpha=0 to compare to results obtained using a log-
# normal mass function. Monochromatic MF constraints from 2007.12697.

# Load data files
m_subaru_mono_Croon, f_max_mono_Croon = load_data("2007.12697/Subaru-HSC_2007.12697.csv")

# Range of central masses
mc_subaru = 10**np.linspace(20, 29, 100)

if "__main__" == __name__:
    # Skew-lognormal MF results
    sigma = 0.5
    params_LN = [sigma]
    params_SLN = [sigma, 0]
    f_pbh_SLN = constraint_Carr(mc_subaru, m_subaru_mono_Croon, f_max_mono_Croon, SLN, params_SLN)
    f_pbh_LN = constraint_Carr(mc_subaru, m_subaru_mono_Croon, f_max_mono_Croon, LN, params_LN)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(mc_subaru, f_pbh_SLN, label=r"Skew-lognormal ($\alpha={:.0f}$, $\sigma={:.1f}$)".format(params_SLN[1], sigma))
    ax.plot(mc_subaru, f_pbh_LN, label=r"Lognormal ($\sigma={:.1f}$)".format(sigma), linestyle="dotted", linewidth=2)
    ax.set_xlabel(r"$m_\mathrm{c}~[\mathrm{g}]$")
    ax.set_ylabel(r"$f_\mathrm{PBH}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.set_xlim(1e21, 1e29)
    ax.set_ylim(1e-3, 1)
    ax.set_title("Croon et al. (2020) [2007.12697]")
    fig.tight_layout()
    #plt.savefig("./Figures/HSC_constraints/test_2007.12697_LN_SLN.pdf")


#%%
# Use skew-lognormal mass function from 2009.03204.
# Include test case with alpha=0 to compare to results obtained using a log-
# normal mass function. Monochromatic MF constraints from 1910.01285.

# Load data files
m_subaru_mono, f_max_subaru_mono = load_data("2002.12778/Subaru-HSC_2002.12778_mono.csv")
mc_subaru_LN, f_pbh_subaru_LN = load_data("2002.12778/Subaru-HSC_2002.12778_LN.csv")

# Convert from solar masses to grams
m_subaru_mono *= 1.989e33
mc_subaru_LN *= 1.989e33

# Range of central masses
mc_subaru = 10**np.linspace(20, 29, 100)

if "__main__" == __name__:
    # Skew-lognormal MF results
    sigma = 2
    params_LN = [sigma]
    params_SLN = [sigma, 0]
    
    f_pbh_SLN = constraint_Carr(mc_subaru, m_subaru_mono, f_max_subaru_mono, SLN, params_SLN)
    f_pbh_LN = constraint_Carr(mc_subaru, m_subaru_mono, f_max_subaru_mono, LN, params_LN)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(mc_subaru, f_pbh_SLN, label=r"Skew-lognormal ($\alpha={:.0f}$, $\sigma={:.1f}$)".format(params_SLN[1], sigma))
    ax.plot(mc_subaru, f_pbh_LN, label=r"Lognormal ($\sigma={:.1f}$)".format(sigma), linestyle="dotted", linewidth=3)
    ax.plot(mc_subaru_LN, f_pbh_subaru_LN, label="Lognormal (Fig.~20 [2002.12778])", linestyle="dotted", color="k")
    ax.set_xlabel(r"$m_\mathrm{c}~[\mathrm{g}]$")
    ax.set_ylabel(r"$f_\mathrm{PBH}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.set_xlim(1e21, 1e29)
    ax.set_ylim(1e-3, 1)
    ax.set_title("Smyth et al. (2019) [1910.01285]")
    fig.tight_layout()
    #plt.savefig("./Figures/HSC_constraints/test_1910.01285_LN_SLN.png", dpi=1200)

#%% Convergence tests



"""
#%% Calculate constraints for extended MFs from 2009.03204.
mc_subaru = 10**np.linspace(17, 29, 1000)

# Constraints for monochromatic MF.
m_subaru_mono, f_max_subaru_mono = load_data("Subaru-HSC_2007.12697.csv")

# Mass function parameter values, from 2009.03204.
Deltas = np.array([0., 0.1, 0.3, 0.5, 1.0, 2.0, 5.0])
sigmas = np.array([0.55, 0.55, 0.57, 0.60, 0.71, 0.97, 2.77])
alphas_SL = np.array([-2.27, -2.24, -2.07, -1.82, -1.31, -0.66, 1.39])

alphas_CC = np.array([3.06, 3.09, 3.34, 3.82, 5.76, 18.9, 13.9])
betas = np.array([2.12, 2.08, 1.72, 1.27, 0.51, 0.0669, 0.0206])

# Log-normal parameter values, from 2008.02389
sigmas_LN = np.array([0.374, 0.377, 0.395, 0.430, 0.553, 0.864, -1])

for i in range(len(Deltas)):

    # Calculate constraints for extended MF from microlensing.
    params_SLN = [sigmas[i], alphas_SL[i]]
    params_CC3 = [alphas_CC[i], betas[i]]
    params_LN = [sigmas_LN[i]]
    
    f_pbh_skew_LN = constraint_Carr_HSC(mc_subaru, m_subaru_mono, skew_LN, params_SLN, f_max_subaru_mono)
    f_pbh_CC3 = constraint_Carr_HSC(mc_subaru, m_subaru_mono, CC3, params_CC3, f_max_subaru_mono)
    f_pbh_LN = constraint_Carr_HSC(mc_subaru, m_subaru_mono, lognormal_number_density, params_LN, f_max_subaru_mono)

    data_filename_SLN = "./Data_files/constraints_extended_MF/SLN_HSC_Carr_Delta={:.1f}".format(Deltas[i])
    data_filename_CC3 = "./Data_files/constraints_extended_MF/CC3_HSC_Carr_Delta={:.1f}".format(Deltas[i])
    data_filename_LN = "./Data_files/constraints_extended_MF/LN_HSC_Carr_Delta={:.1f}".format(Deltas[i])
    
    np.savetxt(data_filename_SLN, [mc_subaru, f_pbh_skew_LN], delimiter="\t")
    np.savetxt(data_filename_CC3, [mc_subaru, f_pbh_CC3], delimiter="\t")
    np.savetxt(data_filename_LN, [mc_subaru, f_pbh_LN], delimiter="\t")
    
"""