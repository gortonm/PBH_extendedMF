#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 11:38:16 2023

@author: ppxmg2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from constraints_extended_MF import *

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
m_subaru_mono, f_max_subaru_mono = load_data("Subaru-HSC_2002.12778_mono.csv")
m_subaru_mono_Smyth, f_max_subaru_mono_Smyth = load_data("Subaru-HSC_1910.01285.csv")
mc_subaru_LN, f_pbh_subaru_LN = load_data("Subaru-HSC_2002.12778_LN.csv")

# Convert 1910.01285 constraint masses from grams to solar masses
m_subaru_mono_Smyth /= 1.989e33

sigma = 2

if "__main__" == __name__:
    
    # Compare constraints from Subaru-HSC for a monochromatic MF.
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(m_subaru_mono, f_max_subaru_mono, label='Extracted (2002.12778)')
    ax.plot(m_subaru_mono_Smyth, f_max_subaru_mono_Smyth, linewidth=7, alpha=0.5, linestyle='dotted', label='Extracted (1910.01285)')
    ax.plot(m_subaru_mono, f_max(m_subaru_mono, m_subaru_mono, f_max_subaru_mono), linestyle='dotted', linewidth=4, label='Interpolated')
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

    for m_c in mc_subaru:
        f_pbh_subaru.append(1/np.trapz(integrand_general_mf(m_subaru_mono_Smyth, lognormal_number_density, m_c, params, m_subaru_mono_Smyth, f_max_subaru_mono_Smyth)))

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

mp_subaru = 10**np.linspace(20, 29, 1000)

# Load data files
m_subaru_mono_Croon, f_max_mono_Croon = load_data("Croon2020_R90_0.csv")

# Load comparison constraint from PBHbounds
PBHbounds = np.transpose(np.genfromtxt("./../PBHbounds/PBHbounds/bounds/HSC.txt", delimiter=" ", skip_header=1))
print(PBHbounds)
m_subaru_mono_PBHbounds, f_max_subaru_mono_PBHbounds = np.transpose(np.genfromtxt("./../PBHbounds/PBHbounds/bounds/HSC.txt", delimiter=" ", skip_header=1))

sigma = 0.2

if "__main__" == __name__:
    # Calculate constraints for extended MF from microlensing.
    f_pbh_subaru = []
    params = [sigma]
    
    for m_c in mp_subaru:
        f_pbh_subaru.append(1/np.trapz(integrand_general_mf(m_subaru_mono_Croon, lognormal_number_density, m_c, params, m_subaru_mono_Croon, f_max_mono_Croon), m_subaru_mono_Croon))

    fig, ax = plt.subplots(figsize=(6,6.5))
    ax.plot(mp_subaru, f_pbh_subaru, label="Calculated", )
    ax.plot(m_subaru_mono_Croon, m_subaru_mono_Croon, linestyle='dotted', label="Monochromatic (Croon et al. (2020))")
    ax.plot(m_subaru_mono_PBHbounds * 1.989e33, f_max_subaru_mono_PBHbounds, linestyle='dotted', linewidth=1.5, label="Monochromatic (PBHbounds)")
    ax.set_xlabel(r'$M_\mathrm{PBH}~[\mathrm{g}]$')
    ax.set_ylabel(r'$f_\mathrm{PBH}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.set_xlim(1e20, 1e29)
    ax.set_ylim(1e-3, 1)
    ax.set_title(r"Log-normal MF ($\sigma = {:.2f}$)".format(sigma) + ", $R_{90} = 0$")
    fig.tight_layout()

#%%
# Calculate Subaru-HSC constraint for an extended MF, using the monochromatic
# MF constraint from 2007.12697.

mp_subaru = 10**np.linspace(20, 29, 100)

# Load data files
m_subaru_mono, f_max_subaru_mono = load_data("Croon2020_R90_0.csv")

sigma = .5

if "__main__" == __name__:

    # Calculate constraints for extended MF from microlensing.
    f_pbh_subaru = []

    for m_c in mp_subaru:

        f_pbh_subaru.append(1/np.trapz(integrand(1, m_subaru_mono, m_c, sigma, m_subaru_mono, f_max_subaru_mono), m_subaru_mono))

    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(mp_subaru, f_pbh_subaru, label="$R_{90} = 0$", )
    ax.set_xlabel('$M_\mathrm{c}~[\mathrm{g}]$')
    ax.set_ylabel('$f_\mathrm{PBH}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.set_xlim(1e20, 1e29)
    ax.set_ylim(1e-3, 1)
    ax.set_title("Log-normal MF ($\sigma = {:.1f}$)".format(sigma))
    fig.tight_layout()


#%%
# Calculate Subaru-HSC constraint for an extended MF, using the monochromatic
# MF constraint from 2007.12697. Plot sigma=0.5 and sigma=1.0 results on the
# same figure.

mp_subaru = 10**np.linspace(20, 29, 100)

# Load data files
m_subaru_mono, f_max_subaru_mono = load_data("Croon2020_R90_0.csv")

if "__main__" == __name__:

    # Calculate constraints for extended MF from microlensing.
    f_pbh_subaru_sigma05 = []
    f_pbh_subaru_sigma10 = []

    for m_c in mp_subaru:

        f_pbh_subaru_sigma05.append(1/np.trapz(integrand(1, m_subaru_mono, m_c, 0.5, m_subaru_mono, f_max_subaru_mono), m_subaru_mono))
        f_pbh_subaru_sigma10.append(1/np.trapz(integrand(1, m_subaru_mono, m_c, 1.0, m_subaru_mono, f_max_subaru_mono), m_subaru_mono))

    fig, ax = plt.subplots(figsize=(5.5,5.5))
    ax.plot(mp_subaru, f_pbh_subaru_sigma05, label=r"$\sigma = 0.5$")
    ax.plot(mp_subaru, f_pbh_subaru_sigma10, label=r"$\sigma = 1.0$")
    ax.plot(m_subaru_mono, f_max_subaru_mono, color='k', linestyle='dotted', linewidth=3, alpha=0.5, label="Monochromatic")
    ax.set_xlabel(r'$M_\mathrm{c} / M_\mathrm{PBH} ~[\mathrm{g}]$')
    ax.set_ylabel(r'$f_\mathrm{PBH}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.set_xlim(1e21, 1e29)
    ax.set_ylim(1e-3, 1)
    ax.set_title("Log-normal MF")
    fig.tight_layout()


#%%
# Use skew-lognormal mass function from 2009.03204.
# Include test case with alpha=0 to compare to results obtained using a log-
# normal mass function. Monochromatic MF constraints from 2007.12697.

# Load data files
m_subaru_mono, f_max_subaru_mono = load_data("Croon2020_R90_0.csv")

# Range of central masses
mp_subaru = 10**np.linspace(20, 29, 100)

# Mass function parameter values, from 2009.03204.
# The Delta=-1.0 cases are a test case for when alpha=0 (i.e. a log-normal, with sigma=0.5)
deltas = np.array([-1])
sigmas = np.array([0.5, 0.55, 0.55, 0.57, 0.60, 0.71, 0.97, 2.77])
alphas_SL = np.array([0., -2.27, -2.24, -2.07, -1.82, -1.31, -0.66, 1.39])

# Skew-lognormal MF results
for i in range(len(deltas)):

    # Calculate constraints for extended MF from microlensing.
    f_pbh_skew_LN = []

    params = [sigmas[i], alphas_SL[i]]

    for m_c in mp_subaru:
        integral = np.trapz(integrand_general_mf(m_subaru_mono, skew_LN, m_c, params, m_subaru_mono, f_max_subaru_mono), m_subaru_mono)
        if integral == 0:
            f_pbh_skew_LN.append(10)
        else:
            f_pbh_skew_LN.append(1/integral)

    # Test case comparison for skew-lognormal with alpha=0 (reduces to a log-normal)
    if deltas[i] == -1.0:
        f_pbh_subaru_LN_sigma05 = []

        for m_c in mp_subaru:
            integral = np.trapz(integrand(1, m_subaru_mono, m_c, 0.5, m_subaru_mono, f_max_subaru_mono), m_subaru_mono)
            if integral == 0:
                f_pbh_subaru_LN_sigma05.append(10)
            else:
                f_pbh_subaru_LN_sigma05.append(1/integral)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    if deltas[i] == -1.0:
        ax.plot(mp_subaru, f_pbh_skew_LN, label=r"Skew-lognormal ($\alpha={:.0f}$, $\sigma={:.1f}$)".format(alphas_SL[0], sigmas[0]))
        ax.plot(mp_subaru, f_pbh_subaru_LN_sigma05, label=r"Lognormal ($\sigma={:.1f}$)".format(sigmas[0]), linestyle="dotted", linewidth=3)

    else:
        ax.plot(mp_subaru, f_pbh_skew_LN, label=r"$\Delta = {:.1f}$".format(deltas[i]))

    ax.set_xlabel(r"$m_\mathrm{c}~[\mathrm{g}]$")
    ax.set_ylabel(r"$f_\mathrm{PBH}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.set_xlim(1e21, 1e29)
    ax.set_ylim(1e-3, 1)
    ax.set_title("Croon et al. (2020) [2007.12697]")
    fig.tight_layout()
    plt.savefig("./Figures/HSC_constraints/test_2007.12697_LN_SLN.png", dpi=1200)


#%%
# Use skew-lognormal mass function from 2009.03204.
# Include test case with alpha=0 to compare to results obtained using a log-
# normal mass function. Monochromatic MF constraints from 1910.01285.

# Load data files
m_subaru_mono, f_max_subaru_mono = load_data("Subaru-HSC_mono.csv")
mp_subaru_LN, f_pbh_subaru_LN = load_data("Subaru-HSC_LN.csv")

# Convert from solar masses to grams
m_subaru_mono *= 1.989e33
mp_subaru_LN *= 1.989e33

sigma = 2

# Range of central masses
mc_subaru = 10**np.linspace(20, 29, 100)

# Mass function parameter values, from 2009.03204.
# The Delta=-1.0 cases are a test case for when alpha=0 (i.e. a log-normal, with sigma=0.5)
deltas = np.array([-1])
sigmas = np.array([2, 0.55, 0.55, 0.57, 0.60, 0.71, 0.97, 2.77])
alphas_SL = np.array([0., -2.27, -2.24, -2.07, -1.82, -1.31, -0.66, 1.39])

# Skew-lognormal MF results
for i in range(len(deltas)):

    # Calculate constraints for extended MF from microlensing.
    f_pbh_skew_LN = []

    params = [sigmas[i], alphas_SL[i]]

    for m_c in mc_subaru:
        integral = np.trapz(integrand_general_mf(m_subaru_mono, skew_LN, m_c, params, m_subaru_mono, f_max_subaru_mono), m_subaru_mono)
        if integral == 0:
            f_pbh_skew_LN.append(10)
        else:
            f_pbh_skew_LN.append(1/integral)

    # Test case comparison for skew-lognormal with alpha=0 (reduces to a log-normal)
    if deltas[i] == -1.0:
        f_pbh_subaru_LN_sigma2 = []

        for m_c in mp_subaru:
            integral = np.trapz(integrand(1, m_subaru_mono, m_c, 2, m_subaru_mono, f_max_subaru_mono), m_subaru_mono)
            if integral == 0:
                f_pbh_subaru_LN_sigma2.append(10)
            else:
                f_pbh_subaru_LN_sigma2.append(1/integral)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    if deltas[i] == -1.0:
        ax.plot(mp_subaru, f_pbh_skew_LN, label=r"Skew-lognormal ($\alpha={:.0f}$, $\sigma={:.1f}$)".format(alphas_SL[0], sigmas[0]))
        ax.plot(mp_subaru, f_pbh_subaru_LN_sigma2, label=r"Lognormal ($\sigma={:.1f}$)".format(sigmas[0]), linestyle="dashed", linewidth=3)
        ax.plot(mp_subaru_LN, f_pbh_subaru_LN, label="Lognormal (Fig.~20 [2002.12778])", linestyle="dotted")
    else:
        ax.plot(mp_subaru, f_pbh_skew_LN, label=r"$\Delta = {:.1f}$".format(deltas[i]))

    ax.set_xlabel(r"$m_\mathrm{c}~[\mathrm{g}]$")
    ax.set_ylabel(r"$f_\mathrm{PBH}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.set_xlim(1e21, 1e29)
    ax.set_ylim(1e-3, 1)
    ax.set_title("Smyth et al. (2019) [1910.01285]")
    fig.tight_layout()
    plt.savefig("./Figures/HSC_constraints/test_1910.01285_LN_SLN.png", dpi=1200)


#%%

mp_subaru = 10**np.linspace(17, 29, 1000)

# Load data files
m_subaru_mono, f_max_subaru_mono = load_data("Croon2020_R90_0.csv")

# Mass function parameter values, from 2009.03204.
Deltas = np.array([0., 0.1, 0.3, 0.5, 1.0, 2.0, 5.0])
sigmas = np.array([0.55, 0.55, 0.57, 0.60, 0.71, 0.97, 2.77])
alphas_SL = np.array([-2.27, -2.24, -2.07, -1.82, -1.31, -0.66, 1.39])

alphas_CC = np.array([3.06, 3.09, 3.34, 3.82, 5.76, 18.9, 13.9])
betas = np.array([2.12, 2.08, 1.72, 1.27, 0.51, 0.0669, 0.0206])

for i in range(len(Deltas)):

    # Calculate constraints for extended MF from microlensing.
    f_pbh_skew_LN = []
    f_pbh_skew_LN_peak = []
    f_pbh_GCC = []

    params_SLN = [sigmas[i], alphas_SL[i]]
    params_SLN_peak = [sigmas[i], alphas_SL[i], i]
    params_GCC = [alphas_CC[i], betas[i]]

    for m_p in mp_subaru:
        integral_SLN_peak = np.trapz(integrand_general_mf(m_subaru_mono, skew_LN_peak, m_p, params_SLN_peak, m_subaru_mono, f_max_subaru_mono), m_subaru_mono)
        integral_GCC = np.trapz(integrand_general_mf(m_subaru_mono, GCC, m_p, params_GCC, m_subaru_mono, f_max_subaru_mono), m_subaru_mono)

        if integral_SLN_peak == 0:
            f_pbh_skew_LN_peak.append(10)
        else:
            f_pbh_skew_LN_peak.append(1/integral_SLN_peak)

        if integral_GCC == 0:
            f_pbh_GCC.append(10)
        else:
            f_pbh_GCC.append(1/integral_GCC)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    ax.plot(mp_subaru, f_pbh_skew_LN_peak, label="SLN")
    ax.plot(mp_subaru, f_pbh_GCC, label="GCC", linestyle="dashed")
    ax.set_xlabel(r"$m_\mathrm{p}~[\mathrm{g}]$")
    ax.set_ylabel(r"$f_\mathrm{PBH}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(title=r"$\Delta = {:.1f}$".format(Deltas[i]))
    ax.set_xlim(min(mp_subaru), max(mp_subaru))
    ax.set_ylim(1e-3, 1)
    fig.tight_layout()
    
    data_filename_SLN = "./Data_files/constraints_extended_MF/SLN_HSC_Carr_Delta={:.1f}".format(Deltas[i])
    data_filename_GCC = "./Data_files/constraints_extended_MF/GCC_HSC_Carr_Delta={:.1f}".format(Deltas[i])
    np.savetxt(data_filename_SLN, [mp_subaru, f_pbh_skew_LN_peak], delimiter="\t")
    np.savetxt(data_filename_GCC, [mp_subaru, f_pbh_GCC], delimiter="\t")
    plt.savefig("./Figures/HSC_constraints/Delta={:.1f}.png".format(Deltas[i]))

