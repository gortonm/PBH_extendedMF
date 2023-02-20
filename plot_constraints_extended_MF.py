#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 14:06:29 2023

@author: ppxmg2
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from constraints_extended_MF import load_data, skew_LN, CC3

# Code for studying 2009.03204, on extended mass functions. 

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

# Path to data for extended MF results
filepath = './Data_files/constraints_extended_MF'

# Path to Isatis
Isatis_path = "./../Downloads/version_finale/scripts/Isatis/"


# Mass function parameter values, from 2009.03204.
Deltas = np.array([0., 0.1, 0.3, 0.5, 1.0, 2.0, 5.0])
sigmas_SLN = np.array([0.55, 0.55, 0.57, 0.60, 0.71, 0.97, 2.77])
alphas_SL = np.array([-2.27, -2.24, -2.07, -1.82, -1.31, -0.66, 1.39])

alphas_CC = np.array([3.06, 3.09, 3.34, 3.82, 5.76, 18.9, 13.9])
betas = np.array([2.12, 2.08, 1.72, 1.27, 0.51, 0.0669, 0.0206])

# Log-normal parameter values, from 2008.02389
sigmas_LN = np.array([0.374, 0.377, 0.395, 0.430, 0.553, 0.864])

mp_subaru = 10**np.linspace(20, 29, 1000)
m_pbh_values = 10**np.arange(11, 19.05, 0.1)

#%% Plot Isatis constraints on Galactic centre photons for a monochromatic MF.

# Load monochromatic MF results from evaporation, and calculate the envelope of constraints.
envelope_evap_mono = []


# Load result from Isatis
results_name = "results_photons_GC_mono"

constraints_file = np.genfromtxt("%s%s.txt" % (Isatis_path,results_name), dtype="str")
constraints_names_bis = constraints_file[0, 1:]
constraints = np.zeros([len(constraints_file)-1, len(constraints_file[0])-1])
for i in range(len(constraints)):
    for j in range(len(constraints[0])):
        constraints[i, j] = float(constraints_file[i+1, j+1])

# Choose which constraints to plot, and create labels.
constraints_names = []
constraints_extended_plotting = []

fig, ax = plt.subplots(figsize=(6,6))

for i in range(len(constraints_names_bis)):
    # Only include labels for constraints that Isatis calculated.
    if not(np.all(constraints[:, i] == -1.) or np.all(constraints[:, i] == 0.)):
        temp = constraints_names_bis[i].split("_")
        temp2 = ""
        
        for j in range(len(temp)-1):
            temp2 = "".join([temp2, temp[j], "\,\,"])
        temp2 = "".join([temp2, "\,\,[arXiv:",temp[-1], "]"])

        constraints_names.append(temp2)
        constraints_extended_plotting.append(constraints[:, i])


for j in range(len(m_pbh_values)):
    constraints_mono = []
    for l in range(len(constraints_names)):
        constraints_mono.append(constraints_extended_plotting[l][j])
        
    # use absolute value sign to not include unphysical cases with
    # f_PBH = -1 in the envelope
    envelope_evap_mono.append(min(abs(np.array(constraints_mono))))

# Plot evaporation constraints loaded from Isatis, to check the envelope
# works correctly
colors_evap = ["tab:blue", "tab:orange", "tab:red", "tab:green"]

for i in range(len(constraints_names)):
    ax.plot(m_pbh_values, constraints_extended_plotting[i], label=constraints_names[i], color=colors_evap[i])
ax.plot(m_pbh_values, envelope_evap_mono, label="Envelope", color="k")
ax.set_xlim(1e14, 1e18)
ax.set_ylim(10**(-10), 1)
ax.set_xlabel("$M_\mathrm{PBH}~[\mathrm{g}]$")
ax.set_ylabel("$f_\mathrm{PBH}$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(fontsize="small")
plt.tight_layout()
plt.savefig("./Figures/Combined_constraints/test_envelope_GC.png")

#%%

# use first three colours from colourblind-friendly colour cycle
# "tableau-colorblind10" 
# viscid-hub.github.io/Viscid-docs/docs/dev/styles/tableau-colorblind10.html
colors=["#006BA4", "#FF800E", "#ABABAB", "k"]
styles=["solid", "dashdot", "dashed", "dotted"]

# Subaru-HSC constraints, for a monochromatic mass function.
m_subaru_mono, f_max_subaru_mono = load_data("Subaru-HSC_2007.12697.csv")
f_mono = np.concatenate((envelope_evap_mono, f_max_subaru_mono))
m_mono = np.concatenate((m_pbh_values, m_subaru_mono))

for i in range(len(Deltas)):

    # Constraints for extended MF from microlensing.    
    data_filename_SLN_HSC = filepath + "/SLN_HSC_Carr_Delta={:.1f}".format(Deltas[i])
    data_filename_CC3_HSC = filepath + "/CC3_HSC_Carr_Delta={:.1f}".format(Deltas[i])
    data_filename_LN_HSC = filepath + "/LN_HSC_Carr_Delta={:.1f}".format(Deltas[i])
    
    mc_HSC, f_pbh_SLN_HSC = np.loadtxt(data_filename_SLN_HSC, delimiter="\t")
    mp_HSC, f_pbh_CC3_HSC = np.loadtxt(data_filename_CC3_HSC, delimiter="\t")
    mc_HSC, f_pbh_LN_HSC = np.loadtxt(data_filename_LN_HSC, delimiter="\t")
    
    # Constraints for extended MF from evaporation.    
    envelope_filename_SLN_evap = filepath + "/SLN_GC_envelope_Carr_Delta={:.1f}".format(Deltas[i])
    envelope_filename_CC3_evap = filepath + "/CC3_GC_envelope_Carr_Delta={:.1f}".format(Deltas[i])
    envelope_filename_LN_evap = filepath + "/LN_GC_envelope_Carr_Delta={:.1f}".format(Deltas[i])

    mc_evap, f_pbh_SLN_evap_envelope = np.loadtxt(envelope_filename_SLN_evap, delimiter="\t")
    mp_evap, f_pbh_CC3_evap_envelope = np.loadtxt(envelope_filename_CC3_evap, delimiter="\t")
    mc_evap, f_pbh_LN_evap_envelope = np.loadtxt(envelope_filename_LN_evap, delimiter="\t")

    mc_values = np.concatenate((mc_evap, mc_HSC))
    mp_values = np.concatenate((mp_evap, mp_HSC))
    f_pbh_SLN = np.concatenate((f_pbh_SLN_evap_envelope, f_pbh_SLN_HSC))
    f_pbh_CC3 = np.concatenate((f_pbh_CC3_evap_envelope, f_pbh_CC3_HSC))
    f_pbh_LN = np.concatenate((f_pbh_LN_evap_envelope, f_pbh_LN_HSC))

    params_SLN = [sigmas_SLN[i], alphas_SL[i]]
    params_CC3 = [alphas_CC[i], betas[i]]
    params_LN = [sigmas_LN[i]]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    ax1 = axes[0][0]
    ax2 = axes[0][1]
    ax3 = axes[1][0]
    ax4 = axes[1][1]
        
    # Estimate mass at which the SLN MF peaks.
    mp_SLN = []
    
    # Mass at which the LN MF peaks
    mp_LN = mc_values * np.exp(-sigmas_LN[i]**2)
    
    # Estimate mean mass of the SLN MF.
    m_mean_SLN = []
    
    # Estimate mean mass of the CC3 MF.
    m_mean_CC3 = []
    
    # Mean mass of the lognormal MF.
    m_mean_LN = mc_values * np.exp(sigmas_LN[i]**2/2)


    for m_c in mc_values:
        m_pbh_values_temp = np.logspace(np.log10(m_c)-4, np.log10(m_c)+4, 10000)
        psi_SLN_values = skew_LN(m_pbh_values_temp, m_c, sigma=sigmas_SLN[i], alpha=alphas_SL[i])
        psi_CC3_values = CC3(m_pbh_values_temp, m_c, alpha=alphas_CC[i], beta=betas[i])
        
        mp_SLN.append(m_pbh_values_temp[np.argmax(psi_SLN_values)])
        
        m_mean_SLN.append(np.trapz(psi_SLN_values*m_pbh_values_temp, m_pbh_values_temp))
        m_mean_CC3.append(np.trapz(psi_CC3_values*m_pbh_values_temp, m_pbh_values_temp))


    ax1.plot(mc_values, f_pbh_SLN, label="SLN", color=colors[0])
    ax2.plot(mp_values, f_pbh_CC3, label="CC3", color=colors[1])
    ax2.plot(mp_SLN, f_pbh_SLN, label="SLN", color=colors[0], linestyle="dashdot")
    ax3.plot(m_mean_SLN, f_pbh_SLN, label="SLN", color=colors[0])
    ax4.plot(m_mean_CC3, f_pbh_CC3, label="CC3", color=colors[1])
    
    # Plot a single point of the CC3 MF so that it appears in the legend
    # for the first axis.
    ax1.plot(0, 0, color=colors[1], label="CC3")

    # Don't plot lognormal results for Delta=5.0
    if Deltas[i] < 5.0:
        ax1.plot(mc_values, f_pbh_LN, color=colors[2], label="LN", linestyle="dashed", linewidth=2)
        ax2.plot(mp_LN, f_pbh_LN, color=colors[2], label="LN", linestyle="dashed", linewidth=2)
        ax3.plot(m_mean_LN, f_pbh_LN, color=colors[2], label="LN", linestyle="dashed", linewidth=2)
        ax4.plot(m_mean_LN, f_pbh_LN, color=colors[2], label="LN", linestyle="dashed", linewidth=2)
    
    ax1.set_xlabel(r"$M_{c}$ [g]")
    ax2.set_xlabel(r"$M_{p}~[\mathrm{g}]$")
    ax3.set_xlabel(r"$\langle M \rangle$ [g]")
    ax4.set_xlabel(r"$\langle M \rangle$ [g]")
   
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_ylabel(r"$f_\mathrm{PBH}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e14, 1e29)
        ax.set_ylim(1e-3, 1)
        ax.plot(m_mono, f_mono, color=colors[3], linestyle="dotted", label="Monochromatic")
    
    ax1.legend(title=r"$\Delta = {:.1f}$".format(Deltas[i]), fontsize="small")

    fig.tight_layout()
    fig.savefig("./Figures/Combined_constraints/constraints_Delta={:.1f}.png".format(Deltas[i]))
