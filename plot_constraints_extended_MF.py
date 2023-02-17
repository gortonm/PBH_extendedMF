#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 14:06:29 2023

@author: ppxmg2
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from constraints_extended_MF import load_data

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

#%% Plot Isatis constraints on Galactic centre photons for a monochromatic MF.
colors=["tab:blue", "tab:orange", "k"]
styles=["solid", "dashdot", "dashed", "dotted"]

# Load monochromatic MF results from evaporation, and calculate the envelope of constraints.
envelope_evap_mono = []
m_pbh_values = 10**np.arange(11, 19.05, 0.1)


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

# Subaru-HSC constraints, for a monochromatic mass function.
m_subaru_mono, f_max_subaru_mono = load_data("Subaru-HSC_2007.12697.csv")


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

    params_SLN = [sigmas_SLN[i], alphas_SL[i]]
    params_CC3 = [alphas_CC[i], betas[i]]
    params_LN = [sigmas_LN[i]]

    fig1, ax1 = plt.subplots(figsize=(5.5, 5.5))
    fig2, ax2 = plt.subplots(figsize=(5.5, 5.5))

    ax1.plot(mc_HSC, f_pbh_SLN_HSC, label="SLN", color=colors[0])
    ax1.plot(mc_evap, f_pbh_SLN_evap_envelope, color=colors[0])
    ax1.plot(mc_evap, f_pbh_LN_evap_envelope, color=colors[1], label="LN", linestyle="dashed", linewidth=3)

    # Don't plot lognormal results for Delta=5.0
    if Deltas[i] < 5.0:
        ax1.plot(mc_HSC, f_pbh_LN_HSC, color=colors[1], linestyle="dashed", linewidth=3)

    ax2.plot(mp_HSC, f_pbh_CC3_HSC, label="CC3", color=colors[0])
    ax2.plot(mp_evap, f_pbh_CC3_evap_envelope, color=colors[0])
    
    # Don't plot lognormal results for Delta=5.0
    if Deltas[i] < 5.0:
        ax2.plot(mc_evap * np.exp(-sigmas_LN[i]**2), f_pbh_LN_evap_envelope, color=colors[1], label="LN", linestyle="dashed", linewidth=3)
        ax2.plot(mc_HSC * np.exp(-sigmas_LN[i]**2), f_pbh_LN_HSC, color=colors[1], linestyle="dashed", linewidth=3)
   
    """
    for j in range(len(constraints_names)):
        mp_evap, constraints_extended_Carr_SLN = np.loadtxt(filepath + "/SLN_GC_" + str(constraints_names[j]) + "_Carr_Delta={:.1f}".format(Deltas[i]), delimiter="\t")
        mp_evap, constraints_extended_Carr_CC3 = np.loadtxt(filepath + "/CC3_GC_" + str(constraints_names[j]) + "_Carr_Delta={:.1f}".format(Deltas[i]), delimiter="\t")
        ax.plot(mp_evap, constraints_extended_Carr_SLN, label=constraints_names[j], color=colors[0], alpha=0.5, linestyle=styles[j])
        ax.plot(mp_evap, constraints_extended_Carr_CC3, color=colors[1], alpha=0.5, linestyle=styles[j])
    """
    
    ax1.set_xlabel(r"$M_{c}~[\mathrm{g}]$")
    ax2.set_xlabel(r"$M_{p}~[\mathrm{g}]$")
    
    for ax in [ax1, ax2]:
        ax.set_ylabel(r"$f_\mathrm{PBH}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e14, 1e29)
        ax.set_ylim(1e-3, 1)
        ax.plot(m_pbh_values, envelope_evap_mono, color=colors[2], linestyle="dotted")
        ax.plot(m_subaru_mono, f_max_subaru_mono, color=colors[2], linestyle="dotted")
        ax.legend(title=r"$\Delta = {:.1f}$".format(Deltas[i]))

    for fig in [fig1, fig2]:
        fig.tight_layout()
    
    fig1.savefig("./Figures/Combined_constraints/SLN_LN_delta_Delta={:.1f}.png".format(Deltas[i]))
    fig2.savefig("./Figures/Combined_constraints/CC3_LN_delta_Delta={:.1f}.png".format(Deltas[i]))
