#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 14:06:29 2023

@author: ppxmg2
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

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


filepath = './Data_files/constraints_extended_MF'


# Mass function parameter values, from 2009.03204.
Deltas = np.array([0., 0.1, 0.3, 0.5, 1.0, 2.0, 5.0])
sigmas = np.array([0.55, 0.55, 0.57, 0.60, 0.71, 0.97, 2.77])
alphas_SL = np.array([-2.27, -2.24, -2.07, -1.82, -1.31, -0.66, -1.39])

alphas_CC = np.array([3.06, 3.09, 3.34, 3.82, 5.76, 18.9, 13.9])
betas = np.array([2.12, 2.08, 1.72, 1.27, 0.51, 0.0669, 0.0206])


mp_subaru = 10**np.linspace(20, 29, 1000)

constraints_names = ["COMPTEL[arXiv:1107.0200]", "EGRET[arXiv:9811211]", "Fermi-LAT[arXiv:1101.1381]", "INTEGRAL[arXiv:1107.0200]"]
colors=["tab:blue", "tab:orange"]
styles=["solid", "dashdot", "dashed", "dotted"]

for i in range(len(Deltas)):
    # Constraints for extended MF from microlensing.    
    data_filename_SLN_HSC = filepath + "/SLN_HSC_Carr_Delta={:.1f}".format(Deltas[i])
    data_filename_GCC_HSC = filepath + "/GCC_HSC_Carr_Delta={:.1f}".format(Deltas[i])
    mp_HSC, f_pbh_SLN_peak_HSC = np.loadtxt(data_filename_SLN_HSC, delimiter="\t")
    mp_HSC, f_pbh_GCC_HSC = np.loadtxt(data_filename_GCC_HSC, delimiter="\t")
    
    # Constraints for extended MF from evaporation.    
    envelope_filename_SLN_evap = filepath + "/SLN_GC_envelope_Carr_Delta={:.1f}".format(Deltas[i])
    envelope_filename_GCC_evap = filepath + "/GCC_GC_envelope_Carr_Delta={:.1f}".format(Deltas[i])
    mp_evap, f_pbh_SLN_peak_evap_envelope = np.loadtxt(envelope_filename_SLN_evap, delimiter="\t")
    mp_evap, f_pbh_GCC_evap_envelope = np.loadtxt(envelope_filename_GCC_evap, delimiter="\t")

    params_SLN = [sigmas[i], alphas_SL[i]]
    params_SLN_peak = [sigmas[i], alphas_SL[i], i]
    params_GCC = [alphas_CC[i], betas[i]]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    ax.plot(mp_HSC, f_pbh_SLN_peak_HSC, label="SLN", color=colors[0])
    ax.plot(mp_HSC, f_pbh_GCC_HSC, label="GCC", color=colors[1], linestyle="dotted", linewidth=3)
    
    ax.plot(mp_evap, f_pbh_SLN_peak_evap_envelope, color=colors[0])
    ax.plot(mp_evap, f_pbh_GCC_evap_envelope, color=colors[1], linestyle="dotted", linewidth=3)
    
    """
    for j in range(len(constraints_names)):
        mp_evap, constraints_extended_Carr_SLN = np.loadtxt(filepath + "/SLN_GC_" + str(constraints_names[j]) + "_Carr_Delta={:.1f}".format(Deltas[i]), delimiter="\t")
        mp_evap, constraints_extended_Carr_GCC = np.loadtxt(filepath + "/GCC_GC_" + str(constraints_names[j]) + "_Carr_Delta={:.1f}".format(Deltas[i]), delimiter="\t")
        ax.plot(mp_evap, constraints_extended_Carr_SLN, label=constraints_names[j], color=colors[0], alpha=0.5, linestyle=styles[j])
        ax.plot(mp_evap, constraints_extended_Carr_GCC, color=colors[1], alpha=0.5, linestyle=styles[j])
    """
    ax.set_xlabel(r"$M_\mathrm{p}~[\mathrm{g}]$")
    ax.set_ylabel(r"$f_\mathrm{PBH}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(title=r"$\Delta = {:.1f}$".format(Deltas[i]))
    ax.set_xlim(1e14, 1e29)
    ax.set_ylim(1e-3, 1)
    fig.tight_layout()
    plt.savefig("./Figures/Combined_constraints/Delta={:.1f}.png".format(Deltas[i]))
