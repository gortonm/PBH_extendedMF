#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 17:38:40 2023
@author: ppxmg2
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from extended_MF_checks import envelope, load_results_Isatis
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


#%%

LN_bool = True
SLN_bool = False
CC3_bool = False

Deltas = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
cutoff_values = [1e-2, 1e-3, 1e-4, 1e-5, 1e-7]

# PBH mass spacing, in log10(PBH mass / grams)
delta_log_m = 1e-3

# Minimum and maximum central masses.
mc_max_values = [1e19, 1e17]

# Number of energies to use
E_number_values = [500, 1000]

E_number_color = ["tab:blue", "tab:orange"]
E_number_marker = ["x", "+"]

# If True, compare results to an especially accurate calculation of the 
# constraint, which does not return results for all Delta (due to Isatis crashing).
most_precise_tot = True

BlackHawk_path = "../../Downloads/version_finale/scripts/Isatis/"
Isatis_path = BlackHawk_path + "scripts/Isatis/"

if LN_bool:
    Deltas = Deltas[:-1]

for mc_max in mc_max_values:

    for i in range(len(Deltas)):
        
        if most_precise_tot:
            E_number_most_precise = 10000
            cutoff_value_most_precise = 1e-7
            dm_value_most_precise = 1e-4
        else:
            E_number_most_precise = 1000
            cutoff_value_most_precise = 1e-7
            dm_value_most_precise = 1e-3
       
        if E_number_most_precise < 1e3:
            energies_string_most_precise = "E{:.0f}".format(E_number_most_precise)
        else:
            energies_string = "E{:.0f}".format(np.log10(E_number_most_precise))
        
        if LN_bool:
            fname_base_most_precise = "LN_D={:.1f}_dm{:.0f}_".format(Deltas[i], -np.log10(dm_value_most_precise)) + energies_string
            fig_name =  "LN_D={:.1f}_mc=1e{:.0f}".format(Deltas[i], np.log10(mc_max))
        elif SLN_bool:
            fname_base_most_precise = "SL_D={:.1f}_dm{:.0f}_".format(Deltas[i], -np.log10(dm_value_most_precise)) + energies_string
            fig_name = "SL_D={:.1f}_mc=1e{:.0f}".format(Deltas[i], np.log10(mc_max))
        elif CC3_bool:
            fname_base_most_precise = "CC_D={:.1f}_dm{:.0f}_".format(Deltas[i], -np.log10(dm_value_most_precise)) + energies_string
            fig_name = "CC_D={:.1f}_mp=1e{:.0f}g".format(Deltas[i], np.log10(mc_max))
        
        fname_base_most_precise += "_c{:.0f}_mc{:.0f}".format(-np.log10(cutoff_value_most_precise), np.log10(mc_max))
        
        if most_precise_tot:
            fig_name += "_most_precise_tot"
        
        constraints_names, f_PBHs_most_precise = load_results_Isatis(mf_string=fname_base_most_precise)
        f_PBH_most_precise = envelope(f_PBHs_most_precise)[0]        
        
        fig, ax = plt.subplots(figsize=(8.5, 6))
        ax.hlines(f_PBH_most_precise, min(cutoff_values), max(cutoff_values), color="k", linestyle="dashed")
        ax.hlines(0.95 * f_PBH_most_precise, min(cutoff_values), max(cutoff_values), color="grey", linestyle="dashed")
        ax.hlines(1.05 * f_PBH_most_precise, min(cutoff_values), max(cutoff_values), color="grey", linestyle="dashed")
        ax.hlines(0.9 * f_PBH_most_precise, min(cutoff_values), max(cutoff_values), color="grey", linestyle="dotted")
        ax.hlines(1.1 * f_PBH_most_precise, min(cutoff_values), max(cutoff_values), color="grey", linestyle="dotted")
       
        for k in range(len(cutoff_values)):
        
            for l in range(len(E_number_values)):
                
                E_number = E_number_values[l]
                
                if E_number < 1e3:
                    energies_string = "E{:.0f}".format(E_number)
                else:
                    energies_string = "E{:.0f}".format(np.log10(E_number))
                
                if LN_bool:
                    fname_base = "LN_D={:.1f}_dm{:.0f}_".format(Deltas[i], -np.log10(delta_log_m)) + energies_string
                    mf_title_string = "LN"
                elif SLN_bool:
                    fname_base = "SL_D={:.1f}_dm{:.0f}_".format(Deltas[i], -np.log10(delta_log_m)) + energies_string
                    mf_title_string = "SLN"
                elif CC3_bool:
                    fname_base = "CC_D={:.1f}_dm{:.0f}_".format(Deltas[i], -np.log10(delta_log_m)) + energies_string
                    mf_title_string = "CC3"
   
                fname_base += "_c{:.0f}_mc{:.0f}".format(-np.log10(cutoff_values[k]), np.log10(mc_max))
                
                filepath_Isatis = BlackHawk_path + fname_base
                constraints_names, fPBHs = load_results_Isatis(mf_string=fname_base)
                f_PBH = envelope(fPBHs)
    
                if k == 0:
                    ax.plot(cutoff_values[k], f_PBH, label="{:.0f}".format(E_number), color=E_number_color[l], linestyle="None", marker=E_number_marker[l], markersize=10)
                else:
                    ax.plot(cutoff_values[k], f_PBH, color=E_number_color[l], marker=E_number_marker[l], markersize=10)  
                    
        ax.set_xlabel("Cutoff in $\psi(m) / \psi_\mathrm{max}$")
        ax.set_ylabel("$f_\mathrm{PBH}$")
        ax.legend(fontsize="small", title="Number of \n primary particle energies")
        ax.set_xscale("log")
        if CC3_bool:
            ax.set_title(mf_title_string + ", $\Delta={:.1f}$, $m_p={:.0e}$".format(Deltas[i], mc_max) + "$~\mathrm{g}$")
        else:
            ax.set_title(mf_title_string + ", $\Delta={:.1f}$, $m_c={:.0e}$".format(Deltas[i], mc_max) + "$~\mathrm{g}$")
        ax.invert_xaxis()
        fig.tight_layout()
        fig.savefig("./Convergence_tests/Galactic_Centre/Figures/%s.png" % fig_name)
