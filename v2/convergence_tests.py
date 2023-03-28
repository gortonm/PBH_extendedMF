#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 17:38:40 2023

@author: ppxmg2
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from extended_MF_checks import envelope
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

def load_results_Isatis(mf_string="mono", modified=True, test_mass_range=False):
    """
    Read in constraints on f_PBH, obtained using Isatis, with a monochromatic PBH mass function.

    Parameters
    ----------
    mf_string : String, optional
        The mass function to load constraints for. Acceptable inputs are "mono" (monochromatic), "LN" (log-normal), "SLN" (skew-lognormal) and "CC3" (critical collapse 3), plus the value of the power spectrum width Delta. 
    modified : Boolean, optional
        If True, use data from the modified version of Isatis. The modified version corrects a typo in the original version on line 1697 in Isatis.c which means that the highest-energy bin in the observational data set is not included. Otherwise, use the version of Isatis containing the typo. The default is True.
    test_mass_range : Boolean, optional
        If True, use data obtained using the same method as for the constraints from 1705.05567.

    Returns
    -------
    constraints_names : Array-like
        Name of instrument and arxiv reference for constraint on PBHs.
    f_PBH_Isatis : Array-like
        Constraint on the fraction of dark matter in PBHs, calculated using Isatis.

    """
    # Choose path to Isatis.
    if modified:
        Isatis_path = "../../Downloads/version_finale/scripts/Isatis/"
    else:
        Isatis_path = "../../Downloads/version_finale_unmodified/scripts/Isatis/"
    
    if test_mass_range:
        mf_string += "_test_range"
    
    # Load Isatis constraints data.
    constraints_file = np.genfromtxt("%sresults_photons_%s.txt"%(Isatis_path, mf_string), dtype = "str", unpack=True)[1:]
    
    constraints_names = []
    f_PBH_Isatis = []
    
    # Create array of constraints for which the constraints are physical
    # (i.e. the constraints are non-zero and positive).
    for i in range(len(constraints_file)):

        constraint = [float(constraints_file[i][j]) for j in range(1, len(constraints_file[i]))]
            
        if not(all(np.array(constraint)<=0)):
    
            f_PBH_Isatis.append(constraint)
            
            # Create labels
            # Based upon code appearing in plotting.py within Isatis.
            temp = constraints_file[i][0].split("_")
            temp2 = ""
            for i in range(len(temp)-1):
                temp2 = "".join([temp2,temp[i],'\,\,'])
            temp2 = "".join([temp2,'\,\,[arXiv:',temp[-1],']'])
    
            constraints_names.append(temp2)
            
    return constraints_names, f_PBH_Isatis



LN_bool = False
SLN_bool = True
CC3_bool = False

Deltas = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
cutoff_values = [1e-2, 1e-3, 1e-5, 1e-7]

# PBH mass spacing, in log10(PBH mass / grams)
dm_values = [1e-3]

# Minimum and maximum central masses.
mc_max_values = [1e19, 1e17, 1e15]

# Number of energies to use
E_number_values = [500, 1000]

delta_log_m = 1e-2

E_number_color = ["tab:blue", "tab:orange"]

BlackHawk_path = "../../Downloads/version_finale/scripts/Isatis/"
Isatis_path = BlackHawk_path + "scripts/Isatis/"


#file_exist = False

for mc_max in mc_max_values:

    for i in range(len(Deltas)):
        
        if i == 0:
            
            if CC3_bool:
                #E_number_values = [1000, 10000]
                E_number_values = [1000]
                cutoff_values = [1e-5, 1e-7]
                               
            elif LN_bool:
                cutoff_values = [1e-3, 1e-5, 1e-7]
                
            if mc_max < 1e17:
                
                if SLN_bool:
                    cutoff_values = [1e-3, 1e-5, 1e-7]
                    
                elif LN_bool:
                    E_number_values = [10000, 100000]
                    
        if i == 6:
            
            if SLN_bool:
                cutoff_values = [1e-5, 1e-7]
            
            elif CC3_bool:
                cutoff_values = [1e-3, 1e-5, 1e-7]
               
            if mc_max < 1e19:
                
                if CC3_bool:
                    cutoff_values = [1e-7, 1e-8]

        
        E_number_most_precise = max(E_number_values)
        cutoff_value_most_precise = min(cutoff_values)
        if E_number_most_precise < 1e3:
            energies_string_most_precise = "E{:.0f}".format(E_number_most_precise)
        else:
            energies_string = "E{:.0f}".format(np.log10(E_number_values[-1]))
        
        if LN_bool:
            fname_base_most_precise = "LN_D={:.1f}_dm{:.0f}_".format(Deltas[i], -np.log10(delta_log_m)) + energies_string + "_c{:.0f}".format(-np.log10(cutoff_value_most_precise))
        elif SLN_bool:
            fname_base_most_precise = "SL_D={:.1f}_dm{:.0f}_".format(Deltas[i], -np.log10(delta_log_m)) + energies_string + "_c{:.0f}".format(-np.log10(cutoff_value_most_precise))
        elif CC3_bool:
            fname_base_most_precise = "CC_D={:.1f}_dm{:.0f}_".format(Deltas[i], -np.log10(delta_log_m)) + energies_string + "_c{:.0f}".format(-np.log10(cutoff_value_most_precise))
        
        fname_base_most_precise += "_mc{:.0f}".format(np.log10(mc_max))

        constraints_names, f_PBHs_most_precise = load_results_Isatis(mf_string=fname_base_most_precise)
        print(f_PBHs_most_precise)
        f_PBH_most_precise = envelope(f_PBHs_most_precise)[0]
        print(f_PBH_most_precise)
        
        
        fig, ax = plt.subplots(figsize=(8, 7))
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
                
                destination_folder = fname_base
                filepath_Isatis = BlackHawk_path + destination_folder

                """
                if not os.path.exists(filepath_Isatis):
                    file_exist = True

                while not file_exist:
                    k +=1
                    while not file_exist:
                        l += 1
                        
                file_exist = False
                """

                constraints_names, fPBHs = load_results_Isatis(mf_string=fname_base)
                f_PBH = envelope(fPBHs)
    
                if k == 0:
                    ax.plot(cutoff_values[k], f_PBH, label="{:.0f}".format(E_number), color=E_number_color[l], marker="x", markersize=10)
                else:
                    ax.plot(cutoff_values[k], f_PBH, color=E_number_color[l], marker="x", markersize=10)  
                    
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
        fig.savefig("./Convergence_tests/Galactic_Centre/Figures/%s.png" % fname_base)