#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 14:45:28 2023

@author: ppxmg2
"""
# Script for calculating Galactic Centre constraints for the numerical mass
# function calculated in 2008.03289, using the method from 1705.05567.

import numpy as np
from preliminaries import load_data, LN, SLN, CC3, constraint_Carr, load_results_Isatis, envelope
import matplotlib.pyplot as plt
import matplotlib as mpl

# Specify the plot style
mpl.rcParams.update({'font.size': 24,'font.family': 'serif'})
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

t_0 = 13.8e9 * 365.25 * 86400    # Age of Universe, in seconds

#%% Constraints from COMPTEL, INTEGRAL, EGRET and Fermi-LAT. Calculated using f_max as the minimum constraint over each energy bin.

if "__main__" == __name__:

    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    mc_values = np.logspace(14, 20, 120)
    m_delta_values_loaded = np.logspace(11, 21, 1000)
    
    # Load delta-function MF constraints calculated using Isatis, to use the method from 1705.05567.
    # Using the envelope of constraints for each instrument for the delta-function MF constraint.
    constraints_names, f_max_Isatis = load_results_Isatis(modified=True)
    colors_evap = ["tab:orange", "tab:green", "tab:red", "tab:blue"]

    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(len(constraints_names)):
        ax.plot(m_delta_values_loaded, f_max_Isatis[i], label=constraints_names[i], color=colors_evap[i])
   
    ax.set_xlim(1e14, 1e18)
    ax.set_ylim(10**(-10), 1)
    ax.set_xlabel("$M_\mathrm{PBH}~[\mathrm{g}]$")
    ax.set_ylabel("$f_\mathrm{PBH}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize="small")
    plt.tight_layout()

    # Boolean determines whether to use evolved mass function.
    evolved = False
    # Boolean determines whether to evaluate the evolved mass function at t=0.
    t_initial = True
    if t_initial:
        evolved = True
    # If True, use extrapolated delta-function MF constraints down to 1e11g (using a power law fit) to calculate extended MF constraint.
    include_extrapolated = True
    # If True, plot extrapolated delta-function MF constraints down to 1e11g.
    plot_extrapolated = True
    
    t = t_0
    
    if not evolved:
        data_folder = "./Data-tests/unevolved"
    elif t_initial:
        data_folder = "./Data-tests/t_initial"
        t = 0
    else:
        data_folder = "./Data-tests/"
    
    
    if include_extrapolated:
        # Power-law exponent to use
        exponent_PL_lower = 0.0
        m_delta_extrapolated = np.logspace(11, 13, 21)
        data_folder += "/PL_exp_{:.0f}/".format(exponent_PL_lower)


    constraints_names_short = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]

    for j in range(len(Deltas)):
        params_LN = [sigmas_LN[j]]
        params_SLN = [sigmas_SLN[j], alphas_SLN[j]]
        params_CC3 = [alphas_CC3[j], betas[j]]
        
        # Using each energy bin per instrument individually for the delta-function MF constraint, then obtaining the tightest constraint from each instrument using envelope().
        mc_constraints_LN = []
        mc_constraints_SLN = []
        mc_constraints_CC3 = []

        for i in range(len(constraints_names_short)):
            # Calculate constraint using method from 1705.05567.

            f_max_loaded = np.array(f_max_Isatis[i])
            
            if include_extrapolated:
                f_max_loaded_truncated = f_max_loaded[m_delta_values_loaded > 1e13]
                f_max_extrapolated = f_max_loaded_truncated[0] * np.power(m_delta_extrapolated / 1e13, exponent_PL_lower)
                f_max_i = np.concatenate((f_max_extrapolated, f_max_loaded_truncated))
                m_delta_values = np.concatenate((m_delta_extrapolated, m_delta_values_loaded[m_delta_values_loaded > 1e13]))
            else:
                f_max_k = f_max_loaded
                m_delta_values = m_delta_values_loaded
                
            
            if plot_extrapolated:
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.plot(m_delta_extrapolated, f_max_extrapolated, linestyle="dashed", color=colors_evap[i])
                ax.plot(m_delta_values_loaded[m_delta_values_loaded > 1e13], f_max_loaded_truncated, label=constraints_names[i], color=colors_evap[i])
                ax.set_xlim(1e11, 1e18)
                ax.set_xlabel("$M_\mathrm{max}~[\mathrm{g}]$")
                ax.set_ylabel("$f_\mathrm{max}$")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.legend(fontsize="small")
                plt.tight_layout()
            
                        
            # Extended MF constraints at each value of the characteristic mass (for LN and SLN) or peak mass (for CC3)
            f_PBH_i_LN = []
            f_PBH_i_SLN = []
            f_PBH_i_CC3 = []
            
            for m_c in mc_values:
                
                mc_values_input = [m_c]
                                    
                # Extended MF constraint from the k'th instrument at a given value of m_c
                f_PBH_i_LN.append(constraint_Carr(mc_values_input, m_delta_values, f_max_i, LN, params_LN, evolved, t))
                f_PBH_i_SLN.append(constraint_Carr(mc_values_input, m_delta_values, f_max_i, SLN, params_SLN, evolved, t))
                f_PBH_i_CC3.append(constraint_Carr(mc_values_input, m_delta_values, f_max_i, CC3, params_CC3, evolved, t))

            # Extended MF constraints from the i'th instrument at a given value of m_c
            mc_constraints_LN.append(f_PBH_i_LN)
            mc_constraints_SLN.append(f_PBH_i_SLN)
            mc_constraints_CC3.append(f_PBH_i_CC3)
                    
        # Set f_PBH to the tightest constraint from the instruments
        f_PBH_Carr_LN = envelope(mc_constraints_LN)
        f_PBH_Carr_SLN = envelope(mc_constraints_SLN)
        f_PBH_Carr_CC3 = envelope(mc_constraints_CC3)
            
        if evolved == False:
            data_filename_LN = data_folder + "/LN_GC_Carr_Delta={:.1f}_approx_unevolved.txt".format(Deltas[j])
            data_filename_SLN = data_folder + "/SLN_GC_Carr_Delta={:.1f}_approx_unevolved.txt".format(Deltas[j])
            data_filename_CC3 = data_folder + "/CC3_GC_Carr_Delta={:.1f}_approx_unevolved.txt".format(Deltas[j])
        else:
            data_filename_LN = data_folder + "/LN_GC_Carr_Delta={:.1f}_approx.txt".format(Deltas[j])
            data_filename_SLN = data_folder + "/SLN_GC_Carr_Delta={:.1f}_approx.txt".format(Deltas[j])
            data_filename_CC3 = data_folder + "/CC3_GC_Carr_Delta={:.1f}_approx.txt".format(Deltas[j])
            
        np.savetxt(data_filename_LN, [mc_values, f_PBH_Carr_LN], delimiter="\t")
        np.savetxt(data_filename_SLN, [mc_values, f_PBH_Carr_SLN], delimiter="\t")
        np.savetxt(data_filename_CC3, [mc_values, f_PBH_Carr_CC3], delimiter="\t")


#%% Constraints from COMPTEL, INTEGRAL, EGRET and Fermi-LAT. Approximate results obtained by using f_max as the constraint from each instrument, rather than the minimum over each energy bin.

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    # Boolean determines whether to useFalse evolved mass function.
    evolved = False
    # Boolean determines whether to evaluate the evolved mass function at t=0.
    t_initial = False
    if t_initial:
        evolved = True
    
    # If True, use extrapolated delta-function MF constraints down to 1e11g (using a power law fit) to calculate extended MF constraint.
    include_extrapolated = True
    # If True, plot extrapolated delta-function MF constraints down to 1e11g.
    plot_extrapolated = False 
    
    m_delta_values_loaded = np.logspace(11, 21, 1000)
    constraints_names, f_max_Isatis = load_results_Isatis(modified=True)
    colors_evap = ["tab:orange", "tab:green", "tab:red", "tab:blue"]
    constraints_names_short = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
    
    mc_values = np.logspace(14, 20, 120)
    
    t = t_0
    
    if not evolved:
        data_folder = "./Data-tests/unevolved"
    elif t_initial:
        data_folder = "./Data-tests/t_initial"
        t = 0
    else:
        data_folder = "./Data-tests/"
        
        
    if include_extrapolated:
        # Power-law exponent to use
        exponent_PL_lower = 2.0
        m_delta_extrapolated = np.logspace(11, 13, 21)
        data_folder += "/PL_exp_{:.0f}/".format(exponent_PL_lower)
        

    for j in range(len(Deltas)):
        params_LN = [sigmas_LN[j]]
        params_SLN = [sigmas_SLN[j], alphas_SLN[j]]
        params_CC3 = [alphas_CC3[j], betas[j]]
        
        for i in range(len(constraints_names)):
            
            # Set non-physical values of f_max (-1) to 1e100 from the f_max values calculated using Isatis
            f_max_allpositive = []
    
            for f_max in f_max_Isatis[i]:
                if f_max == -1:
                    f_max_allpositive.append(1e100)
                else:
                    f_max_allpositive.append(f_max)
            
            # Extrapolate f_max at masses below 1e13g using a power-law
            if include_extrapolated:
                f_max_loaded_truncated = np.array(f_max_allpositive)[m_delta_values_loaded > 1e13]
                f_max_extrapolated = f_max_loaded_truncated[0] * np.power(m_delta_extrapolated / 1e13, exponent_PL_lower)
                f_max_i = np.concatenate((f_max_extrapolated, f_max_loaded_truncated))
                m_delta_values = np.concatenate((m_delta_extrapolated, m_delta_values_loaded[m_delta_values_loaded > 1e13]))
            else:
                f_max_i = f_max_allpositive
                m_delta_values = m_delta_values_loaded
            
            # Plot the extrapolated power-law fit to f_max
            if plot_extrapolated:
                ax.plot(m_delta_extrapolated, f_max_extrapolated, linestyle="dashed", color=colors_evap[i])
                ax.plot(m_delta_values_loaded[m_delta_values_loaded > 1e13], f_max_loaded_truncated, color=colors_evap[i])
            
            f_PBH_i_LN = constraint_Carr(mc_values, m_delta_values, f_max_i, LN, params_LN, evolved, t)
            f_PBH_i_SLN = constraint_Carr(mc_values, m_delta_values, f_max_i, SLN, params_SLN, evolved, t)
            f_PBH_i_CC3 = constraint_Carr(mc_values, m_delta_values, f_max_i, CC3, params_CC3, evolved, t)
      
            if evolved == False:
                data_filename_LN = data_folder + "/LN_GC_%s" % constraints_names_short[i] + "_Carr_Delta={:.1f}_approx_unevolved.txt".format(Deltas[j])
                data_filename_SLN = data_folder + "/SLN_GC_%s" % constraints_names_short[i]  + "_Carr_Delta={:.1f}_approx_unevolved.txt".format(Deltas[j])
                data_filename_CC3 = data_folder + "/CC3_GC_%s" % constraints_names_short[i]  + "_Carr_Delta={:.1f}_approx_unevolved.txt".format(Deltas[j])
            else:
                data_filename_LN = data_folder + "/LN_GC_%s" % constraints_names_short[i] + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[j])
                data_filename_SLN = data_folder + "/SLN_GC_%s" % constraints_names_short[i]  + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[j])
                data_filename_CC3 = data_folder + "/CC3_GC_%s" % constraints_names_short[i]  + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[j])
    
            np.savetxt(data_filename_LN, [mc_values, f_PBH_i_LN], delimiter="\t")
            np.savetxt(data_filename_SLN, [mc_values, f_PBH_i_SLN], delimiter="\t")
            np.savetxt(data_filename_CC3, [mc_values, f_PBH_i_CC3], delimiter="\t")

    if plot_extrapolated: 
        ax.set_xlim(1e11, 1e18)
        ax.set_ylim(10**(-14), 1)
        ax.set_xlabel("$M_\mathrm{PBH}~[\mathrm{g}]$")
        ax.set_ylabel("$f_\mathrm{PBH}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize="small")
        fig.tight_layout()
   

#%% Constraints from 2302.04408 (MW diffuse SPI with NFW template).

if "__main__" == __name__:
    # If True, use extrapolated delta-function MF constraints down to 1e11g (using a power law fit) to calculate extended MF constraint.
    include_extrapolated = False
    # If True, use extrapolated delta-function MF constraints down to 1e15g (using a power law fit) to calculate extended MF constraint.
    include_extrapolated_upper = True    
    # If True, plot extrapolated delta-function MF constraints down to 1e11g.
    plot_extrapolate = False
    # Boolean determines whether to use evolved mass function.
    evolved = True
    # Boolean determines whether to evaluate the evolved mass function at t=0.
    t_initial = False
    if t_initial:
        evolved = True
    
    t = t_0
    
    if not evolved:
        data_folder = "./Data-tests/unevolved"
    elif t_initial:
        data_folder = "./Data-tests/t_initial"
        t = 0
    else:
        data_folder = "./Data-tests"

    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    mc_values = np.logspace(14, 20, 120)
    
    # Load delta function MF constraints calculated using Isatis, to use the method from 1705.05567.
    m_delta_values, f_max = load_data("2302.04408/2302.04408_MW_diffuse_SPI.csv")
    
    if include_extrapolated:

        # Power-law exponent to use between 1e15g and 1e16g.
        exponent_PL_upper = 2.0
        # Power-law exponent to use between 1e11g and 1e15g.
        exponent_PL_lower = 4.0
        
        m_delta_extrapolated_upper = np.logspace(15, 16, 11)
        m_delta_extrapolated_lower = np.logspace(11, 15, 41)
        
        f_max_extrapolated_upper = min(f_max) * np.power(m_delta_extrapolated_upper / min(m_delta_values), exponent_PL_upper)
        f_max_extrapolated_lower = min(f_max_extrapolated_upper) * np.power(m_delta_extrapolated_lower / min(m_delta_extrapolated_upper), exponent_PL_lower)
    
        f_max_total = np.concatenate((f_max_extrapolated_lower, f_max_extrapolated_upper, f_max))
        m_delta_total = np.concatenate((m_delta_extrapolated_lower, m_delta_extrapolated_upper, m_delta_values))
    
        data_folder += "/PL_exp_{:.0f}".format(exponent_PL_lower)

    elif include_extrapolated_upper:

        # Power-law exponent to use between 1e15g and 1e16g.
        exponent_PL_upper = 2.0
        
        m_delta_extrapolated_upper = np.logspace(15, 16, 11)
        
        f_max_extrapolated_upper = min(f_max) * np.power(m_delta_extrapolated_upper / min(m_delta_values), exponent_PL_upper)
    
        f_max_total = np.concatenate((f_max_extrapolated_upper, f_max))
        m_delta_total = np.concatenate((m_delta_extrapolated_upper, m_delta_values))
    
        data_folder += "/upper_PL_exp_{:.0f}".format(exponent_PL_lower)

           
    else:
        f_max_total = f_max
        m_delta_total = m_delta_values
   
    
    for j in range(len(Deltas)):                
        if include_extrapolated:                     
            data_filename_LN = data_folder + "/LN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
            data_filename_SLN = data_folder + "/SLN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
            data_filename_CC3 = data_folder + "/CC3_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)

        else:          
            data_filename_LN = data_folder + "/LN_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[j])
            data_filename_SLN = data_folder + "/SLN_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[j])
            data_filename_CC3 = data_folder + "/CC3_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[j])
            
        params_LN = [sigmas_LN[j]]
        params_SLN = [sigmas_SLN[j], alphas_SLN[j]]
        params_CC3 = [alphas_CC3[j], betas[j]]
        
        f_pbh_LN = constraint_Carr(mc_values, m_delta_total, f_max_total, LN, params_LN, evolved, t)
        f_pbh_SLN = constraint_Carr(mc_values, m_delta_total, f_max_total, SLN, params_SLN, evolved, t)
        f_pbh_CC3 = constraint_Carr(mc_values, m_delta_total, f_max_total, CC3, params_CC3, evolved, t)
        
        np.savetxt(data_filename_LN, [mc_values, f_pbh_LN], delimiter="\t")                          
        np.savetxt(data_filename_SLN, [mc_values, f_pbh_SLN], delimiter="\t")
        np.savetxt(data_filename_CC3, [mc_values, f_pbh_CC3], delimiter="\t")



#%% Constraints from 2202.07483 (SPI constraints using NFW template).

if "__main__" == __name__:
    # If True, use extrapolated delta-function MF constraints down to 1e11g (using a power law fit) to calculate extended MF constraint.
    include_extrapolated = True
    # If True, plot extrapolated delta-function MF constraints down to 1e11g.
    plot_extrapolate = False
    # Boolean determines whether to use evolved mass function.
    evolved = True
    # Boolean determines whether to evaluate the evolved mass function at t=0.
    t_initial = False
    if t_initial:
        evolved = True
    
    t = t_0
    
    if not evolved:
        data_folder = "./Data-tests/unevolved"
    elif t_initial:
        data_folder = "./Data-tests/t_initial"
        t = 0
    else:
        data_folder = "./Data-tests"

    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    mc_values = np.logspace(14, 20, 120)
    
    # Load delta function MF constraints calculated using Isatis, to use the method from 1705.05567.
    m_delta_values, f_max = load_data("2202.07483/2202.07483_Fig3.csv")
    
    if include_extrapolated:

        # Power-law exponent to use between 1e11g and 3e16g.
        exponent_PL_lower = 4.0
        
        m_delta_extrapolated = 10**np.arange(11, np.log10(min(m_delta_values))+0.01, 0.1)
        f_max_extrapolated = min(f_max) * np.power(m_delta_extrapolated / min(m_delta_values), exponent_PL_lower)
    
        f_max_total = np.concatenate((f_max_extrapolated, f_max))
        m_delta_total = np.concatenate((m_delta_extrapolated, m_delta_values))
    
        data_folder += "/PL_exp_{:.0f}".format(exponent_PL_lower)

           
    else:
        f_max_total = f_max
        m_delta_total = m_delta_values
    
    for j in range(len(Deltas)):                
        if include_extrapolated:                     
            data_filename_LN = data_folder + "/LN_2202.07483_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
            data_filename_SLN = data_folder + "/SLN_2202.07483_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
            data_filename_CC3 = data_folder + "/CC3_2202.07483_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)

        else:          
            data_filename_LN = data_folder + "/LN_2202.07483_Carr_Delta={:.1f}.txt".format(Deltas[j])
            data_filename_SLN = data_folder + "/SLN_2202.07483_Carr_Delta={:.1f}.txt".format(Deltas[j])
            data_filename_CC3 = data_folder + "/CC3_2202.07483_Carr_Delta={:.1f}.txt".format(Deltas[j])
            
        params_LN = [sigmas_LN[j]]
        params_SLN = [sigmas_SLN[j], alphas_SLN[j]]
        params_CC3 = [alphas_CC3[j], betas[j]]
        
        f_pbh_LN = constraint_Carr(mc_values, m_delta_total, f_max_total, LN, params_LN, evolved, t)
        f_pbh_SLN = constraint_Carr(mc_values, m_delta_total, f_max_total, SLN, params_SLN, evolved, t)
        f_pbh_CC3 = constraint_Carr(mc_values, m_delta_total, f_max_total, CC3, params_CC3, evolved, t)
        
        np.savetxt(data_filename_LN, [mc_values, f_pbh_LN], delimiter="\t")                          
        np.savetxt(data_filename_SLN, [mc_values, f_pbh_SLN], delimiter="\t")
        np.savetxt(data_filename_CC3, [mc_values, f_pbh_CC3], delimiter="\t")


#%% Prospective constraints from 2101.01370 (GECCO).

if "__main__" == __name__:
    # If True, use extrapolated delta-function MF constraints down to 1e11g (using a power law fit) to calculate extended MF constraint
    include_extrapolated = True
    # If True, plot extrapolated delta-function MF constraints down to 1e11g
    evolved = True
    # Boolean determines whether to evaluate the evolved mass function at t=0.
    t_initial = False
    if t_initial:
        evolved = True
    # If True, plot the projected constraints for an NFW profile. If False, plot the projected constraints for an Einasto profile
    NFW = False
    
    t = t_0
    
    if not evolved:
        data_folder = "./Data-tests/unevolved"
    elif t_initial:
        data_folder = "./Data-tests/t_initial"
        t = 0
    else:
        data_folder = "./Data-tests"

    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    mc_values = np.logspace(14, 21, 130)
    
    # Load delta function MF constraints calculated using Isatis, to use the method from 1705.05567.
    if NFW:
        m_delta_values, f_max = load_data("2101.01370/2101.01370_Fig9_GC_NFW.csv")
        profile_string = "NFW"
        
    else:
        m_delta_values, f_max = load_data("2101.01370/2101.01370_Fig9_GC_Einasto.csv")
        profile_string = "Einasto"
    
    if include_extrapolated:

        # Power-law exponent to use between 1e11g and 1e15g.
        exponent_PL_lower = 0.0
        
        m_delta_extrapolated = np.logspace(11, 15, 41)
        f_max_extrapolated = min(f_max) * np.power(m_delta_extrapolated / min(m_delta_values), exponent_PL_lower)
    
        f_max_total = np.concatenate((f_max_extrapolated, f_max))
        m_delta_total = np.concatenate((m_delta_extrapolated, m_delta_values))
        data_folder += "/PL_exp_{:.0f}".format(exponent_PL_lower)
    
    else:
        f_max_total = f_max
        m_delta_total = m_delta_values


    for j in range(len(Deltas)):                
        if include_extrapolated:                     
            data_filename_LN = data_folder + "/LN_2101.01370_Carr_Delta={:.1f}_".format(Deltas[j]) + "%s_" % profile_string + "extrapolated_exp{:.0f}.txt".format(exponent_PL_lower)
            data_filename_SLN = data_folder + "/SLN_2101.01370_Carr_Delta={:.1f}_".format(Deltas[j]) + "%s_" % profile_string + "extrapolated_exp{:.0f}.txt".format( exponent_PL_lower)
            data_filename_CC3 = data_folder + "/CC3_2101.01370_Carr_Delta={:.1f}_".format(Deltas[j]) + "%s_" % profile_string + "extrapolated_exp{:.0f}.txt".format(exponent_PL_lower)
                                  
        else:           
            data_filename_LN = data_folder + "/LN_2101.01370_Carr_Delta={:.1f}_".format(Deltas[j]) + "%s_" % profile_string + ".txt"
            data_filename_SLN = data_folder + "/SLN_2101.01370_Carr_Delta={:.1f}_".format(Deltas[j]) + "%s_" % profile_string + ".txt"
            data_filename_CC3 = data_folder + "/CC3_2101.01370_Carr_Delta={:.1f}_".format(Deltas[j]) + "%s_" % profile_string + ".txt"
            
        params_LN = [sigmas_LN[j]]
        params_SLN = [sigmas_SLN[j], alphas_SLN[j]]
        params_CC3 = [alphas_CC3[j], betas[j]]
        
        f_pbh_LN = constraint_Carr(mc_values, m_delta_total, f_max_total, LN, params_LN, evolved, t)
        f_pbh_SLN = constraint_Carr(mc_values, m_delta_total, f_max_total, SLN, params_SLN, evolved, t)
        f_pbh_CC3 = constraint_Carr(mc_values, m_delta_total, f_max_total, CC3, params_CC3, evolved, t)
        
        np.savetxt(data_filename_LN, [mc_values, f_pbh_LN], delimiter="\t")                          
        np.savetxt(data_filename_SLN, [mc_values, f_pbh_SLN], delimiter="\t")
        np.savetxt(data_filename_CC3, [mc_values, f_pbh_CC3], delimiter="\t")



#%% Constraints from 2302.04408 (MW diffuse SPI with NFW template).
# Use for convergence tests with number of masses in preliminaries.py and range of masses included in power-law extrapolation of the delta-function MF constraint.
# Consider the evolved MFs only, with PL extrapolation.

if "__main__" == __name__:
    
    t = t_0
    evolved = True

    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    mc_values = np.logspace(14, 20, 120)
    
    # Load delta function MF constraints calculated using Isatis, to use the method from 1705.05567.
    m_delta_values, f_max = load_data("2302.04408/2302.04408_MW_diffuse_SPI.csv")
    
    # Power-law exponent to use between 1e15g and 1e16g.
    exponent_PL_upper = 2.0
    # Power-law exponent to use between 1e11g and 1e15g.
    exponent_PL_lower = 2.0
    
    log_m_min_values = [7, 9, 11]
    n_steps_values = [1000, 10000, 100000]

    for log_m_min in log_m_min_values:
    
        m_delta_extrapolated_upper = np.logspace(15, 16, 11)
        #m_delta_extrapolated_lower = np.logspace(11, 15, 41)
        m_delta_extrapolated_lower = 10**np.arange(log_m_min, 15, 0.1)
        
        f_max_extrapolated_upper = min(f_max) * np.power(m_delta_extrapolated_upper / min(m_delta_values), exponent_PL_upper)
        f_max_extrapolated_lower = min(f_max_extrapolated_upper) * np.power(m_delta_extrapolated_lower / min(m_delta_extrapolated_upper), exponent_PL_lower)
    
        f_max_total = np.concatenate((f_max_extrapolated_lower, f_max_extrapolated_upper, f_max))
        m_delta_total = np.concatenate((m_delta_extrapolated_lower, m_delta_extrapolated_upper, m_delta_values))
     
        for n_steps in n_steps_values:
            
            data_folder = "./Data-tests/PL_exp_{:.0f}/log_m_min={:.0f}/log_n_steps={:.0f}".format(exponent_PL_lower, log_m_min, np.log10(n_steps))
        
            for j in range(len(Deltas)):                
                data_filename_LN = data_folder + "/LN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
                data_filename_SLN = data_folder + "/SLN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
                data_filename_CC3 = data_folder + "/CC3_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
                    
                params_LN = [sigmas_LN[j]]
                params_SLN = [sigmas_SLN[j], alphas_SLN[j]]
                params_CC3 = [alphas_CC3[j], betas[j]]
                
                f_pbh_LN = constraint_Carr(mc_values, m_delta_total, f_max_total, LN, params_LN, evolved, t, n_steps)
                f_pbh_SLN = constraint_Carr(mc_values, m_delta_total, f_max_total, SLN, params_SLN, evolved, t, n_steps)
                f_pbh_CC3 = constraint_Carr(mc_values, m_delta_total, f_max_total, CC3, params_CC3, evolved, t, n_steps)
                
                np.savetxt(data_filename_LN, [mc_values, f_pbh_LN], delimiter="\t")                          
                np.savetxt(data_filename_SLN, [mc_values, f_pbh_SLN], delimiter="\t")
                np.savetxt(data_filename_CC3, [mc_values, f_pbh_CC3], delimiter="\t")


#%% Constraints from COMPTEL, INTEGRAL, EGRET and Fermi-LAT. Approximate results obtained by using f_max as the constraint from each instrument, rather than the minimum over each energy bin.
# Consider the results obtained using a power-law extrapolation below 1e15g (see https://www.evernote.com/shard/s463/nl/233603521/c2003730-1f06-3c27-6d59-75fad354ef79?title=Evolved%20MF%20constraints:%20why%20are%20evolved%20MF%20constraints%20weaker%20for%20Delta%20=%205%20but%20not%20smaller%20Delta?).

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    # Boolean determines whether to useFalse evolved mass function.
    evolved = True
    include_extrapolated = True
    
    m_delta_values_loaded = np.logspace(11, 21, 1000)
    constraints_names, f_max_Isatis = load_results_Isatis(modified=True)
    colors_evap = ["tab:orange", "tab:green", "tab:red", "tab:blue"]
    constraints_names_short = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
    
    mc_values = np.logspace(14, 20, 120)
    
    t = t_0

    if include_extrapolated:
        # Power-law exponent to use
        exponent_PL_lower = 4.0
        m_delta_extrapolated = 10**np.arange(11, 15, 0.1)
        data_folder = "./Data-tests/PL_exp_{:.0f}/".format(exponent_PL_lower)
        
    else:
        data_folder = "./Data-tests/"

    constraints_names_short = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]

    for j in range(len(Deltas)):
        params_LN = [sigmas_LN[j]]
        params_SLN = [sigmas_SLN[j], alphas_SLN[j]]
        params_CC3 = [alphas_CC3[j], betas[j]]
        
        # Using each energy bin per instrument individually for the delta-function MF constraint, then obtaining the tightest constraint from each instrument using envelope().
        mc_constraints_LN = []
        mc_constraints_SLN = []
        mc_constraints_CC3 = []

        for i in range(len(constraints_names_short)):
            # Calculate constraint using method from 1705.05567.

            f_max_loaded = np.array(f_max_Isatis[i])
            
            if include_extrapolated:
                f_max_loaded_truncated = f_max_loaded[m_delta_values_loaded > 1e15]
                f_max_extrapolated = f_max_loaded_truncated[0] * np.power(m_delta_extrapolated / 1e15, exponent_PL_lower)
                f_max_i = np.concatenate((f_max_extrapolated, f_max_loaded_truncated))
                m_delta_values = np.concatenate((m_delta_extrapolated, m_delta_values_loaded[m_delta_values_loaded > 1e15]))
            else:
                f_max_i = f_max_loaded
                m_delta_values = m_delta_values_loaded
                        
            f_PBH_i_LN = constraint_Carr(mc_values, m_delta_values, f_max_i, LN, params_LN, evolved, t)
            f_PBH_i_SLN = constraint_Carr(mc_values, m_delta_values, f_max_i, SLN, params_SLN, evolved, t)
            f_PBH_i_CC3 = constraint_Carr(mc_values, m_delta_values, f_max_i, CC3, params_CC3, evolved, t)
      
            data_filename_LN = data_folder + "/LN_GC_%s" % constraints_names_short[i] + "_Carr_Delta={:.1f}_approx_mmin=1e15g.txt".format(Deltas[j])
            data_filename_SLN = data_folder + "/SLN_GC_%s" % constraints_names_short[i]  + "_Carr_Delta={:.1f}_approx_mmin=1e15g.txt".format(Deltas[j])
            data_filename_CC3 = data_folder + "/CC3_GC_%s" % constraints_names_short[i]  + "_Carr_Delta={:.1f}_approx_mmin=1e15g.txt".format(Deltas[j])
    
            np.savetxt(data_filename_LN, [mc_values, f_PBH_i_LN], delimiter="\t")
            np.savetxt(data_filename_SLN, [mc_values, f_PBH_i_SLN], delimiter="\t")
            np.savetxt(data_filename_CC3, [mc_values, f_PBH_i_CC3], delimiter="\t")

