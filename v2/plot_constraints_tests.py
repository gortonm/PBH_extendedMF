#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:39:43 2023

@author: ppxmg2
"""

# Script plots results obtained for the extended PBH mass functions given by
# the fitting functions in 2009.03204.
# Compares a number of different constraints obtained using the extended
# PBH mass function.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from preliminaries import load_data, m_max_SLN, load_results_Isatis, envelope, LN, SLN, CC3
from plot_constraints import plotter_GC_Isatis

# Specify the plot style
mpl.rcParams.update({'font.size': 20, 'font.family':'serif'})
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


def plot_KP23(j, ax1, ax2, ax3, exponent_PL_lower=2):
    """
    Plot the evolved MF constraints from Korwar & Profumo (2023)

    Parameters
    ----------
    j : Integer
        Index indicating which value of Delta to plot constraints for.
    ax1 : Matplotlib Axes object
        Axis on which to plot the LN MF constraint.
    ax2 : Matplotlib Axes object
        Axis on which to plot the SLN MF constraint.
    ax3 : Matplotlib Axes object
        Axis on which to plot the CC3 MF constraint.
    exponent_PL_lower : Integer, optional
        Power-law exponent to extrapolate the delta-function constraint below 1e15g. The default is 2.

    Returns
    -------
    None.

    """
    # Path to extended MF constraints
    data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower) 
            
    data_filename_LN = data_folder + "/LN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
    data_filename_SLN = data_folder + "/SLN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
    data_filename_CC3 = data_folder + "/CC3_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
    
    mc_KP23_LN, f_PBH_KP23_LN = np.genfromtxt(data_filename_LN, delimiter="\t")
    mc_KP23_SLN, f_PBH_KP23_SLN = np.genfromtxt(data_filename_SLN, delimiter="\t")
    mp_KP23_CC3, f_PBH_KP23_CC3 = np.genfromtxt(data_filename_CC3, delimiter="\t")
 
    # Peak mass for log-normal MF
    mp_LN = mc_KP23_LN * np.exp(-sigmas_LN[j]**2)               
    # Estimate peak mass of skew-lognormal MF
    mp_KP23_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[j], alpha=alphas_SLN[j], log_m_factor=3, n_steps=1000) for m_c in mc_KP23_SLN]
 
    ax1.plot(mp_LN, f_PBH_KP23_LN, dashes=[6, 2], color="tab:grey")
    ax2.plot(mp_KP23_SLN, f_PBH_KP23_SLN, color="tab:grey", linestyle=(0, (5, 7)))
    ax3.plot(mp_KP23_CC3, f_PBH_KP23_CC3, color="tab:grey", linestyle="dashed")


#%% Tests evaluating the mass functions at the initial time (or unevolved mass functions), and comparing to results obtained before June 2023.

if "__main__" == __name__:
        
    # Choose colors to match those from Fig. 5 of 2009.03204
    colors = ['silver', 'r', 'b', 'g', 'k']
                    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
        
    for i in range(len(Deltas)):
                
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
        ax0 = axes[0]
        ax1 = axes[1]
        ax2 = axes[2]
        
        # Loading constraints from Subaru-HSC.
        mc_Carr_SLN, f_PBH_Carr_SLN = np.genfromtxt("./Data-old/SLN_HSC_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
        mp_Subaru_CC3, f_PBH_Carr_CC3 = np.genfromtxt("./Data-old/CC3_HSC_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
        mc_Carr_LN, f_PBH_Carr_LN = np.genfromtxt("./Data-old/LN_HSC_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
        
        fig.suptitle("Using 511 keV line constraints (Korwar \& Profumo 2023), $\Delta={:.1f}$".format(Deltas[i]))
        
        
        # Delta-function MF constraints
        m_delta_evap, f_PBH_delta_evap = load_data("2302.04408/2302.04408_MW_diffuse_SPI.csv")
        m_delta_Subaru, f_PBH_delta_Subaru = load_data("2007.12697/Subaru-HSC_2007.12697_dx=5.csv")
        
        for ax in [ax0, ax1, ax2]:
            ax.set_xlabel("$m_p~[\mathrm{g}]$")
            ax.plot(m_delta_evap, f_PBH_delta_evap, color=colors[0], label="Delta function", linewidth=2)
            ax.plot(m_delta_Subaru, f_PBH_delta_Subaru, color=colors[0], linewidth=2)
            ax.set_ylabel("$f_\mathrm{PBH}$")
            ax.set_xscale("log")
            ax.set_yscale("log")
            

        # Load constraints from Galactic Centre 511 keV line emission (from 2302.04408).            
        mc_KP23_SLN, f_PBH_KP23_SLN = np.genfromtxt("./Data-old/SLN_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
        mp_KP23_CC3, f_PBH_KP23_CC3 = np.genfromtxt("./Data-old/CC3_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
        mc_KP23_LN, f_PBH_KP23_LN = np.genfromtxt("./Data-old/LN_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")

        mc_KP23_SLN_t_init, f_PBH_KP23_SLN_t_init = np.genfromtxt("./Data-tests/t_initial/SLN_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
        mp_KP23_CC3_t_init, f_PBH_KP23_CC3_t_init = np.genfromtxt("./Data-tests/t_initial/CC3_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
        mc_KP23_LN_t_init, f_PBH_KP23_LN_t_init = np.genfromtxt("./Data-tests/t_initial/LN_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
        
        mc_KP23_SLN_unevolved, f_PBH_KP23_SLN_unevolved = np.genfromtxt("./Data-tests/unevolved/SLN_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
        mp_KP23_CC3_unevolved, f_PBH_KP23_CC3_unevolved = np.genfromtxt("./Data-tests/unevolved/CC3_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
        mc_KP23_LN_unevolved, f_PBH_KP23_LN_unevolved = np.genfromtxt("./Data-tests/unevolved/LN_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")


        # Estimate peak mass of skew-lognormal MF
        mp_KP23_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_KP23_SLN]
        mp_KP23_SLN_t_init = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_KP23_SLN_t_init]
        mp_KP23_SLN_unevolved = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_KP23_SLN_unevolved]

        mp_Subaru_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_Carr_SLN]
        
        ax0.plot(mp_KP23_SLN, f_PBH_KP23_SLN, color=colors[2], linestyle=(0, (5, 7)))
        ax0.plot(mp_KP23_CC3, f_PBH_KP23_CC3, color=colors[3], linestyle="dashed")
        ax1.plot(mp_KP23_SLN, f_PBH_KP23_SLN, color=colors[2], linestyle=(0, (5, 7)))
        ax1.plot(mp_KP23_CC3, f_PBH_KP23_CC3, color=colors[3], linestyle="dashed")
        ax0.plot(mc_KP23_LN * np.exp(-sigmas_LN[i]**2), f_PBH_KP23_LN, color=colors[1], dashes=[6, 2], label="LN")
        ax1.plot(mc_KP23_LN * np.exp(-sigmas_LN[i]**2), f_PBH_KP23_LN, color=colors[1], dashes=[6, 2])

        ax0.plot(mp_KP23_SLN_t_init, f_PBH_KP23_SLN_t_init, color=colors[2], linestyle="None", marker="x")
        ax0.plot(mp_KP23_CC3_t_init, f_PBH_KP23_CC3_t_init, color=colors[3], linestyle="None", marker="x")
        ax1.plot(mp_KP23_SLN_t_init, f_PBH_KP23_SLN_t_init, color=colors[2], linestyle="None", marker="x")
        ax1.plot(mp_KP23_CC3_t_init, f_PBH_KP23_CC3_t_init, color=colors[3], linestyle="None", marker="x")
        ax0.plot(mc_KP23_LN_t_init * np.exp(-sigmas_LN[i]**2), f_PBH_KP23_LN_t_init, color=colors[1], linestyle="None", marker="x")
        ax1.plot(mc_KP23_LN_t_init * np.exp(-sigmas_LN[i]**2), f_PBH_KP23_LN_t_init, color=colors[1], linestyle="None", marker="x")

        ax0.plot(mp_KP23_SLN_unevolved, f_PBH_KP23_SLN_unevolved, color=colors[2], linestyle="None", marker="+")
        ax0.plot(mp_KP23_CC3_unevolved, f_PBH_KP23_CC3_unevolved, color=colors[3], linestyle="None", marker="+")
        ax1.plot(mp_KP23_SLN_unevolved, f_PBH_KP23_SLN_unevolved, color=colors[2], linestyle="None", marker="+")
        ax1.plot(mp_KP23_CC3_unevolved, f_PBH_KP23_CC3_unevolved, color=colors[3], linestyle="None", marker="+")
        ax0.plot(mc_KP23_LN_unevolved * np.exp(-sigmas_LN[i]**2), f_PBH_KP23_LN_unevolved, color=colors[1], linestyle="None", marker="+")
        ax1.plot(mc_KP23_LN_unevolved * np.exp(-sigmas_LN[i]**2), f_PBH_KP23_LN_unevolved, color=colors[1], linestyle="None", marker="+")

        ax0.plot(0, 0, linestyle="None", color="k", marker="x", label="Test: $t=0$")
        ax0.plot(0, 0, linestyle="None", color="k", marker="+", label="Test: unevolved")
       
        mc_Carr_LN, f_PBH_Carr_LN = np.genfromtxt("./Data-old/LN_HSC_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")

        # Set axis limits
        if Deltas[i] < 5:
            xmin_HSC, xmax_HSC = 1e21, 1e29
            xmin_evap, xmax_evap = 1e16, 7e17
            ymin, ymax = 1e-5, 1
        
        else:
            xmin_HSC, xmax_HSC = 9e18, 1e29
            xmin_evap, xmax_evap = 1e16, 2e18
            ymin, ymax = 1e-5, 1

        ax0.plot(mp_Subaru_SLN, f_PBH_Carr_SLN, color=colors[2], label="SLN", linestyle=(0, (5, 7)))
        ax0.plot(mp_Subaru_CC3, f_PBH_Carr_CC3, color=colors[3], label="CC3", linestyle="dashed")
        ax0.plot(mc_Carr_LN * np.exp(-sigmas_LN[i]**2), f_PBH_Carr_LN, color=colors[1], dashes=[6, 2])
        ax2.plot(mp_Subaru_SLN, f_PBH_Carr_SLN, color=colors[2], label="SLN", linestyle=(0, (5, 7)))
        ax2.plot(mp_Subaru_CC3, f_PBH_Carr_CC3, color=colors[3], label="CC3", linestyle="dashed")
        ax2.plot(mc_Carr_LN * np.exp(-sigmas_LN[i]**2), f_PBH_Carr_LN, color=colors[1], dashes=[6, 2])
                  
        ax0.set_xlim(xmin_evap, xmax_HSC)
        ax0.set_ylim(ymin, ymax)
        ax1.set_xlim(xmin_evap, xmax_evap)
        ax1.set_ylim(ymin, ymax)
        ax2.set_xlim(xmin_HSC, xmax_HSC)
        ax2.set_ylim(4e-3, 1)
        ax0.legend(fontsize="xx-small")
        fig.tight_layout()
        
#%% Plot the Galactic Centre photon constraints for an unevolved MF (and no power-law extrapolation in the delta-function MF constraint below 1e13g), calculated as the minimum over each energy bin
# Compare results to those obtained before June 2023.

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)

    mc_values_new = np.logspace(14, 20, 120)
    mc_values_old = np.logspace(14, 19, 100)
   
    colors_evap = ["tab:orange", "tab:green", "tab:red", "tab:blue"]
    constraints_names_short = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
    
    # Folder that new results are stored in.
    data_folder = "./Data-tests/unevolved"
    
    # Booleans control which mass function to load constraints from:
    plot_LN = False
    plot_SLN = False
    plot_CC3 = True
    
    if plot_LN:
        mf_string_old = mf_string_new = "LN"
    elif plot_SLN:
        mf_string_old = "SL"
        mf_string_new = "SLN"
    elif plot_CC3:
        mf_string_old = "CC"
        mf_string_new = "CC3"

    for i in range(len(Deltas)):
        fig, ax = plt.subplots(figsize=(6,6))
        
        mc_values_old = np.logspace(14, 19, 100)          
        fname_base = "CC_D={:.1f}_test_range_wide".format(Deltas[i])    # compare to old constraints obtained using the same PBH mass range and number of masses as those from before June 2023
        constraints_names, f_PBHs_GC_old = load_results_Isatis(mf_string=fname_base, modified=True)

        for j in range(len(constraints_names_short)):
            
            data_filename = data_folder + "/%s_GC_%s" % (mf_string_new, constraints_names_short[j]) + "_Carr_Delta={:.1f}_unevolved.txt".format(Deltas[i])

            mc_new, f_PBH_new = np.genfromtxt(data_filename)
 
            ax.plot(mc_new, f_PBH_new, color=colors_evap[j], linestyle="None", marker="x", alpha=0.5)
            ax.plot(mc_values_old, f_PBHs_GC_old[j], color=colors_evap[j])
        
        if plot_CC3:
            ax.set_xlabel("$m_p~[\mathrm{g}]$")
        else:
            ax.set_xlabel("$m_c~[\mathrm{g}]$")
        ax.set_ylabel("$f_\mathrm{PBH}$")
        ax.set_title("$\Delta={:.1f}$".format(Deltas[i]) + ", %s" % mf_string_new)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e14, 1e18)
        ax.set_ylim(1e-10, 1)
        plt.tight_layout()

#%% Tests of the results obtained using different power-law exponents in f_max at low masses (Korwar & Profumo (2023) [2302.04408])

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    mc_values = np.logspace(14, 20, 120)

    # Array of power law exponents to use at masses below 1e15g
    exponents_PL_lower = [0, 2, 3, 4]
    
    linestyles = ["solid", "dashed", "dashdot", "dotted"]
    
    for j in range(len(Deltas)):
        
        fig, axes = plt.subplots(2, 2, figsize=(13, 13))
        
        ax0 = axes[0][0]
        ax1 = axes[0][1]
        ax2 = axes[1][0]
        ax3 = axes[1][1]
        
        m_delta_values, f_max = load_data("2302.04408/2302.04408_MW_diffuse_SPI.csv")
                            
        # Power-law exponent to use between 1e15g and 1e16g (motivated by mass-dependence of the positron spectrum emitted over energy)
        exponent_PL_upper = 2.0
        
        m_delta_extrapolated_upper = np.logspace(15, 16, 11)
        m_delta_extrapolated_lower = np.logspace(11, 15, 41)
        f_max_extrapolated_upper = min(f_max) * np.power(m_delta_extrapolated_upper / min(m_delta_values), exponent_PL_upper)
            
        for k, exponent_PL_lower in enumerate(exponents_PL_lower):
            
            if k == 0:
                data_folder = "./Data-tests/upper_PL_exp_2"
                data_filename_LN = data_folder + "/LN_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[j])
                data_filename_SLN = data_folder + "/SLN_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[j])
                data_filename_CC3 = data_folder + "/CC3_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[j])
                """
                mc_KP23_SLN_unevolved, f_PBH_KP23_SLN_unevolved = np.genfromtxt("./Data-tests/unevolved/SLN_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[j]), delimiter="\t")
                mp_KP23_CC3_unevolved, f_PBH_KP23_CC3_unevolved = np.genfromtxt("./Data-tests/unevolved/CC3_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[j]), delimiter="\t")
                mc_KP23_LN_unevolved, f_PBH_KP23_LN_unevolved = np.genfromtxt("./Data-tests/unevolved/LN_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[j]), delimiter="\t")
                """
            else:
                data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower) 
                data_filename_LN = data_folder + "/LN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
                data_filename_SLN = data_folder + "/SLN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
                data_filename_CC3 = data_folder + "/CC3_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
            
            mc_KP23_LN_evolved, f_PBH_KP23_LN_evolved = np.genfromtxt(data_filename_LN, delimiter="\t")
            mc_KP23_SLN_evolved, f_PBH_KP23_SLN_evolved = np.genfromtxt(data_filename_SLN, delimiter="\t")
            mp_KP23_CC3_evolved, f_PBH_KP23_CC3_evolved = np.genfromtxt(data_filename_CC3, delimiter="\t")
            mp_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[j], alpha=alphas_SLN[j], log_m_factor=3, n_steps=1000) for m_c in mc_KP23_SLN_evolved]
            mp_LN = mc_KP23_LN_evolved * np.exp(-sigmas_LN[j]**2)
                        
            ax1.plot(mp_LN, f_PBH_KP23_LN_evolved, linestyle=linestyles[k], color="r", marker="None")
            ax2.plot(mp_SLN, f_PBH_KP23_SLN_evolved, linestyle=linestyles[k], color="b", marker="None")
            ax3.plot(mp_KP23_CC3_evolved, f_PBH_KP23_CC3_evolved, linestyle=linestyles[k], color="g", marker="None")
            
            ax1.set_title("LN")
            ax2.set_title("SLN")
            ax3.set_title("CC3")
            
            if k == 0:
                """
                mp_SLN_unevolved = [m_max_SLN(m_c, sigma=sigmas_SLN[j], alpha=alphas_SLN[j], log_m_factor=3, n_steps=1000) for m_c in mc_KP23_SLN_unevolved]
                mp_LN_unevolved = mc_KP23_LN_unevolved * np.exp(-sigmas_LN[j]**2)
                ax1.plot(mp_LN, f_PBH_KP23_LN_unevolved, color="r", linestyle="None", marker="x")
                ax2.plot(mp_SLN_unevolved, f_PBH_KP23_SLN_unevolved, color="b", linestyle="None", marker="x")
                ax3.plot(mp_KP23_CC3_unevolved, f_PBH_KP23_CC3_unevolved, color="g", linestyle="None", marker="x")
                ax1.plot(0, 0, color="k", label="No constraint at $m < 10^{16}~\mathrm{g}$ [evolved MF]")
                ax1.plot(0, 0, linestyle="None", marker="x", color="k", label="No constraint at $m < 10^{16}~\mathrm{g}$ [unevolved MF]")
                """
            else:
                ax1.plot(0, 0, marker="None", linestyle=linestyles[k], color="k", label="{:.0f}".format(exponent_PL_lower))
                f_max_extrapolated_lower = min(f_max_extrapolated_upper) * np.power(m_delta_extrapolated_lower / min(m_delta_extrapolated_upper), exponent_PL_lower)
                ax0.plot(m_delta_extrapolated_lower, f_max_extrapolated_lower, color=(0.5294, 0.3546, 0.7020), linestyle=linestyles[k], label="{:.0f}".format(exponent_PL_lower))
            
        ax0.plot(np.concatenate((m_delta_extrapolated_upper, m_delta_values)), np.concatenate((f_max_extrapolated_upper, f_max)), color=(0.5294, 0.3546, 0.7020))
        ax0.set_xlabel("$m$ [g]")
        ax0.set_ylabel("$f_\mathrm{max}$")
        ax0.set_xscale("log")
        ax0.set_yscale("log")
        ax0.set_xlim(min(m_delta_extrapolated_lower), 1e18)
        ax0.set_ylim(min(f_max_extrapolated_lower), 1)
                
        for ax in [ax1, ax2, ax3]:
            if Deltas[j] < 5:
                ax.set_xlim(1e16, 1e18)
            else:
                ax.set_xlim(1e16, 2e18)
               
            ax.set_ylim(1e-6, 1)
            ax.set_ylabel("$f_\mathrm{PBH}$")
            ax.set_xlabel("$m_p~[\mathrm{g}]$")
            ax.set_xscale("log")
            ax.set_yscale("log")

        ax0.legend(fontsize="x-small", title="PL exponent in $f_\mathrm{max}$ \n ($m < 10^{15}~\mathrm{g}$)")        
        ax1.legend(fontsize="x-small", title="PL exponent in $f_\mathrm{max}$ \n ($m < 10^{15}~\mathrm{g}$)")
        fig.suptitle("[2302.04408], $\Delta={:.1f}$".format(Deltas[j]))
        fig.tight_layout()

#%% Tests of the results obtained using different power-law exponents in f_max at low masses (Berteaud et al. (2022) [2202.07483])

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    mc_values = np.logspace(14, 20, 120)

    # Array of power law exponents to use at masses below 1e15g
    exponents_PL_lower = [0, 2, 3, 4]
    
    linestyles = ["solid", "dashed", "dashdot", "dotted"]
    
    m_delta_values, f_max = load_data("2202.07483/2202.07483_Fig3.csv")
    m_delta_extrapolated = 10**np.arange(11, np.log10(min(m_delta_values))+0.01, 0.1)

    for j in range(len(Deltas)):
        
        fig, axes = plt.subplots(2, 2, figsize=(13, 13))
        
        ax0 = axes[0][0]
        ax1 = axes[0][1]
        ax2 = axes[1][0]
        ax3 = axes[1][1]
                        
        for k, exponent_PL_lower in enumerate(exponents_PL_lower):
            
            data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower) 
            data_filename_LN = data_folder + "/LN_2202.07483_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
            data_filename_SLN = data_folder + "/SLN_2202.07483_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
            data_filename_CC3 = data_folder + "/CC3_2202.07483_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
            
            mc_KP23_LN_evolved, f_PBH_KP23_LN_evolved = np.genfromtxt(data_filename_LN, delimiter="\t")
            mc_KP23_SLN_evolved, f_PBH_KP23_SLN_evolved = np.genfromtxt(data_filename_SLN, delimiter="\t")
            mp_KP23_CC3_evolved, f_PBH_KP23_CC3_evolved = np.genfromtxt(data_filename_CC3, delimiter="\t")
            mp_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[j], alpha=alphas_SLN[j], log_m_factor=3, n_steps=1000) for m_c in mc_KP23_SLN_evolved]
            mp_LN = mc_KP23_LN_evolved * np.exp(-sigmas_LN[j]**2)
                        
            ax1.plot(mp_LN, f_PBH_KP23_LN_evolved, linestyle=linestyles[k], color="r", marker="None")
            ax2.plot(mp_SLN, f_PBH_KP23_SLN_evolved, linestyle=linestyles[k], color="b", marker="None")
            ax3.plot(mp_KP23_CC3_evolved, f_PBH_KP23_CC3_evolved, linestyle=linestyles[k], color="g", marker="None")
            
            ax1.set_title("LN")
            ax2.set_title("SLN")
            ax3.set_title("CC3")
            
            ax1.plot(0, 0, marker="None", linestyle=linestyles[k], color="k", label="{:.0f}".format(exponent_PL_lower))
            f_max_extrapolated = min(f_max) * np.power(m_delta_extrapolated / min(m_delta_values), exponent_PL_lower)
            ax0.plot(m_delta_extrapolated, f_max_extrapolated, color="tab:grey", linestyle=linestyles[k], label="{:.0f}".format(exponent_PL_lower))
        
        # Plot extended MF constraints from Korwar & Profumo (2023)
        plot_KP23(j, ax1, ax2, ax3)

        ax0.plot(m_delta_values, f_max, color="tab:grey")
        ax0.set_xlabel("$m$ [g]")
        ax0.set_ylabel("$f_\mathrm{max}$")
        ax0.set_xscale("log")
        ax0.set_yscale("log")
        ax0.set_xlim(min(m_delta_extrapolated), 1e18)
        ax0.set_ylim(min(f_max_extrapolated), 1)
        
        for ax in [ax1, ax2, ax3]:
            if Deltas[j] < 5:
                ax.set_xlim(1e16, 1e18)
            else:
                ax.set_xlim(1e16, 2e18)
               
            ax.set_ylim(1e-6, 1)
            ax.set_ylabel("$f_\mathrm{PBH}$")
            ax.set_xlabel("$m_p~[\mathrm{g}]$")
            ax.set_xscale("log")
            ax.set_yscale("log")

        ax0.legend(fontsize="x-small", title="PL exponent in $f_\mathrm{max}$ \n ($m < 3e16~\mathrm{g}$)")        
        ax1.legend(fontsize="x-small", title="PL exponent in $f_\mathrm{max}$ \n ($m < 3e16~\mathrm{g}$)")
        fig.suptitle("Gamma rays [2202.07483], $\Delta={:.1f}$".format(Deltas[j]))
        fig.tight_layout()
     
        
#%% Tests of the results obtained using different power-law exponents in f_max at low masses 
# Constraints from 1807.03075 (Voyager-1 electron / positron detections).

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    mc_values = 10**np.arange(14, 20.5, 0.1)

    # Power-law exponent to use between 1e11g and the smallest mass the delta-function MF constraint is calculated for.   
    exponents_PL_lower = [0, 2, 3, 4]
    
    linestyles = ["solid", "dashed", "dashdot", "dotted"]
    
    # Boolean determines which propagation model to load data from
    prop_A = False
    prop_B = not prop_A
    
    with_bkg_subtr = False
    
    # If True, load the more stringent or less stringent "prop B" data
    prop_B_lower = False
    
    if prop_A:
        prop_string = "prop_A"
        if with_bkg_subtr:
            m_delta_values, f_max = load_data("1807.03075/1807.03075_prop_A_bkg.csv")
        else:
            m_delta_values, f_max = load_data("1807.03075/1807.03075_prop_A_nobkg.csv")

    elif prop_B:
        prop_string = "prop_B"
        if with_bkg_subtr:
            if not prop_B_lower:
                m_delta_values, f_max = load_data("1807.03075/1807.03075_prop_B_bkg_upper.csv")
            else:
                m_delta_values, f_max = load_data("1807.03075/1807.03075_prop_B_bkg_lower.csv")                        
        else:
            if not prop_B_lower:
                m_delta_values, f_max = load_data("1807.03075/1807.03075_prop_B_nobkg_upper.csv")
            else:
                m_delta_values, f_max = load_data("1807.03075/1807.03075_prop_B_nobkg_lower.csv")                        
        
    if not with_bkg_subtr:
        prop_string += "_nobkg"
    if prop_B_lower:
        prop_string += "_lower"
    elif prop_B:
        prop_string += "_upper"
    
    for j in range(len(Deltas)):
        
        if Deltas[j] == 5:
        
            fig, axes = plt.subplots(2, 2, figsize=(13, 13))
            
            ax0 = axes[0][0]
            ax1 = axes[0][1]
            ax2 = axes[1][0]
            ax3 = axes[1][1]
                    
            m_delta_extrapolated = 10**np.arange(11, np.log10(min(m_delta_values))+0.01, 0.1)
                    
            for k, exponent_PL_lower in enumerate(exponents_PL_lower):
                
                data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower) 
                data_filename_LN = data_folder + "/LN_1807.03075_Carr_" + prop_string + "_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
                data_filename_SLN = data_folder + "/SLN_1807.03075_Carr_" + prop_string + "_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
                data_filename_CC3 = data_folder + "/CC3_1807.03075_Carr_" + prop_string + "_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
                
                mc_LN_evolved, f_PBH_LN_evolved = np.genfromtxt(data_filename_LN, delimiter="\t")
                mc_SLN_evolved, f_PBH_SLN_evolved = np.genfromtxt(data_filename_SLN, delimiter="\t")
                mp_CC3_evolved, f_PBH_CC3_evolved = np.genfromtxt(data_filename_CC3, delimiter="\t")
                mp_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[j], alpha=alphas_SLN[j], log_m_factor=3, n_steps=1000) for m_c in mc_SLN_evolved]
                mp_LN = mc_LN_evolved * np.exp(-sigmas_LN[j]**2)
                            
                ax1.plot(mp_LN, f_PBH_LN_evolved, linestyle=linestyles[k], color="r", marker="None")
                ax2.plot(mp_SLN, f_PBH_SLN_evolved, linestyle=linestyles[k], color="b", marker="None")
                ax3.plot(mp_CC3_evolved, f_PBH_CC3_evolved, linestyle=linestyles[k], color="g", marker="None")
                
                ax1.set_title("LN")
                ax2.set_title("SLN")
                ax3.set_title("CC3")
                
                ax1.plot(0, 0, marker="None", linestyle=linestyles[k], color="k", label="{:.0f}".format(exponent_PL_lower))
                f_max_extrapolated = min(f_max) * np.power(m_delta_extrapolated / min(m_delta_values), exponent_PL_lower)
                ax0.plot(m_delta_extrapolated, f_max_extrapolated, color="tab:grey", linestyle=linestyles[k], label="{:.0f}".format(exponent_PL_lower))
                
            # Plot extended MF constraints from Korwar & Profumo (2023)
            plot_KP23(j, ax1, ax2, ax3)
            
            # Plot extended MF constraints from Galactic Centre photons calculated using Isatis         
            plotter_GC_Isatis(Deltas, j, ax1, color="tab:grey", mf=LN, params=[sigmas_LN[j]], linestyle="dotted")
            plotter_GC_Isatis(Deltas, j, ax2, color="tab:grey", mf=SLN, params=[sigmas_SLN[j], alphas_SLN[j]], linestyle="dotted")
            plotter_GC_Isatis(Deltas, j, ax3, color="tab:grey", mf=CC3, params=[alphas_CC3[j], betas[j]], linestyle="dotted")
            
            ax0.plot(m_delta_values, f_max, color="tab:grey")
            ax0.set_xlabel("$m$ [g]")
            ax0.set_ylabel("$f_\mathrm{max}$")
            ax0.set_xscale("log")
            ax0.set_yscale("log")
            ax0.set_xlim(min(m_delta_extrapolated), 1e18)
            ax0.set_ylim(min(f_max_extrapolated), 1)
            
            for ax in [ax1, ax2, ax3]:
                if Deltas[j] < 5:
                    ax.set_xlim(1e16, 1e18)
                else:
                    ax.set_xlim(1e16, 7e18)
                   
                ax.set_ylim(1e-6, 1)
                ax.set_ylabel("$f_\mathrm{PBH}$")
                ax.set_xlabel("$m_p~[\mathrm{g}]$")
                ax.set_xscale("log")
                ax.set_yscale("log")
    
            ax0.legend(fontsize="x-small", title="PL exponent in $f_\mathrm{max}$ \n " + "($m < {:.0e}".format(min(m_delta_values)) + "~\mathrm{g}$)")        
            ax1.legend(fontsize="x-small", title="PL exponent in $f_\mathrm{max}$ \n " + "($m < {:.0e}".format(min(m_delta_values)) + "~\mathrm{g}$)")
            fig.suptitle("Voyager 1 [1807.03075], $\Delta={:.1f}$".format(Deltas[j]) + " %s" % prop_string)
            fig.tight_layout()

#%% Tests of the results obtained using different power-law exponents in f_max at low masses 
# Constraints from 1912.01014 (511 keV line).

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    mc_values = np.logspace(14, 20, 120)

    # Power-law exponent to use between 1e11g and the smallest mass the delta-function MF constraint is calculated for.   
    exponents_PL_lower = [0, -2, -4]
    
    linestyles = ["solid", "dashed", "dashdot", "dotted"]
    
    # Load delta function MF constraints calculated using Isatis, to use the method from 1705.05567.
    m_delta_values, f_max = load_data("1912.01014/1912.01014_Fig2_a__0_newaxes_2.csv")
        
    for j in range(len(Deltas)):
        
        fig, axes = plt.subplots(2, 2, figsize=(13, 13))
        
        ax0 = axes[0][0]
        ax1 = axes[0][1]
        ax2 = axes[1][0]
        ax3 = axes[1][1]
                
        m_delta_extrapolated = 10**np.arange(11, np.log10(min(m_delta_values))+0.01, 0.1)
                
        for k, exponent_PL_lower in enumerate(exponents_PL_lower):
            
            data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower) 
            data_filename_LN = data_folder + "/LN_1912.01014_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
            data_filename_SLN = data_folder + "/SLN_1912.01014_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
            data_filename_CC3 = data_folder + "/CC3_1912.01014_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
            
            mc_LN_evolved, f_PBH_LN_evolved = np.genfromtxt(data_filename_LN, delimiter="\t")
            mc_SLN_evolved, f_PBH_SLN_evolved = np.genfromtxt(data_filename_SLN, delimiter="\t")
            mp_CC3_evolved, f_PBH_CC3_evolved = np.genfromtxt(data_filename_CC3, delimiter="\t")
            mp_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[j], alpha=alphas_SLN[j], log_m_factor=3, n_steps=1000) for m_c in mc_SLN_evolved]
            mp_LN = mc_LN_evolved * np.exp(-sigmas_LN[j]**2)
                        
            ax1.plot(mp_LN, f_PBH_LN_evolved, linestyle=linestyles[k], color="r", marker="None")
            ax2.plot(mp_SLN, f_PBH_SLN_evolved, linestyle=linestyles[k], color="b", marker="None")
            ax3.plot(mp_CC3_evolved, f_PBH_CC3_evolved, linestyle=linestyles[k], color="g", marker="None")
            
            ax1.set_title("LN")
            ax2.set_title("SLN")
            ax3.set_title("CC3")
            
            ax1.plot(0, 0, marker="None", linestyle=linestyles[k], color="k", label="{:.0f}".format(exponent_PL_lower))
            f_max_extrapolated = f_max[0] * np.power(m_delta_extrapolated / min(m_delta_values), exponent_PL_lower)
            ax0.plot(m_delta_extrapolated, f_max_extrapolated, color="tab:grey", linestyle=linestyles[k], label="{:.0f}".format(exponent_PL_lower))
        
        # Plot extended MF constraints from Korwar & Profumo (2023)
        plot_KP23(j, ax1, ax2, ax3)
        # Plot extended MF constraints from Galactic Centre photons calculated using Isatis         
        plotter_GC_Isatis(Deltas, j, ax1, color="tab:grey", mf=LN, params=[sigmas_LN[j]], linestyle="dotted")
        plotter_GC_Isatis(Deltas, j, ax2, color="tab:grey", mf=SLN, params=[sigmas_SLN[j], alphas_SLN[j]], linestyle="dotted")
        plotter_GC_Isatis(Deltas, j, ax3, color="tab:grey", mf=CC3, linestyle="dotted")
        
        ax0.plot(m_delta_values, f_max, color="tab:grey")
        ax0.set_xlabel("$m$ [g]")
        ax0.set_ylabel("$f_\mathrm{max}$")
        ax0.set_xscale("log")
        ax0.set_yscale("log")
        ax0.set_xlim(min(m_delta_extrapolated), 1e18)
        ax0.set_ylim(min(f_max), 1)
        
        for ax in [ax1, ax2, ax3]:
            if Deltas[j] < 5:
                ax.set_xlim(1e16, 1e18)
            else:
                ax.set_xlim(1e16, 7e18)
               
            ax.set_ylim(1e-6, 1)
            ax.set_ylabel("$f_\mathrm{PBH}$")
            ax.set_xlabel("$m_p~[\mathrm{g}]$")
            ax.set_xscale("log")
            ax.set_yscale("log")

        ax0.legend(fontsize="x-small", title="PL exponent in $f_\mathrm{max}$ \n " + "($m < {:.0e}".format(min(m_delta_values)) + "~\mathrm{g}$)")        
        ax1.legend(fontsize="x-small", title="PL exponent in $f_\mathrm{max}$ \n " + "($m < {:.0e}".format(min(m_delta_values)) + "~\mathrm{g}$)")
        fig.suptitle("511 keV line [1912.01014], $\Delta={:.1f}$".format(Deltas[j]))
        fig.tight_layout()

#%% Tests of the results obtained using different power-law exponents in f_max at low masses 
# Constraints from 2108.13256 (CMB anisotropies).

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    mc_values = np.logspace(14, 20, 120)

    # Power-law exponent to use between 1e11g and the smallest mass the delta-function MF constraint is calculated for.   
    exponents_PL_lower = [0, -2, -4]
    
    linestyles = ["solid", "dashed", "dashdot", "dotted"]
    
    # Load delta function MF constraints calculated using Isatis, to use the method from 1705.05567.
    m_delta_values, f_max = load_data("2108.13256/2108.13256_Fig4_CMB.csv")
    
    # Boolean determines whether to use evolved mass function (evaluated at the present-day).
    evolved = True
    
    for j in range(len(Deltas)):
        
        fig, axes = plt.subplots(2, 2, figsize=(13, 13))
        
        ax0 = axes[0][0]
        ax1 = axes[0][1]
        ax2 = axes[1][0]
        ax3 = axes[1][1]
                
        m_delta_extrapolated = 10**np.arange(11, np.log10(min(m_delta_values))+0.01, 0.1)
                
        for k, exponent_PL_lower in enumerate(exponents_PL_lower):
            
            if evolved:
                data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower) 
            else:
                data_folder = "./Data-tests/unevolved/PL_exp_{:.0f}".format(exponent_PL_lower)
            
            data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower) 
            data_filename_LN = data_folder + "/LN_2201.01265_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
            data_filename_SLN = data_folder + "/SLN_2201.01265_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
            data_filename_CC3 = data_folder + "/CC3_2201.01265_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
            
            mc_LN_evolved, f_PBH_LN_evolved = np.genfromtxt(data_filename_LN, delimiter="\t")
            mc_SLN_evolved, f_PBH_SLN_evolved = np.genfromtxt(data_filename_SLN, delimiter="\t")
            mp_CC3_evolved, f_PBH_CC3_evolved = np.genfromtxt(data_filename_CC3, delimiter="\t")
            mp_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[j], alpha=alphas_SLN[j], log_m_factor=3, n_steps=1000) for m_c in mc_SLN_evolved]
            mp_LN = mc_LN_evolved * np.exp(-sigmas_LN[j]**2)
                        
            ax1.plot(mp_LN, f_PBH_LN_evolved, linestyle=linestyles[k], color="r", marker="None")
            ax2.plot(mp_SLN, f_PBH_SLN_evolved, linestyle=linestyles[k], color="b", marker="None")
            ax3.plot(mp_CC3_evolved, f_PBH_CC3_evolved, linestyle=linestyles[k], color="g", marker="None")
                        
            ax1.set_title("LN")
            ax2.set_title("SLN")
            ax3.set_title("CC3")
            
            ax1.plot(0, 0, marker="None", linestyle=linestyles[k], color="k", label="{:.0f}".format(exponent_PL_lower))
            f_max_extrapolated = f_max[0] * np.power(m_delta_extrapolated / min(m_delta_values), exponent_PL_lower)
            ax0.plot(m_delta_extrapolated, f_max_extrapolated, color="tab:grey", linestyle=linestyles[k], label="{:.0f}".format(exponent_PL_lower))
        
        # Plot extended MF constraints from Korwar & Profumo (2023)
        plot_KP23(j, ax1, ax2, ax3)
        # Plot extended MF constraints from Galactic Centre photons calculated using Isatis         
        plotter_GC_Isatis(Deltas, j, ax1, color="tab:grey", mf=LN, params=[sigmas_LN[j]], linestyle="dotted")
        plotter_GC_Isatis(Deltas, j, ax2, color="tab:grey", mf=SLN, params=[sigmas_SLN[j], alphas_SLN[j]], linestyle="dotted")
        plotter_GC_Isatis(Deltas, j, ax3, color="tab:grey", mf=CC3, linestyle="dotted")
            
        ax0.plot(m_delta_values, f_max, color="tab:grey")
        ax0.set_xlabel("$m$ [g]")
        ax0.set_ylabel("$f_\mathrm{max}$")
        ax0.set_xscale("log")
        ax0.set_yscale("log")
        ax0.set_xlim(min(m_delta_extrapolated), 1e18)
        ax0.set_ylim(min(f_max), 1)
        
        for ax in [ax1, ax2, ax3]:
            if Deltas[j] < 5:
                ax.set_xlim(1e16, 1e18)
            else:
                ax.set_xlim(1e16, 7e18)
               
            ax.set_ylim(1e-6, 1)
            ax.set_ylabel("$f_\mathrm{PBH}$")
            ax.set_xlabel("$m_p~[\mathrm{g}]$")
            ax.set_xscale("log")
            ax.set_yscale("log")

        ax0.legend(fontsize="x-small", title="PL exponent in $f_\mathrm{max}$ \n " + "($m < {:.0e}".format(min(m_delta_values)) + "~\mathrm{g}$)")        
        ax1.legend(fontsize="x-small", title="PL exponent in $f_\mathrm{max}$ \n " + "($m < {:.0e}".format(min(m_delta_values)) + "~\mathrm{g}$)")
        fig.suptitle("CMB anisotropies [2108.13256], $\Delta={:.1f}$".format(Deltas[j]))
        fig.tight_layout()

#%% Plot the Galactic Centre photon constraints for an extended mass function, with the delta-function MF constraints obtained using Isatis.
# Compare to the approximate results obtained by using f_max as the constraint from each instrument, rather than the minimum over each energy bin, for the unevolved mass function and the 'evolved' mass function evaluated at the initial time t=0.

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    colors_evap = ["tab:orange", "tab:green", "tab:red", "tab:blue"]
    constraints_names_short = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
    
    # Parameters used for convergence tests in Galactic Centre constraints.
    cutoff = 1e-4
    delta_log_m = 1e-3
    E_number = 500    
    
    if E_number < 1e3:
        energies_string = "E{:.0f}".format(E_number)
    else:
        energies_string = "E{:.0f}".format(np.log10(E_number))
    
    plot_LN = False
    plot_SLN = True
    plot_CC3 = False
        
    for j in range(len(Deltas)):
        
        fig, ax = plt.subplots(figsize=(8, 8))
        fig1, ax1 = plt.subplots(figsize=(5,5))
        
        # Load constraints from Galactic Centre photons
        mc_values_old = np.logspace(14, 19, 100)
        
        fname_base_CC3 = "CC_D={:.1f}_dm{:.0f}_".format(Deltas[j], -np.log10(delta_log_m)) + energies_string + "_c{:.0f}".format(-np.log10(cutoff))
        fname_base_SLN = "SL_D={:.1f}_dm{:.0f}_".format(Deltas[j], -np.log10(delta_log_m)) + energies_string + "_c{:.0f}".format(-np.log10(cutoff))
        fname_base_LN = "LN_D={:.1f}_dm{:.0f}_".format(Deltas[j], -np.log10(delta_log_m)) + energies_string + "_c{:.0f}".format(-np.log10(cutoff))
        
        constraints_names_evap, f_PBH_SLN_unevolved = load_results_Isatis(mf_string=fname_base_SLN, modified=True)
        constraints_names_evap, f_PBH_CC3_unevolved = load_results_Isatis(mf_string=fname_base_CC3, modified=True)
        constraints_names, f_PBH_LN_unevolved = load_results_Isatis(mf_string=fname_base_LN, modified=True)
        
        mp_SLN_unevolved = [m_max_SLN(m_c, sigma=sigmas_SLN[j], alpha=alphas_SLN[j], log_m_factor=3, n_steps=1000) for m_c in mc_values_old]
        mp_LN_unevolved = mc_values_old * np.exp(-sigmas_LN[j]**2)
        
        for i in range(len(constraints_names)):
            if plot_LN:
                ax.plot(mp_LN_unevolved, f_PBH_LN_unevolved[i], label=constraints_names[i], color=colors_evap[i])
            elif plot_SLN:
                ax.plot(mp_SLN_unevolved, f_PBH_SLN_unevolved[i], label=constraints_names[i], color=colors_evap[i])
            elif plot_CC3:
                ax.plot(mc_values_old, f_PBH_CC3_unevolved[i], label=constraints_names[i], color=colors_evap[i])
                
            # Load and plot results for the unevolved mass functions
            data_filename_LN_unevolved = "./Data-tests/unevolved" + "/LN_GC_%s" % constraints_names_short[i] + "_Carr_Delta={:.1f}_unevolved.txt".format(Deltas[j])
            data_filename_SLN_unevolved = "./Data-tests/unevolved" + "/SLN_GC_%s" % constraints_names_short[i]  + "_Carr_Delta={:.1f}_unevolved.txt".format(Deltas[j])
            data_filename_CC3_unevolved = "./Data-tests/unevolved" + "/CC3_GC_%s" % constraints_names_short[i]  + "_Carr_Delta={:.1f}_unevolved.txt".format(Deltas[j])

            mc_LN_unevolved, f_PBH_LN_unevolved = np.genfromtxt(data_filename_LN_unevolved, delimiter="\t")
            mc_SLN_unevolved, f_PBH_SLN_unevolved = np.genfromtxt(data_filename_SLN_unevolved, delimiter="\t")
            mp_CC3_unevolved, f_PBH_CC3_unevolved = np.genfromtxt(data_filename_CC3_unevolved, delimiter="\t")
            
            mp_SLN_unevolved = [m_max_SLN(m_c, sigma=sigmas_SLN[j], alpha=alphas_SLN[j], log_m_factor=3, n_steps=1000) for m_c in mc_SLN_unevolved]
            mp_LN_unevolved = mc_LN_unevolved * np.exp(-sigmas_LN[j]**2)
            
            if plot_LN:
                ax.plot(mp_LN_unevolved, f_PBH_LN_unevolved, color=colors_evap[i], linestyle="None", marker="x")
            elif plot_SLN:
                ax.plot(mp_SLN_unevolved, f_PBH_SLN_unevolved, color=colors_evap[i], linestyle="None", marker="x")
            elif plot_CC3:
                ax.plot(mp_CC3_unevolved, f_PBH_CC3_unevolved, color=colors_evap[i], linestyle="None", marker="x")                

            # Load and plot results for the 'evolved' mass functions evaluated at the initial time t_init = 0
            data_filename_LN_t_init = "./Data-tests/t_initial" + "/LN_GC_%s" % constraints_names_short[i] + "_Carr_Delta={:.1f}.txt".format(Deltas[j])
            data_filename_SLN_t_init = "./Data-tests/t_initial" + "/SLN_GC_%s" % constraints_names_short[i]  + "_Carr_Delta={:.1f}.txt".format(Deltas[j])
            data_filename_CC3_t_init = "./Data-tests/t_initial" + "/CC3_GC_%s" % constraints_names_short[i]  + "_Carr_Delta={:.1f}.txt".format(Deltas[j])

            mc_LN_t_init, f_PBH_LN_t_init = np.genfromtxt(data_filename_LN_t_init, delimiter="\t")
            mc_SLN_t_init, f_PBH_SLN_t_init = np.genfromtxt(data_filename_SLN_t_init, delimiter="\t")
            mp_CC3_t_init, f_PBH_CC3_t_init = np.genfromtxt(data_filename_CC3_t_init, delimiter="\t")
            
            mp_SLN_t_init = [m_max_SLN(m_c, sigma=sigmas_SLN[j], alpha=alphas_SLN[j], log_m_factor=3, n_steps=1000) for m_c in mc_SLN_t_init]
            mp_LN_t_init = mc_LN_t_init * np.exp(-sigmas_LN[j]**2)
            
            if plot_LN:
                ax.plot(mp_LN_t_init, f_PBH_LN_t_init, color=colors_evap[i], linestyle="None", marker="+")    
                ax1.plot(mp_LN_unevolved, np.interp(mp_LN_unevolved, mp_LN_unevolved, f_PBH_SLN_unevolved) / f_PBH_LN_unevolved[i] - 1, color=colors_evap[i], linestyle="None", marker="+")
            elif plot_SLN:
                ax.plot(mp_SLN_t_init, f_PBH_SLN_t_init, color=colors_evap[i], linestyle="None", marker="+")   
                ax1.plot(mp_SLN_unevolved, np.interp(mp_SLN_unevolved, mp_SLN_unevolved, f_PBH_SLN_unevolved) / f_PBH_SLN_unevolved[i] - 1, color=colors_evap[i], linestyle="None", marker="+")
            elif plot_CC3:
                ax.plot(mp_CC3_t_init, f_PBH_CC3_t_init, color=colors_evap[i], linestyle="None", marker="+")
                ax1.plot(mc_values_old, np.interp(mc_values_old, mp_CC3_unevolved, f_PBH_CC3_unevolved) / f_PBH_CC3_unevolved[i] - 1, color=colors_evap[i], linestyle="None", marker="+")
                
            print(f_PBH_LN_unevolved[0:5])
            print(f_PBH_LN_t_init[0:5])
                    
        ax.plot(0, 0, linestyle="None", color="k", marker="+", label="Test (approximate): $t=0$")
        ax.plot(0, 0, linestyle="None", color="k", marker="x", label="Test (approximate): unevolved")
        ax.set_xlim(1e14, 1e18)
        ax.set_ylim(10**(-10), 1)
        ax.set_xlabel("$m_p~[\mathrm{g}]$")
        ax.set_ylabel("$f_\mathrm{PBH}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize="x-small")
        fig.tight_layout()
        fig.suptitle("$\Delta={:.1f}$".format(Deltas[j]))
        
        ax1.set_xlim(1e14, 1e18)
        ax1.set_xlabel("$m_p~[\mathrm{g}]$")
        ax1.set_ylabel("$\Delta f_\mathrm{PBH} / f_\mathrm{PBH}$")
        ax1.set_xscale("log")
        ax1.set_title("Approximate / full - 1 ($\Delta={:.1f}$)".format(Deltas[j]), fontsize="small")
        fig1.tight_layout()


#%% Tests of the results obtained using different power-law exponents in f_max at low masses (Galactic Centre photon constraints)

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    mc_values = np.logspace(14, 20, 120)

    # Array of power law exponents to use at masses below 1e15g
    exponents_PL_lower = [0, 2, 4]
    
    style_markers = ["--", "+", "x"]
    
    approx = False
    
    for j in range(len(Deltas)):
        
        fig, axes = plt.subplots(2, 2, figsize=(13, 13))
        
        ax0 = axes[0][0]
        ax1 = axes[0][1]
        ax2 = axes[1][0]
        ax3 = axes[1][1]
        
        m_delta_values_loaded = np.logspace(11, 21, 1000)
        constraints_names, f_max_Isatis = load_results_Isatis(modified=True)
        colors_evap = ["tab:orange", "tab:green", "tab:red", "tab:blue"]
        constraints_names_short = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
                                   
        m_delta_extrapolated = np.logspace(11, 13, 21)
        
        for k, exponent_PL_lower in enumerate(exponents_PL_lower):
            data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower)
            
            f_PBH_instrument_LN = []
            f_PBH_instrument_SLN = []
            f_PBH_instrument_CC3 = []
                
            for i in range(len(constraints_names)):
                
                # Set non-physical values of f_max (-1) to 1e100 from the f_max values calculated using Isatis
                f_max_allpositive = []
        
                for f_max in f_max_Isatis[i]:
                    if f_max == -1:
                        f_max_allpositive.append(1e100)
                    else:
                        f_max_allpositive.append(f_max)
                
                # Extrapolate f_max at masses below 1e13g using a power-law
                f_max_loaded_truncated = np.array(f_max_allpositive)[m_delta_values_loaded > 1e13]
                f_max_extrapolated = f_max_loaded_truncated[0] * np.power(m_delta_extrapolated / 1e13, exponent_PL_lower)
                f_max_i = np.concatenate((f_max_extrapolated, f_max_loaded_truncated))
                m_delta_values = np.concatenate((m_delta_extrapolated, m_delta_values_loaded[m_delta_values_loaded > 1e13]))

                ax0.plot(m_delta_extrapolated, f_max_extrapolated, style_markers[k], color=colors_evap[i])
                ax0.plot(m_delta_values_loaded[m_delta_values_loaded > 1e13], f_max_loaded_truncated, color=colors_evap[i])
                
                if approx:
                    # Load constraints for an evolved extended mass function obtained from each instrument
                    data_filename_LN = data_folder + "/LN_GC_%s" % constraints_names_short[i] + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[j])
                    data_filename_SLN = data_folder + "/SLN_GC_%s" % constraints_names_short[i]  + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[j])
                    data_filename_CC3 = data_folder + "/CC3_GC_%s" % constraints_names_short[i]  + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[j])
                
                else:
                    # Load constraints for an evolved extended mass function obtained from each instrument
                    data_filename_LN = data_folder + "/LN_GC_%s" % constraints_names_short[i] + "_Carr_Delta={:.1f}.txt".format(Deltas[j])
                    data_filename_SLN = data_folder + "/SLN_GC_%s" % constraints_names_short[i]  + "_Carr_Delta={:.1f}.txt".format(Deltas[j])
                    data_filename_CC3 = data_folder + "/CC3_GC_%s" % constraints_names_short[i]  + "_Carr_Delta={:.1f}.txt".format(Deltas[j])

                mc_LN_evolved, f_PBH_LN_evolved = np.genfromtxt(data_filename_LN, delimiter="\t")
                mc_SLN_evolved, f_PBH_SLN_evolved = np.genfromtxt(data_filename_SLN, delimiter="\t")
                mp_CC3_evolved, f_PBH_CC3_evolved = np.genfromtxt(data_filename_CC3, delimiter="\t")
                
                # Compile constraints from all instruments
                f_PBH_instrument_LN.append(f_PBH_LN_evolved)
                f_PBH_instrument_SLN.append(f_PBH_SLN_evolved)
                f_PBH_instrument_CC3.append(f_PBH_CC3_evolved)
                
                                
            mp_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[j], alpha=alphas_SLN[j], log_m_factor=3, n_steps=1000) for m_c in mc_values]
            mp_LN = mc_values * np.exp(-sigmas_LN[j]**2)
            mp_CC3 = mc_values
            
            # Plot the tightest constraint (of the different instruments) for each peak mass
            ax1.plot(mp_LN, envelope(f_PBH_instrument_LN), style_markers[k], color="r")
            ax2.plot(mp_SLN, envelope(f_PBH_instrument_SLN), style_markers[k], color="b")
            ax3.plot(mp_CC3, envelope(f_PBH_instrument_CC3), style_markers[k], color="g")
            
            if exponent_PL_lower == 2:
                print("\n data_filename_LN [in plot_constraints_tests.py]")
                print(data_filename_LN)
                
            # Plot extended MF constraints from Galactic Centre photons calculated using Isatis  [sanity check]   
            plotter_GC_Isatis(Deltas, j, ax1, color="tab:grey", mf=LN, params=[sigmas_LN[j]], linestyle="dotted", approx=approx, linewidth=5)
            plotter_GC_Isatis(Deltas, j, ax2, color="tab:grey", mf=SLN, params=[sigmas_SLN[j], alphas_SLN[j]], linestyle="dotted", approx=approx, linewidth=5)
            plotter_GC_Isatis(Deltas, j, ax3, color="tab:grey", mf=CC3, linestyle="dotted", approx=approx, linewidth=5)
                        
            ax1.set_title("LN")
            ax2.set_title("SLN")
            ax3.set_title("CC3")
            
            # Plot extended MF constraints from Korwar & Profumo (2023)
            plot_KP23(j, ax1, ax2, ax3)

            ax0.plot(0, 0, style_markers[k], color="k", label="{:.0f}".format(exponent_PL_lower))            
            ax1.plot(0, 0, style_markers[k], color="k", label="{:.0f}".format(exponent_PL_lower))
                
            ax0.set_xlim(1e11, 3e17)
            ax0.set_ylim(10**(-15), 1)
            ax0.set_xlabel("m$~[\mathrm{g}]$")
            ax0.set_ylabel("$f_\mathrm{max}$")
            ax0.set_xscale("log")
            ax0.set_yscale("log")
            
            for ax in [ax1, ax2, ax3]:
                
                if Deltas[j] < 5:
                    ax.set_xlim(1e16, 3e17)
                else:
                    ax.set_xlim(1e16, 1e18)
                   
                ax.set_ylim(1e-4, 1)
                ax.set_ylabel("$f_\mathrm{PBH}$")
                ax.set_xlabel("$m_p~[\mathrm{g}]$")
                ax.set_xscale("log")
                ax.set_yscale("log")
    
            ax0.legend(fontsize="x-small", title="PL exponent in $f_\mathrm{max}$ \n ($m < 10^{13}~\mathrm{g}$)")        
            ax1.legend(fontsize="x-small", title="PL exponent in $f_\mathrm{max}$ \n ($m < 10^{13}~\mathrm{g}$)")
            fig.suptitle("GC photons [Isatis], $\Delta={:.1f}$".format(Deltas[j]))
            fig.tight_layout()

#%% Tests of the results obtained using different power-law exponents in f_max at low masses (Extragalactic gamma ray photon constraints)

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    mc_values = np.logspace(14, 20, 120)

    # Array of power law exponents to use at masses below 1e15g
    exponents_PL_lower = [0, -2, -4]
    
    style_markers = ["--", "+", "x"]
    
    for j in range(len(Deltas)):
        
        fig, axes = plt.subplots(2, 2, figsize=(13, 13))
        
        ax0 = axes[0][0]
        ax1 = axes[0][1]
        ax2 = axes[1][0]
        ax3 = axes[1][1]
        
        m_delta_values_loaded = np.logspace(14, 17, 32)
        constraints_names, f_max_Isatis = load_results_Isatis(mf_string="EXGB_Hazma")
        colors_evap = ["tab:orange", "tab:green", "tab:red", "tab:blue", "k"]
        constraints_names_short = ["COMPTEL_1502.06116", "COMPTEL_1107.0200", "EGRET_0405441", "EGRET_9811211", "Fermi-LAT_1410.3696", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200", "HEAO+balloon_9903492"]                                   
        m_delta_extrapolated = np.logspace(11, 14, 31)        
        
        for k, exponent_PL_lower in enumerate(exponents_PL_lower):
            data_folder = "./Data-tests/unevolved/PL_exp_{:.0f}".format(exponent_PL_lower)
            
            f_PBH_instrument_LN = []
            f_PBH_instrument_SLN = []
            f_PBH_instrument_CC3 = []
            
            for i in range(len(constraints_names)):
                
                if i in (0, 2, 4, 7):
                    # Set non-physical values of f_max (-1) to 1e100 from the f_max values calculated using Isatis
                    f_max_allpositive = []
            
                    for f_max in f_max_Isatis[i]:
                        if f_max == -1:
                            f_max_allpositive.append(1e100)
                        else:
                            f_max_allpositive.append(f_max)
                    
                    # Extrapolate f_max at masses below 1e13g using a power-law
                    f_max_loaded_truncated = np.array(f_max_allpositive)[m_delta_values_loaded > 1e14]
                    f_max_extrapolated = f_max_loaded_truncated[0] * np.power(m_delta_extrapolated / 1e14, exponent_PL_lower)
                    f_max_i = np.concatenate((f_max_extrapolated, f_max_loaded_truncated))
                    m_delta_values = np.concatenate((m_delta_extrapolated, m_delta_values_loaded[m_delta_values_loaded > 1e14]))
    
                    ax0.plot(m_delta_extrapolated, f_max_extrapolated, style_markers[k], color=colors_evap[int(i/2)])
                    ax0.plot(m_delta_values_loaded[m_delta_values_loaded > 1e14], f_max_loaded_truncated, color=colors_evap[int(i/2)])
    
                    # Load constraints for an evolved extended mass function obtained from each instrument
                    data_filename_LN = data_folder + "/LN_EXGB_%s" % constraints_names_short[i] + "_Carr_Delta={:.1f}_unevolved.txt".format(Deltas[j])
                    data_filename_SLN = data_folder + "/SLN_EXGB_%s" % constraints_names_short[i]  + "_Carr_Delta={:.1f}_unevolved.txt".format(Deltas[j])
                    data_filename_CC3 = data_folder + "/CC3_EXGB_%s" % constraints_names_short[i]  + "_Carr_Delta={:.1f}_unevolved.txt".format(Deltas[j])
                        
                    mc_LN_evolved, f_PBH_LN_evolved = np.genfromtxt(data_filename_LN, delimiter="\t")
                    mc_SLN_evolved, f_PBH_SLN_evolved = np.genfromtxt(data_filename_SLN, delimiter="\t")
                    mp_CC3_evolved, f_PBH_CC3_evolved = np.genfromtxt(data_filename_CC3, delimiter="\t")
                    
                    # Compile constraints from all instruments
                    f_PBH_instrument_LN.append(f_PBH_LN_evolved)
                    f_PBH_instrument_SLN.append(f_PBH_SLN_evolved)
                    f_PBH_instrument_CC3.append(f_PBH_CC3_evolved)
                                        
            mp_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[j], alpha=alphas_SLN[j], log_m_factor=3, n_steps=1000) for m_c in mc_values]
            mp_LN = mc_values * np.exp(-sigmas_LN[j]**2)
            mp_CC3 = mc_values
            
            # Plot the tightest constraint (of the different instruments) for each peak mass
            ax1.plot(mp_LN, envelope(f_PBH_instrument_LN), style_markers[k], color="r")
            ax2.plot(mp_SLN, envelope(f_PBH_instrument_SLN), style_markers[k], color="b")
            ax3.plot(mp_CC3, envelope(f_PBH_instrument_CC3), style_markers[k], color="g")
                        
            ax1.set_title("LN")
            ax2.set_title("SLN")
            ax3.set_title("CC3")

            ax0.plot(0, 0, style_markers[k], color="k", label="{:.0f}".format(exponent_PL_lower))            
            ax1.plot(0, 0, style_markers[k], color="k", label="{:.0f}".format(exponent_PL_lower))
            
            # Plot extended MF constraints from Korwar & Profumo (2023)
            plot_KP23(j, ax1, ax2, ax3)
            # Plot extended MF constraints from Galactic Centre photons calculated using Isatis         
            plotter_GC_Isatis(Deltas, j, ax1, color="tab:grey", mf=LN, params=[sigmas_LN[j]], linestyle="dotted")
            plotter_GC_Isatis(Deltas, j, ax2, color="tab:grey", mf=SLN, params=[sigmas_SLN[j], alphas_SLN[j]], linestyle="dotted")
            plotter_GC_Isatis(Deltas, j, ax3, color="tab:grey", mf=CC3, linestyle="dotted")
            
            ax0.set_xlim(1e11, 3e17)
            ax0.set_ylim(10**(-15), 1)
            ax0.set_xlabel("m$~[\mathrm{g}]$")
            ax0.set_ylabel("$f_\mathrm{max}$")
            ax0.set_xscale("log")
            ax0.set_yscale("log")
            
            for ax in [ax1, ax2, ax3]:
                
                if Deltas[j] < 5:
                    ax.set_xlim(1e16, 3e17)
                else:
                    ax.set_xlim(1e16, 1e18)
                   
                ax.set_ylim(1e-4, 1)
                ax.set_ylabel("$f_\mathrm{PBH}$")
                ax.set_xlabel("$m_p~[\mathrm{g}]$")
                ax.set_xscale("log")
                ax.set_yscale("log")
    
            ax0.legend(fontsize="x-small", title="PL exponent in $f_\mathrm{max}$ \n ($m < 10^{13}~\mathrm{g}$)")        
            ax1.legend(fontsize="x-small", title="PL exponent in $f_\mathrm{max}$ \n ($m < 10^{13}~\mathrm{g}$)")
            fig.suptitle("EXGB photons [Isatis], $\Delta={:.1f}$".format(Deltas[j]))
            fig.tight_layout()


#%% Compare results obtained at t=0 and the unevolved MF (prospective evaporation constraints from GECCO [2101.01370]).
# CHECK PASSED: Differences are < 5% in the range of peak masses of interest, even for Delta = 5.

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
            
    LN = False
    SLN = True
    CC3 = False
    
    profile_string = "NFW"
            
    for j in range(len(Deltas)):
        
        fig, ax = plt.subplots(figsize=(8, 8))                                                
        # Load and plot results for the unevolved mass functions
        data_filename_LN_unevolved = "./Data-tests/unevolved" + "/LN_2101.01370_Carr_Delta={:.1f}_".format(Deltas[j]) + "%s_" % profile_string + ".txt"
        data_filename_SLN_unevolved = "./Data-tests/unevolved" + "/SLN_2101.01370_Carr_Delta={:.1f}_".format(Deltas[j]) + "%s_" % profile_string + ".txt"
        data_filename_CC3_unevolved = "./Data-tests/unevolved" + "/CC3_2101.01370_Carr_Delta={:.1f}_".format(Deltas[j]) + "%s_" % profile_string + ".txt"

        mc_LN_unevolved, f_PBH_LN_unevolved = np.genfromtxt(data_filename_LN_unevolved, delimiter="\t")
        mc_SLN_unevolved, f_PBH_SLN_unevolved = np.genfromtxt(data_filename_SLN_unevolved, delimiter="\t")
        mp_CC3_unevolved, f_PBH_CC3_unevolved = np.genfromtxt(data_filename_CC3_unevolved, delimiter="\t")
        
        mp_SLN_unevolved = [m_max_SLN(m_c, sigma=sigmas_SLN[j], alpha=alphas_SLN[j], log_m_factor=3, n_steps=1000) for m_c in mc_SLN_unevolved]
        mp_LN_unevolved = mc_LN_unevolved * np.exp(-sigmas_LN[j]**2)
        
        ax.plot(mp_LN_unevolved, f_PBH_LN_unevolved, linestyle="None", marker="x", color="r", label="Test (approximate): unevolved")
        ax.plot(mp_SLN_unevolved, f_PBH_SLN_unevolved, linestyle="None", marker="x", color="b")
        ax.plot(mp_CC3_unevolved, f_PBH_CC3_unevolved, linestyle="None", marker="x", color="g")                

        # Load and plot results for the 'evolved' mass functions evaluated at the initial time t_init = 0
        data_filename_LN_t_init = "./Data-tests/t_initial" + "/LN_2101.01370_Carr_Delta={:.1f}_".format(Deltas[j]) + "%s_" % profile_string + ".txt"
        data_filename_SLN_t_init = "./Data-tests/t_initial" + "/SLN_2101.01370_Carr_Delta={:.1f}_".format(Deltas[j]) + "%s_" % profile_string + ".txt"
        data_filename_CC3_t_init = "./Data-tests/t_initial" + "/CC3_2101.01370_Carr_Delta={:.1f}_".format(Deltas[j]) + "%s_" % profile_string + ".txt"

        mc_LN_t_init, f_PBH_LN_t_init = np.genfromtxt(data_filename_LN_t_init, delimiter="\t")
        mc_SLN_t_init, f_PBH_SLN_t_init = np.genfromtxt(data_filename_SLN_t_init, delimiter="\t")
        mp_CC3_t_init, f_PBH_CC3_t_init = np.genfromtxt(data_filename_CC3_t_init, delimiter="\t")
        
        mp_SLN_t_init = [m_max_SLN(m_c, sigma=sigmas_SLN[j], alpha=alphas_SLN[j], log_m_factor=3, n_steps=1000) for m_c in mc_SLN_t_init]
        mp_LN_t_init = mc_LN_t_init * np.exp(-sigmas_LN[j]**2)
        
        ax.plot(mp_LN_t_init, f_PBH_LN_t_init, marker="+", linestyle="None", color="r", label="Test (approximate): $t=0$")    
        ax.plot(mp_SLN_t_init, f_PBH_SLN_t_init, marker="+", linestyle="None", color="b")   
        ax.plot(mp_CC3_t_init, f_PBH_CC3_t_init, marker="+", linestyle="None", color="g")
               
            
        if Deltas[j] < 5:
            ax.set_xlim(1e16, 2e18)
        else:
            ax.set_xlim(1e16, 5e18)
           
        ax.set_xlim(1e16, 1e19)
        ax.set_ylim(1e-6, 1)
        ax.set_ylabel("$f_\mathrm{PBH}$")
        ax.set_xlabel("$m_p~[\mathrm{g}]$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize="x-small")
        fig.suptitle("$\Delta={:.1f}$".format(Deltas[j]))
        fig.tight_layout()


#%% Tests of the results obtained using different power-law exponents in f_max at low masses (prospective evaporation constraints from GECCO [2101.01370])

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    # Array of power law exponents to use at masses below 1e15g
    exponents_PL_lower = [0, 2, 4]
    
    linestyles = ["dashed", "dashdot", "dotted"]
    
    for j in range(len(Deltas)):
        
        fig, axes = plt.subplots(2, 2, figsize=(13, 13))
        
        ax0 = axes[0][0]
        ax1 = axes[0][1]
        ax2 = axes[1][0]
        ax3 = axes[1][1]
        
        m_delta_values, f_max = load_data("2101.01370/2101.01370_Fig9_GC_Einasto.csv")
        m_delta_extrapolated = np.logspace(11, 15, 41)
                
        for k, exponent_PL_lower in enumerate(exponents_PL_lower):
            
            data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower) 
            data_filename_LN = data_folder + "/LN_2101.01370_Carr_Delta={:.1f}_Einasto_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
            data_filename_SLN = data_folder + "/SLN_2101.01370_Carr_Delta={:.1f}_Einasto_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
            data_filename_CC3 = data_folder + "/CC3_2101.01370_Carr_Delta={:.1f}_Einasto_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
            
            mc_LN_evolved, f_PBH_LN_evolved = np.genfromtxt(data_filename_LN, delimiter="\t")
            mc_SLN_evolved, f_PBH_SLN_evolved = np.genfromtxt(data_filename_SLN, delimiter="\t")
            mp_CC3_evolved, f_PBH_CC3_evolved = np.genfromtxt(data_filename_CC3, delimiter="\t")
            mp_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[j], alpha=alphas_SLN[j], log_m_factor=3, n_steps=1000) for m_c in mc_SLN_evolved]
            mp_LN = mc_LN_evolved * np.exp(-sigmas_LN[j]**2)
                        
            ax1.plot(mp_LN, f_PBH_LN_evolved, linestyle=linestyles[k], color="r", marker="None")
            ax2.plot(mp_SLN, f_PBH_SLN_evolved, linestyle=linestyles[k], color="b", marker="None")
            ax3.plot(mp_CC3_evolved, f_PBH_CC3_evolved, linestyle=linestyles[k], color="g", marker="None")
            
            ax1.set_title("LN")
            ax2.set_title("SLN")
            ax3.set_title("CC3")
            
            ax1.plot(0, 0, marker="None", linestyle=linestyles[k], color="k", label="{:.0f}".format(exponent_PL_lower))
            f_max_extrapolated = min(f_max) * np.power(m_delta_extrapolated / min(m_delta_values), exponent_PL_lower)
            ax0.plot(m_delta_extrapolated, f_max_extrapolated, color=(0.5294, 0.3546, 0.7020), linestyle=linestyles[k], label="{:.0f}".format(exponent_PL_lower))
            
            # Plot extended MF constraints from Korwar & Profumo (2023)
            plot_KP23(j, ax1, ax2, ax3)
            
        ax0.plot(np.concatenate((m_delta_extrapolated, m_delta_values)), np.concatenate((f_max_extrapolated, f_max)), color=(0.5294, 0.3546, 0.7020))
        ax0.set_xlabel("$m$ [g]")
        ax0.set_ylabel("$f_\mathrm{max}$")
        ax0.set_xscale("log")
        ax0.set_yscale("log")
        ax0.set_xlim(min(m_delta_extrapolated), 1e18)
        ax0.set_ylim(min(f_max_extrapolated), 1)
        
        for ax in [ax1, ax2, ax3]:
            if Deltas[j] < 5:
                ax.set_xlim(1e16, 2e18)
            else:
                ax.set_xlim(1e16, 5e18)
               
            ax.set_ylim(1e-6, 1)
            ax.set_ylabel("$f_\mathrm{PBH}$")
            ax.set_xlabel("$m_p~[\mathrm{g}]$")
            ax.set_xscale("log")
            ax.set_yscale("log")

        ax0.legend(fontsize="x-small", title="PL exponent in $f_\mathrm{max}$ \n ($m < 10^{15}~\mathrm{g}$)")        
        ax1.legend(fontsize="x-small", title="PL exponent in $f_\mathrm{max}$ \n ($m < 10^{15}~\mathrm{g}$)")
        fig.suptitle("Prpspective (GECCO) [2101.01370], $\Delta={:.1f}$".format(Deltas[j]))
        fig.tight_layout()


#%% Check how the Subaru-HSC microlensing constraints for extended MFs change when using different power law extrapolations below ~1e22g in the delta-function MF constraint
if "__main__" == __name__:

    mc_subaru = 10**np.linspace(17, 29, 1000)
    
    # Constraints for Delta-function MF.
    m_subaru_delta_loaded, f_max_subaru_delta_loaded = load_data("./2007.12697/Subaru-HSC_2007.12697_dx=5.csv")
    
    # Mass function parameter values, from 2009.03204.
    [Deltas, sigmas_LN, ln_mc_SL, mp_SL, sigmas_SLN, alphas_SLN, mp_CC, alphas_CC, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
        
    exponents_PL_lower = [-2, -4]
    markers = [":", "--"]
    
    for i in range(len(Deltas)):
        
        fig, axes = plt.subplots(2, 2, figsize=(13, 13))
        
        ax0 = axes[0][0]
        ax1 = axes[0][1]
        ax2 = axes[1][0]
        ax3 = axes[1][1]
        
        for k, exponent_PL_lower in enumerate(exponents_PL_lower):
        
            data_filename_SLN = "./Data-tests/PL_exp_{:.0f}/SLN_HSC_Carr_Delta={:.1f}.txt".format(exponent_PL_lower, Deltas[i])
            data_filename_CC3 = "./Data-tests/PL_exp_{:.0f}/CC3_HSC_Carr_Delta={:.1f}.txt".format(exponent_PL_lower, Deltas[i])
            data_filename_LN = "./Data-tests/PL_exp_{:.0f}/LN_HSC_Carr_Delta={:.1f}.txt".format(exponent_PL_lower, Deltas[i])
            
            mc_LN, f_PBH_LN = np.genfromtxt(data_filename_LN, delimiter="\t")
            mc_SLN, f_PBH_SLN = np.genfromtxt(data_filename_LN, delimiter="\t")
            mp_CC3, f_PBH_CC3 = np.genfromtxt(data_filename_LN, delimiter="\t")
            
            mp_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_SLN]
            mp_LN = mc_LN * np.exp(-sigmas_LN[i]**2)
                        
            ax1.plot(mp_LN, f_PBH_LN, markers[k], color="r")
            ax2.plot(mp_SLN, f_PBH_SLN, markers[k], color="b")
            ax3.plot(mp_CC3, f_PBH_CC3, markers[k], color="g")

            m_subaru_delta_extrapolated = np.logspace(18, np.log10(min(m_subaru_delta_loaded)), 50)
            f_max_subaru_delta_extrapolated = f_max_subaru_delta_loaded[0] * np.power(m_subaru_delta_extrapolated / min(m_subaru_delta_loaded), exponent_PL_lower)
            ax0.plot(m_subaru_delta_extrapolated, f_max_subaru_delta_extrapolated, markers[k], color="tab:blue", label="{:.0f}".format(exponent_PL_lower))
            ax1.plot(0, 0, marker="None", linestyle=linestyles[k], color="k", label="{:.0f}".format(exponent_PL_lower))

        ax0.plot(m_subaru_delta_loaded, f_max_subaru_delta_loaded, color="tab:blue")
        ax0.set_xlabel(r"$m~[\mathrm{g}]$")
        ax0.set_ylabel(r"$f_\mathrm{max}$")
        ax0.set_xscale("log")
        ax0.set_yscale("log")
        ax0.set_xlim(1e18, 1e29)
        ax0.set_ylim(1e-3, 1e4)
        
        # Plot the constraints obtained with no power-law extrapolation:
        data_filename_SLN = "./Data/SLN_HSC_Carr_Delta={:.1f}.txt".format(Deltas[i])
        data_filename_CC3 = "./Data/CC3_HSC_Carr_Delta={:.1f}.txt".format(Deltas[i])
        data_filename_LN = "./Data/LN_HSC_Carr_Delta={:.1f}.txt".format(Deltas[i])
        
        mc_LN, f_PBH_LN = np.genfromtxt(data_filename_LN, delimiter="\t")
        mc_SLN, f_PBH_SLN = np.genfromtxt(data_filename_LN, delimiter="\t")
        mp_CC3, f_PBH_CC3 = np.genfromtxt(data_filename_LN, delimiter="\t")
        
        mp_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_SLN]
        mp_LN = mc_LN * np.exp(-sigmas_LN[i]**2)
                    
        ax1.plot(mp_LN, f_PBH_LN, color="r", marker="None")
        ax2.plot(mp_SLN, f_PBH_SLN, color="b", marker="None")
        ax3.plot(mp_CC3, f_PBH_CC3, color="g", marker="None")
    
        for ax in [ax1, ax2, ax3]:
            if Deltas[i] < 5:
                ax.set_xlim(1e21, 1e29)
            else:
                ax.set_xlim(1e19, 1e29)
               
            ax.set_ylim(1e-3, 1)
            ax.set_ylabel("$f_\mathrm{PBH}$")
            ax.set_xlabel("$m_p~[\mathrm{g}]$")
            ax.set_xscale("log")
            ax.set_yscale("log")

        ax0.legend(fontsize="x-small", title="PL exponent in $f_\mathrm{max}$ \n ($m < 10^{22}~\mathrm{g}$)")        
        ax1.legend(fontsize="x-small", title="PL exponent in $f_\mathrm{max}$ \n ($m < 10^{22}~\mathrm{g}$)")
        fig.suptitle("Subaru-HSC, $\Delta={:.1f}$".format(Deltas[i]))
        fig.tight_layout()


#%% Plot constraints from 2302.04408 (MW diffuse SPI with NFW template).
# Convergence test with number of masses in preliminaries.py and range of masses included in power-law extrapolation of the delta-function MF constraint.
# Consider the evolved MFs only, with PL extrapolation.

if "__main__" == __name__:

    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    plot_LN = False
    plot_SLN = True
    plot_CC3 = False
    
    mc_values = np.logspace(14, 20, 120)
        
    # Power-law exponent to use between 1e11g and 1e15g.
    exponent_PL_lower = 2.0
    
    log_m_min_values = [7, 9, 11]
    n_steps_values = [1000, 10000, 100000]
    
    markers = ["x", "+", "1"]
    colors = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200']

    for j in range(len(Deltas)):
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax1 = ax.twinx()

        for k, log_m_min in enumerate(log_m_min_values):
            
            ax.plot(0, 0, markers[k], color="k", label="{:.0e}g".format(10**log_m_min))
                     
            for l, n_steps in enumerate(n_steps_values):
                
                if k == 0:
                    ax1.plot(0, 0, color=colors[l], label="{:.0f}".format(n_steps))                   
                
                data_folder = "./Data-tests/PL_exp_{:.0f}/log_m_min={:.0f}/log_n_steps={:.0f}".format(exponent_PL_lower, log_m_min, np.log10(n_steps))
                            
                data_filename_LN = data_folder + "/LN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
                data_filename_SLN = data_folder + "/SLN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
                data_filename_CC3 = data_folder + "/CC3_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
                    
                mc_LN_evolved, f_PBH_LN_evolved = np.genfromtxt(data_filename_LN, delimiter="\t")
                mc_SLN_evolved, f_PBH_SLN_evolved = np.genfromtxt(data_filename_SLN, delimiter="\t")
                mp_CC3_evolved, f_PBH_CC3_evolved = np.genfromtxt(data_filename_CC3, delimiter="\t")
                
                if plot_CC3:
                    data_filename_CC3 = data_folder + "/CC3_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
                    mp_CC3_evolved, f_PBH_CC3_evolved = np.genfromtxt(data_filename_CC3, delimiter="\t")
                    ax.plot(mp_CC3_evolved, f_PBH_CC3_evolved, markers[k], color=colors[l])
                    
                elif plot_LN:
                    data_filename_LN = data_folder + "/LN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
                    mc_LN_evolved, f_PBH_LN_evolved = np.genfromtxt(data_filename_LN, delimiter="\t")
                    mp_LN = mc_LN_evolved * np.exp(-sigmas_LN[j]**2)
                    ax.plot(mp_LN, f_PBH_LN_evolved, markers[k], color=colors[l])
                    
                elif plot_SLN:
                    data_filename_SLN = data_folder + "/SLN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
                    mc_SLN_evolved, f_PBH_SLN_evolved = np.genfromtxt(data_filename_SLN, delimiter="\t")
                    mp_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[j], alpha=alphas_SLN[j], log_m_factor=3, n_steps=1000) for m_c in mc_SLN_evolved]
                    ax.plot(mp_SLN, f_PBH_SLN_evolved, markers[k], color=colors[l])
      
        if Deltas[j] < 5:
            ax.set_xlim(1e16, 1e18)
        else:
            ax.set_xlim(1e16, 2e18)
        ax.set_ylim(1e-6, 1)

        ax.legend(fontsize="x-small", title="Min mass in $f_\mathrm{max}$")
        ax1.legend(fontsize="x-small", title="Number of steps")
        ax1.set_yticks([])
        ax.set_ylabel("$f_\mathrm{PBH}$")
        ax.set_xlabel("$m_p~[\mathrm{g}]$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        fig.suptitle("KP '23 convergence test ($\Delta={:.1f}$)".format(Deltas[j]))
        fig.tight_layout()

#%% Tests of the results obtained using different power-law exponents in f_max at masses M < 1e15g (Galactic Centre photon constraints)

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    mc_values = np.logspace(14, 20, 120)

    # Array of power law exponents to use at masses below 1e15g
    exponents_PL_lower = [0, 2, 4]
    
    style_markers = ["--", "+", "x"]
    
    for j in range(len(Deltas)):
        
        fig, axes = plt.subplots(2, 2, figsize=(13, 13))
        
        ax0 = axes[0][0]
        ax1 = axes[0][1]
        ax2 = axes[1][0]
        ax3 = axes[1][1]
        
        m_delta_values_loaded = np.logspace(11, 21, 1000)
        constraints_names, f_max_Isatis = load_results_Isatis(modified=True)
        colors_evap = ["tab:orange", "tab:green", "tab:red", "tab:blue"]
        constraints_names_short = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
                                   
        m_delta_extrapolated = np.logspace(11, 15, 21)
        
        
        for k, exponent_PL_lower in enumerate(exponents_PL_lower):
            
            data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower)
            
            f_PBH_instrument_LN = []
            f_PBH_instrument_SLN = []
            f_PBH_instrument_CC3 = []
                
            for i in range(len(constraints_names)):
                
                # Set non-physical values of f_max (-1) to 1e100 from the f_max values calculated using Isatis
                f_max_allpositive = []
        
                for f_max in f_max_Isatis[i]:
                    if f_max == -1:
                        f_max_allpositive.append(1e100)
                    else:
                        f_max_allpositive.append(f_max)
                
                # Extrapolate f_max at masses below 1e13g using a power-law
                f_max_loaded_truncated = np.array(f_max_allpositive)[m_delta_values_loaded > 1e15]
                f_max_extrapolated = f_max_loaded_truncated[0] * np.power(m_delta_extrapolated / 1e15, exponent_PL_lower)
                f_max_i = np.concatenate((f_max_extrapolated, f_max_loaded_truncated))
                m_delta_values = np.concatenate((m_delta_extrapolated, m_delta_values_loaded[m_delta_values_loaded > 1e15]))

                ax0.plot(m_delta_extrapolated, f_max_extrapolated, style_markers[k], color=colors_evap[i])
                ax0.plot(m_delta_values_loaded[m_delta_values_loaded > 1e15], f_max_loaded_truncated, color=colors_evap[i])

                # Load constraints for an evolved extended mass function obtained from each instrument
                data_filename_LN = data_folder + "/LN_GC_%s" % constraints_names_short[i] + "_Carr_Delta={:.1f}_mmin=1e15g.txt".format(Deltas[j])
                data_filename_SLN = data_folder + "/SLN_GC_%s" % constraints_names_short[i]  + "_Carr_Delta={:.1f}_mmin=1e15g.txt".format(Deltas[j])
                data_filename_CC3 = data_folder + "/CC3_GC_%s" % constraints_names_short[i]  + "_Carr_Delta={:.1f}_mmin=1e15g.txt".format(Deltas[j])
                    
                mc_LN_evolved, f_PBH_LN_evolved = np.genfromtxt(data_filename_LN, delimiter="\t")
                mc_SLN_evolved, f_PBH_SLN_evolved = np.genfromtxt(data_filename_SLN, delimiter="\t")
                mp_CC3_evolved, f_PBH_CC3_evolved = np.genfromtxt(data_filename_CC3, delimiter="\t")
                
                # Compile constraints from all instruments
                f_PBH_instrument_LN.append(f_PBH_LN_evolved)
                f_PBH_instrument_SLN.append(f_PBH_SLN_evolved)
                f_PBH_instrument_CC3.append(f_PBH_CC3_evolved)
                
                print("f_PBH_LN_evolved[0] =", f_PBH_LN_evolved[0])
                
            mp_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[j], alpha=alphas_SLN[j], log_m_factor=3, n_steps=1000) for m_c in mc_values]
            mp_LN = mc_values * np.exp(-sigmas_LN[j]**2)
            mp_CC3 = mc_values
            
            # Plot the tightest constraint (of the different instruments) for each peak mass
            ax1.plot(mp_LN, envelope(f_PBH_instrument_LN), style_markers[k], color="r")
            ax2.plot(mp_SLN, envelope(f_PBH_instrument_SLN), style_markers[k], color="b")
            ax3.plot(mp_CC3, envelope(f_PBH_instrument_CC3), style_markers[k], color="g")
            print("envelope(f_PBH_instrument_LN)[0] =", envelope(f_PBH_instrument_LN)[0])
                        
            ax1.set_title("LN")
            ax2.set_title("SLN")
            ax3.set_title("CC3")

            ax0.plot(0, 0, style_markers[k], color="k", label="{:.0f}".format(exponent_PL_lower))            
            ax1.plot(0, 0, style_markers[k], color="k", label="{:.0f}".format(exponent_PL_lower))
                
            ax0.set_xlim(1e11, 3e17)
            ax0.set_ylim(10**(-15), 1)
            ax0.set_xlabel("m$~[\mathrm{g}]$")
            ax0.set_ylabel("$f_\mathrm{max}$")
            ax0.set_xscale("log")
            ax0.set_yscale("log")
            
            for ax in [ax1, ax2, ax3]:
                
                if Deltas[j] < 5:
                    ax.set_xlim(1e16, 3e17)
                else:
                    ax.set_xlim(1e16, 1e18)
                   
                ax.set_ylim(1e-4, 1)
                ax.set_ylabel("$f_\mathrm{PBH}$")
                ax.set_xlabel("$m_p~[\mathrm{g}]$")
                ax.set_xscale("log")
                ax.set_yscale("log")
    
            ax0.legend(fontsize="x-small", title="PL exponent in $f_\mathrm{max}$ \n ($m < 10^{15}~\mathrm{g}$)")        
            ax1.legend(fontsize="x-small", title="PL exponent in $f_\mathrm{max}$ \n ($m < 10^{15}~\mathrm{g}$)")
            fig.tight_layout()
            fig.suptitle("$\Delta={:.1f}$".format(Deltas[j]))


#%% Compare how stringent the (approximate) extragalactic gamma ray constraints from Carr et al. (2021) are compared to the constraints from Korwar & Profumo (2023).

from preliminaries import constraint_Carr, LN, SLN, CC3

t_0 = 13.8e9 * 365.25 * 86400    # Age of Universe, in seconds

def Solmass_to_g(m):
    """Convert a mass m (in solar masses) to grams."""
    return 1.989e33 * m


def g_to_Solmass(m):
    """Convert a mass m (in grams) to solar masses."""
    return m / 1.989e33
    

def f_PBH_beta_prime(m_values, beta_prime):
    """
    Calcualte f_PBH from the initial PBH fraction beta_prime, using Eq. 57 of 2002.12778.

    Parameters
    ----------
    m_values : Array-like
        PBH masses, in grams.
    beta_prime : Array-like / float
        Scaled fraction of the energy density of the Universe in PBHs at their formation time (see Eq. 8 of 2002.12778), at each mass in m_values.

    Returns
    -------
    Array-like
        Value of f_PBH evaluated at each mass in m_values.

    """
    return 3.81e8 * beta_prime * np.power(m_values / 1.989e33, -1/2)
 
    
def beta_prime_gamma_rays(m_values, epsilon=0.4):
    """
    Calculate values of beta prime allowed from extragalactic gamma-rays, using the simple power-law expressions in 2002.12778.

    Parameters
    ----------
    m_values : Array-like
        PBH masses, in grams.
    epsilon : Float, optional
        Parameter describing the power-law dependence of x-ray and gamma-ray spectra on photon energy. The default is 0.4.

    Returns
    -------
    Array-like
        Scaled fraction of the energy density of the Universe in PBHs at their formation time (see Eq. 8 of 2002.12778), at each mass in m_values.

    """
    beta_prime_values = []
    
    for m in m_values:
        if m < m_star:
            beta_prime_values.append(5e-28 * np.power(m/m_star, -5/2-2*epsilon))
        else:
            beta_prime_values.append(5e-26 * np.power(m/m_star, 7/2+epsilon))
    
    return np.array(beta_prime_values)


if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)       
    
    m_min = 3e13    # Minimum mass corresponds to the smallest value that is shown in Fig. 7 of 2002.12778.
    m_max = 1e20
    epsilon = 0.4
    m_star = 5.1e14
    
    evolved = True
    t = t_0
    
    # Range of characteristic masses for obtaining constraints.
    mc_Carr21 = np.logspace(14, 22, 1000)

    colors=["r", "b", "g"]
      
    # Calculate delta-function MF constraints, using Eqs. 32-33 of 2002.12778 for beta_prime, and Eq. 57 to convert to f_PBH
    m_delta_Carr21 = 10**np.arange(np.log10(m_min), np.log10(m_max), 0.1)
    f_max_Carr21 = f_PBH_beta_prime(m_delta_Carr21, beta_prime_gamma_rays(m_delta_Carr21))
    
    # Calculate extended MF constraints obtained for a log-normal with the delta-function MF constraint from Korwar & Profumo (2023).
    for j in range(len(Deltas)):
        
        fig, ax = plt.subplots(figsize=(6, 6))
        
        params_LN = [sigmas_LN[j]]
        params_SLN = [sigmas_SLN[j], alphas_SLN[j]]
        params_CC3 = [alphas_CC3[j], betas[j]]
        
        f_pbh_LN = constraint_Carr(mc_Carr21, m_delta_Carr21, f_max_Carr21, LN, params_LN, evolved, t)
        f_pbh_SLN = constraint_Carr(mc_Carr21, m_delta_Carr21, f_max_Carr21, SLN, params_SLN, evolved, t)
        f_pbh_CC3 = constraint_Carr(mc_Carr21, m_delta_Carr21, f_max_Carr21, CC3, params_CC3, evolved, t)
        
        # Peak mass for log-normal MF
        mp_LN = mc_Carr21 * np.exp(-sigmas_LN[j]**2)
        # Estimate peak mass of skew-lognormal MF
        mp_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[j], alpha=alphas_SLN[j], log_m_factor=3, n_steps=1000) for m_c in mc_Carr21]

        ax.plot(mp_LN, f_pbh_LN, color=colors[0], dashes=[6, 2], label="LN")        
        ax.plot(mp_SLN, f_pbh_SLN, color=colors[1], linestyle=(0, (5, 7)), label="SLN")            
        ax.plot(mc_Carr21, f_pbh_CC3, color=colors[2], linestyle="dashed", label="CC3")            
        ax.set_label("CC3")
        
        
        # Plot extended MF constraints from Korwar & Profumo (2023)
        # Path to extended MF constraints
        exponent_PL_lower = 2
        data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower) 
                
        data_filename_LN = data_folder + "/LN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
        data_filename_SLN = data_folder + "/SLN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
        data_filename_CC3 = data_folder + "/CC3_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
        
        mc_KP23_LN, f_PBH_KP23_LN = np.genfromtxt(data_filename_LN, delimiter="\t")
        mc_KP23_SLN, f_PBH_KP23_SLN = np.genfromtxt(data_filename_SLN, delimiter="\t")
        mp_KP23_CC3, f_PBH_KP23_CC3 = np.genfromtxt(data_filename_CC3, delimiter="\t")
 
        # Peak mass for log-normal MF
        mp_LN = mc_KP23_LN * np.exp(-sigmas_LN[j]**2)               
        # Estimate peak mass of skew-lognormal MF
        mp_KP23_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[j], alpha=alphas_SLN[j], log_m_factor=3, n_steps=1000) for m_c in mc_KP23_SLN]
 
        ax.plot(mp_LN, f_PBH_KP23_LN, color=colors[0], dashes=[6, 2], alpha=0.5)
        ax.plot(mp_KP23_SLN, f_PBH_KP23_SLN, color=colors[1], linestyle=(0, (5, 7)), alpha=0.5)
        ax.plot(mp_KP23_CC3, f_PBH_KP23_CC3, color=colors[2], linestyle="dashed", alpha=0.5)
        

        ax.set_xlabel("$m_p~[\mathrm{g}]$")
        ax.set_ylabel("$f_\mathrm{PBH}$")
        ax.legend(fontsize="x-small")
    
        ax.set_xscale("log")
        ax.set_yscale("log")
            
        ax.set_xlim(1e16, 5e18)
        ax.set_ylim(1e-6, 1)
        
        ax.set_title("C+ '21 and KP '23 constraints, $\Delta={:.1f}$".format(Deltas[j]))
        
        x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 10)
        ax.xaxis.set_major_locator(x_major)
        x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
        ax.xaxis.set_minor_locator(x_minor)
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        
        fig.tight_layout()

#%% Check how closely the constraints on beta prime (with the fitting functions) match Fig. 7 of Carr et al. (2021)
if "__main__" == __name__:
    
    m_min = 3e13    # Minimum mass corresponds to the smallest value that is shown in Fig. 7 of 2002.12778.
    m_max = 1e20
    
    m_delta_Carr21_Fig7, beta_prime_Carr21_Fig7 = load_data("/2002.12778/Carr+21_Fig7.csv")
    
    beta_prime_Carr21 = beta_prime_gamma_rays(m_delta_Carr21_Fig7, epsilon=0.4)
    
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(m_delta_Carr21_Fig7, beta_prime_Carr21_Fig7, label="From Fig. 7 of Carr et al. (2021)")
    ax.plot(m_delta_Carr21_Fig7, beta_prime_Carr21, linestyle="dotted", label="Fitting functions")
    ax.legend(fontsize="x-small")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$m_i~[\mathrm{g}]$")
    ax.set_ylabel(r"$\beta '$")
    fig.tight_layout()
    
#%% Compare data used for COMPTEL in Carr et al. (2021) and Auffinger (2022)
if "__main__" == __name__:
    
    fig, ax = plt.subplots(figsize=(7,7))

    # Load data used in Carr et al. (2021) [2002.12778]
    E_lower, flux_mid = load_data("Weidenspointner+2000/Weidenspointner+2000_Fig1_lower_E.csv")
    E_upper, flux_mid = load_data("Weidenspointner+2000/Weidenspointner+2000_Fig1_upper_E.csv")
    E_mid, flux_lower = load_data("Weidenspointner+2000/Weidenspointner+2000_Fig1_lower_flux.csv")
    E_mid, flux_upper = load_data("Weidenspointner+2000/Weidenspointner+2000_Fig1_upper_flux.csv")
    
    # Calculate error bars, and convert energy units to GeV
    error_flux_upper = (flux_upper - flux_mid) / 1e3
    error_flux_lower = (flux_mid - flux_lower) / 1e3
    error_E_upper = (E_upper - E_mid) / 1e3
    error_E_lower = (E_mid - E_lower) / 1e3
    E_mid /= 1e3
    flux_mid /= 1e3
      
    # Load data used in Auffinger (2022), from Ackermann et al. (2015) [1502.06116]
    file_path_data = "./../../Downloads/version_finale/scripts/Isatis/constraints/photons/"
    append = "COMPTEL_1502.06116"
    energies, energies_minus, energies_plus, flux, flux_minus, flux_plus = np.genfromtxt("%sflux_%s.txt"%(file_path_data, append), skip_header = 6).transpose()[0:6]

    ax.errorbar(E_mid, flux_mid, yerr=([error_flux_upper, error_flux_lower]), xerr=([error_E_lower, error_E_upper]), label="Carr et al. (2021)", linestyle="None")
    ax.errorbar(energies, flux*energies**2, yerr=([flux_minus, flux_plus]*energies**2), xerr=([energies_minus, energies_plus]), label="Auffinger (2022)", linestyle="None")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$E$ [GeV]")
    ax.set_ylabel('$E^2 {\\rm d}\Phi/{\\rm d}E\,\, ({\\rm GeV} \cdot {\\rm s}^{-1}\cdot{\\rm cm}^{-2} \cdot {\\rm sr}^{-1})$')
    ax.legend(fontsize="small")
    fig.tight_layout()
    
    
#%% Compare extragalactic gamma-ray background constraints calculated with PYTHIA and Hazma hadronisation tables.

from preliminaries import load_results_Isatis

if "__main__" == __name__:
    
    m_pbh_values = np.logspace(14, 17, 32)
    m_pbh_values_long = 10**np.arange(14, 17.04, 0.05)
    
    constraints_names, constraints_Hazma = load_results_Isatis(mf_string="EXGB_Hazma")   
    constraints_names, constraints_PYTHIA = load_results_Isatis(mf_string="EXGB_PYTHIA")
    constraints_names, constraints_PYTHIA_BBN = load_results_Isatis(mf_string="EXGB_PYTHIA_BBN")
    
    constraints_names_short = ["COMPTEL", "EGRET", "Fermi-LAT", "HEAO balloon"]
    
    fig, ax = plt.subplots(figsize=(7,7))
    
    colors = ["tab:orange", "tab:green", "tab:red", "tab:blue", "k"]
    
    for i in range(len(constraints_Hazma[0:8])):
        # Select constraints calculated from existing measurements of the extragalactic photon background.
        if i in (0, 2, 4, 7):
            print(constraints_names[i])
            ax.plot(m_pbh_values, constraints_Hazma[i], color=colors[int(i/2)], label=constraints_names_short[int(i/2)])
            ax.plot(m_pbh_values, constraints_PYTHIA[i], color=colors[int(i/2)], linestyle="None", marker="x")
            ax.plot(m_pbh_values_long, constraints_PYTHIA_BBN[i], color=colors[int(i/2)], linestyle="None", marker="+")

    ax.plot(0, 0, color="k", label="Hazma")    
    ax.plot(0, 0, color="k", linestyle="None", marker="x", label="PYTHIA (present-day)")
    ax.plot(0, 0, color="k", linestyle="None", marker="+", label="PYTHIA (BBN)")
    
    ax.legend(fontsize="xx-small")
    ax.set_xlabel("$m_i~[\mathrm{g}]$")
    ax.set_ylabel("$f_\mathrm{PBH}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e14, 1e18)
    ax.set_ylim(1e-10, 1)
    ax.grid()
    fig.tight_layout()
    fig.savefig("EXGB_Hazma_PYTHIA.pdf")
    fig.savefig("EXGB_Hazma_PYTHIA.png")
    
    
#%% Compare extended MF constraints obtained with a power-law MF to the delta-function MF constraints
from preliminaries import PL_MF
from plot_constraints import plotter_GC_Isatis, plotter_BC19, plotter_KP23, plotter_Subaru_Croon20, plotter_Sugiyama, plotter_GECCO

if "__main__" == __name__:
    
        title_string = ", PL MF, $\gamma=-1/2$"
    
        Deltas = [0]
        Delta_index = 0


        """Isatis Galactic Centre photon constraints"""   
        fig1, ax1 = plt.subplots(figsize=(7,7))
        fig1.suptitle("GC Isatis %s" % title_string)
      
        m_delta_values_loaded = np.logspace(11, 22, 1000)
        colors_evap = ["tab:orange", "tab:green", "tab:red", "tab:blue"]
        
        m_min_values = np.logspace(14, 20, 120)
        
        include_extrapolated = True
        data_folder_base = "./Data-tests/unevolved"
        exponent_PL_lower = 2
        
        plotter_GC_Isatis(Deltas, Delta_index, ax1, color="tab:grey", mf=None, show_label=True)
        plotter_GC_Isatis(Deltas, Delta_index, ax1, color="tab:blue", mf=PL_MF, evolved=False, approx=False)
        ax1.plot(0, 0, color="tab:blue", alpha=0.5, label="Extended MF")
        

        """Korwar & Profumo (2023) constraints"""
        fig2, ax2 = plt.subplots(figsize=(7,7))
        fig2.suptitle("KP '23 %s" % title_string)

        plotter_KP23(Deltas, Delta_index, ax2, color="tab:grey", mf=None, evolved=False, show_label=True)     
        plotter_KP23(Deltas, Delta_index, ax2, color="tab:blue", mf=PL_MF, evolved=False)
        
        # Sanity check: re-calculate results directly
        m_delta_values_loaded, f_max_loaded = load_data("./2302.04408/2302.04408_MW_diffuse_SPI.csv")
        # Power-law exponent to use between 1e15g and 1e16g.
        exponent_PL_upper = 2.0
        # Power-law exponent to use between 1e11g and 1e15g.
        exponent_PL_lower = 2.0
        
        m_delta_extrapolated_upper = np.logspace(15, 16, 11)
        m_delta_extrapolated_lower = np.logspace(11, 15, 41)
        
        f_max_extrapolated_upper = min(f_max_loaded) * np.power(m_delta_extrapolated_upper / min(m_delta_values_loaded), exponent_PL_upper)
        f_max_extrapolated_lower = min(f_max_extrapolated_upper) * np.power(m_delta_extrapolated_lower / min(m_delta_extrapolated_upper), exponent_PL_lower)
    
        f_max_total = np.concatenate((f_max_extrapolated_lower, f_max_extrapolated_upper, f_max_loaded))
        m_delta_total = np.concatenate((m_delta_extrapolated_lower, m_delta_extrapolated_upper, m_delta_values_loaded))

        f_PBH_sanity_check = 1 / np.array([np.trapz(PL_MF(m_delta_total, m_min, m_max=1e20) / f_max_total, m_delta_total) for m_min in m_min_values])
        ax2.plot(m_min_values, f_PBH_sanity_check, marker="+", linestyle="None", color="tab:blue", label="Extended MF (sanity check)")
            
        
        """Boudaud & Cirelli Voyager-1 constraints"""
        fig3, ax3 = plt.subplots(figsize=(7,7))
        fig3.suptitle("BC '19 %s" % title_string)
        
        plotter_BC19(Deltas, Delta_index, ax3, color="b", prop_A=True, with_bkg_subtr=False, mf=None)
        plotter_BC19(Deltas, Delta_index, ax3, color="b", prop_A=True, with_bkg_subtr=True, mf=None, evolved=False, linestyle="dotted")
        plotter_BC19(Deltas, Delta_index, ax3, color="r", prop_A=False, with_bkg_subtr=False, mf=None)
        plotter_BC19(Deltas, Delta_index, ax3, color="r", prop_A=False, with_bkg_subtr=True, mf=None, evolved=False, linestyle="dotted")
      
        plotter_BC19(Deltas, Delta_index, ax3, color="b", prop_A=True, with_bkg_subtr=False, mf=PL_MF, evolved=False, linestyle="None", marker="x")
        plotter_BC19(Deltas, Delta_index, ax3, color="b", prop_A=True, with_bkg_subtr=True, mf=PL_MF, evolved=False,  linestyle="None", marker="1")
        
        plotter_BC19(Deltas, Delta_index, ax3, color="r", prop_A=False, with_bkg_subtr=False, mf=PL_MF, evolved=False, linestyle="None", marker="x")
        plotter_BC19(Deltas, Delta_index, ax3, color="r", prop_A=False, with_bkg_subtr=True, mf=PL_MF, evolved=False, linestyle="None", marker="1")
        
        ax3.plot(0, 0, color="b", label="Prop A (delta func.)", alpha=0.5)        
        ax3.plot(0, 0, color="r", label="Prop B (delta func.)", alpha=0.5)
        ax3.plot(0, 0, color="k", label="w/o background subtraction (delta func.)")
        ax3.plot(0, 0, color="k", linestyle="dotted", label="w background subtraction (delta func.)")
        ax3.plot(0, 0, color="k", linestyle="None", marker="x", label="Extended MF (w/o background subtraction)")
        ax3.plot(0, 0, color="k", linestyle="None", marker="1", label="Extended MF (w/o background subtraction)")
    
    
        """Microlensing constraints (existing and prospective)"""
        fig4, ax4 = plt.subplots(figsize=(7,7))
        fig4.suptitle("Microlensing (existing and prospective) %s" % title_string, fontsize="x-small")
 
        plotter_Subaru_Croon20(Deltas, Delta_index, ax4, color="tab:blue", show_label=False)
        plotter_Sugiyama(Deltas, Delta_index, ax4, color="tab:orange", show_label=False)
 
        plotter_Subaru_Croon20(Deltas, Delta_index, ax4, mf=PL_MF, color="tab:blue", show_label=False, linestyle="None", marker="x")
        plotter_Sugiyama(Deltas, Delta_index, ax4, mf=PL_MF, color="tab:orange", linestyle="None", show_label=False, marker="x")
        
        ax4.plot(0, 0, color="k", label="Delta func.")
        ax4.plot(0, 0, marker="x", linestyle="None", color="k", label="Extended MF")
        ax4.plot(0, 0, color="tab:blue", label="Subaru-HSC")        
        ax4.plot(0, 0, color="tab:orange", linestyle="dashed", label="Prospective")
       
        
        """Coogan et al. (prospective evaporation constraints)"""
        fig5, ax5 = plt.subplots(figsize=(7,7))
        fig5.suptitle("GECCO %s" % title_string)
        
        plotter_GECCO(Deltas, Delta_index, ax5, color="tab:grey", mf=None, show_label=True)
        plotter_GECCO(Deltas, Delta_index, ax5, color="tab:blue", mf=PL_MF, show_label=False, evolved=False)
        
        ax5.plot(0, 0, color="tab:blue", label="Extended MF")
      
        
        """Other (existing) evaporation constraints"""
        fig6, ax6 = plt.subplots(figsize=(7,7))
        fig6.suptitle("Other evap. %s" % title_string)
        
        mp_511keV, f_PBH_511keV = np.genfromtxt("./Data-tests/unevolved/PL_exp_-2/PL_1912.01014_Carr_extrapolated_exp-2.txt")
        m_delta_511keV, f_max_511keV = load_data("1912.01014/1912.01014_Fig2_a__0_newaxes_2.csv")
        mp_CMB, f_PBH_CMB = np.genfromtxt("./Data-tests/unevolved/PL_exp_-2/PL_2108.13256_Carr_extrapolated_exp-2.txt")
        m_delta_CMB, f_max_CMB = load_data("2108.13256/2108.13256_Fig4_CMB.csv") 
        
        # Sanity check: re-calculate results directly
        exponent_PL_lower = -2
        m_delta_extrapolated_CMB = 10**np.arange(11, np.log10(min(m_delta_CMB))+0.01, 0.1)
        m_delta_extrapolated_511keV = 10**np.arange(11, np.log10(min(m_delta_511keV))+0.01, 0.1)
        f_max_extrapolated_511keV = f_max_511keV[0] * np.power(m_delta_extrapolated_511keV / min(m_delta_511keV), exponent_PL_lower)
        f_max_extrapolated_CMB = f_max_CMB[0] * np.power(m_delta_extrapolated_CMB / min(m_delta_CMB), exponent_PL_lower)
      
        f_max_total_511keV = np.concatenate((f_max_extrapolated_511keV, f_PBH_511keV))
        m_delta_total_511keV = np.concatenate((m_delta_extrapolated_511keV, m_delta_511keV))

        f_PBH_511keV_sanity_check = 1 / np.array([np.trapz(PL_MF(m_delta_511keV, m_min, m_max=1e20) / f_max_511keV, m_delta_511keV) for m_min in m_min_values])
        f_PBH_CMB_sanity_check = 1 / np.array([np.trapz(PL_MF(m_delta_CMB, m_min, m_max=1e20) / f_max_CMB, m_delta_CMB) for m_min in m_min_values])
       
        ax6.plot(m_delta_511keV, f_max_511keV, color="tab:blue", alpha=0.5, label="511 keV line (delta func.)")
        ax6.plot(m_delta_CMB, f_max_CMB, linestyle="dashed", color="tab:red", alpha=0.5, label="CMB (delta func.)")

        ax6.plot(mp_511keV, f_PBH_511keV, color="tab:blue", label="511 keV line (extended MF)")        
        ax6.plot(mp_CMB, f_PBH_CMB, linestyle="dashed", color="tab:red", label="CMB (extended MF)")

        ax6.plot(mp_511keV, f_PBH_511keV_sanity_check, linestyle="None", marker="x", color="tab:blue", label="511 keV line (extended MF, sanity check)")            
        ax6.plot(mp_CMB, f_PBH_CMB_sanity_check, linestyle="None", marker="x", color="tab:red", label="CMB (extended MF, sanity check)")        

        """EXGB photon constraints"""
        fig7, ax7 = plt.subplots(figsize=(7,7))
        fig7.suptitle("EXGB %s" % title_string)

        colors_evap = ["tab:orange", "tab:green", "tab:red", "tab:blue", "k"]
        constraints_names_short = ["COMPTEL_1502.06116", "COMPTEL_1107.0200", "EGRET_0405441", "EGRET_9811211", "Fermi-LAT_1410.3696", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200", "HEAO+balloon_9903492"]

        exponent_PL_lower = -2
        data_folder = "./Data-tests/unevolved/PL_exp_{:.0f}".format(exponent_PL_lower)
        constraints_names, f_max_Isatis = load_results_Isatis(mf_string="EXGB_Hazma")
        m_delta_values_loaded = np.logspace(14, 17, 32)

        f_PBH_instrument_PL = []
        
        for i in range(len(constraints_names_short)):
            
            if i in (0, 2, 4, 7):

                # Load constraints for an evolved extended mass function obtained from each instrument
                data_filename_PL = data_folder + "/PL_EXGB_%s" % constraints_names_short[i]  + "_Carr_unevolved.txt"
                mp_PL, f_PBH_PL = np.genfromtxt(data_filename_PL, delimiter="\t")
                                                            
                # Plot the tightest constraint (of the different instruments) for each peak mass
                ax7.plot(m_delta_values_loaded, f_max_Isatis[i], color=colors_evap[int(i/2)], label=constraints_names_short[i])
                ax7.plot(mp_PL, f_PBH_PL, color=colors_evap[int(i/2)], marker="x", linestyle="None")
        
        ax7.plot(color="k", label="Delta-func.")
        ax7.plot(color="k", marker="x", linestyle="None", label="Extended MF")
        
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
            ax.set_xlabel("$m_\mathrm{min}~[\mathrm{g}]$")
            ax.set_ylabel("$f_\mathrm{PBH}$")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(1e-10, 1)
            ax.legend(fontsize="xx-small")
            
        ax4.set_ylim(1e-3, 1)    
            
        for fig in [fig1, fig2, fig3, fig4, fig5, fig6, fig7]:
            fig.tight_layout()
            
#%% Check which constraint is being shown from the Voyager-1 delta function MF constraints

with_bkg_subtr = True

if with_bkg_subtr:
    m_delta_values_propA, f_max_propA = load_data("1807.03075/1807.03075_prop_A_bkg.csv")
    m_delta_values_propB, f_max_propB = load_data("1807.03075/1807.03075_prop_B_bkg.csv")
else:
    m_delta_values_propA, f_max_propA = load_data("1807.03075/1807.03075_prop_A_nobkg.csv")
    m_delta_values_propB, f_max_propB = load_data("1807.03075/1807.03075_prop_B_nobkg_lower.csv")

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(m_delta_values_propA, f_max_propA, label="prop A")
ax.plot(m_delta_values_propB, f_max_propB, label="prop B (lower)")
ax.set_ylabel("$f_\mathrm{PBH}$")
ax.set_xlabel("$m~[\mathrm{g}]$")
ax.legend()
ax.set_xscale("log")
ax.set_yscale("log")
fig.tight_layout()


#%% Compare microlensing constraints obtained with an evolved MF and without

from plot_constraints import plotter_Subaru_Croon20, plotter_Sugiyama

if "__main__" == __name__:

    mc_subaru = 10**np.linspace(17, 29, 1000)
        
    # Mass function parameter values, from 2009.03204.
    [Deltas, sigmas_LN, ln_mc_SL, mp_SL, sigmas_SLN, alphas_SLN, mp_CC, alphas_CC, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
            
    for i in range(len(Deltas)):
        
        fig1, ax1 = plt.subplots(figsize=(7, 7))
        fig2, ax2 = plt.subplots(figsize=(7, 7))
               
        # Unevolved MF constraints
        plotter_Subaru_Croon20(Deltas, i, ax1, mf=LN, color="r", linestyle="dotted")
        plotter_Subaru_Croon20(Deltas, i, ax1, mf=SLN, color="b", linestyle="dotted")
        plotter_Subaru_Croon20(Deltas, i, ax1, mf=CC3, color="g", linestyle="dotted")

        plotter_Sugiyama(Deltas, i, ax2, mf=LN, color="r", linestyle="dotted")
        plotter_Sugiyama(Deltas, i, ax2, mf=SLN, color="b", linestyle="dotted")
        plotter_Sugiyama(Deltas, i, ax2, mf=CC3, color="g", linestyle="dotted")

        # Evolved MF constraints (Subaru-HSC):            
        data_filename_SLN = "./Data-tests/SLN_HSC_Carr_Delta={:.1f}_evolved.txt".format(Deltas[i])
        data_filename_CC3 = "./Data-tests/CC3_HSC_Carr_Delta={:.1f}_evolved.txt".format(Deltas[i])
        data_filename_LN = "./Data-tests/LN_HSC_Carr_Delta={:.1f}_evolved.txt".format(Deltas[i])
        
        mc_LN, f_PBH_LN = np.genfromtxt(data_filename_LN, delimiter="\t")
        mc_SLN, f_PBH_SLN = np.genfromtxt(data_filename_SLN, delimiter="\t")
        mp_CC3, f_PBH_CC3 = np.genfromtxt(data_filename_CC3, delimiter="\t")
        
        mp_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_SLN]
        mp_LN = mc_LN * np.exp(-sigmas_LN[i]**2)
                    
        ax1.plot(mp_LN, f_PBH_LN, color="r", marker="None", alpha=0.5, label="Evolved")
        ax1.plot(mp_SLN, f_PBH_SLN, color="b", marker="None", alpha=0.5, label="Evolved")
        ax1.plot(mp_CC3, f_PBH_CC3, color="g", marker="None", alpha=0.5, label="Evolved")
 
        # Evolved MF constraints (Sugiyama):            
        data_filename_SLN = "./Data-tests/SLN_Sugiyama20_Carr_Delta={:.1f}_evolved.txt".format(Deltas[i])
        data_filename_CC3 = "./Data-tests/CC3_Sugiyama20_Carr_Delta={:.1f}_evolved.txt".format(Deltas[i])
        data_filename_LN = "./Data-tests/LN_Sugiyama20_Carr_Delta={:.1f}_evolved.txt".format(Deltas[i])
        
        mc_LN, f_PBH_LN = np.genfromtxt(data_filename_LN, delimiter="\t")
        mc_SLN, f_PBH_SLN = np.genfromtxt(data_filename_SLN, delimiter="\t")
        mp_CC3, f_PBH_CC3 = np.genfromtxt(data_filename_CC3, delimiter="\t")
        
        mp_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_SLN]
        mp_LN = mc_LN * np.exp(-sigmas_LN[i]**2)
                    
        ax2.plot(mp_LN, f_PBH_LN, color="r", marker="None", alpha=0.3, label="Evolved")
        ax2.plot(mp_SLN, f_PBH_SLN, color="b", marker="None", alpha=0.3, label="Evolved")
        ax2.plot(mp_CC3, f_PBH_CC3, color="g", marker="None", alpha=0.3, label="Evolved")
   
        for ax in [ax1, ax2]:
            if Deltas[i] < 5:
                ax.set_xlim(1e21, 1e29)
            else:
                ax.set_xlim(1e19, 1e29)
               
            ax.set_ylim(1e-3, 1)
            ax.set_ylabel("$f_\mathrm{PBH}$")
            ax.set_xlabel("$m_p~[\mathrm{g}]$")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend(fontsize="x-small")

        ax1.set_title("Subaru-HSC, $\Delta={:.1f}$".format(Deltas[i]))
        ax2.set_title("Sugiyama et al. (2020), $\Delta={:.1f}$".format(Deltas[i]))
        fig1.tight_layout()
        fig2.tight_layout()

