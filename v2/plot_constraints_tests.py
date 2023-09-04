#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:39:43 2023

@author: ppxmg2
"""

# Script plots results obtained for the extended PBH mass functions given by
# the fitting functions in 2009.03204.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from preliminaries import load_data, m_max_SLN, load_results_Isatis, envelope

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
                        

#%% Tests of the results obtained using different power-law exponents in f_max at low masses (Korwar & Profumo (2023))

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
        fig.tight_layout()
        fig.suptitle("$\Delta={:.1f}$".format(Deltas[j]))

#%% Tests of the results obtained using different power-law exponents in f_max at low masses (Berteaud et al. (2022))

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
            
        ax0.plot(m_delta_values, f_max, color="tab:grey")
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

        ax0.legend(fontsize="x-small", title="PL exponent in $f_\mathrm{max}$ \n ($m < 3e16~\mathrm{g}$)")        
        ax1.legend(fontsize="x-small", title="PL exponent in $f_\mathrm{max}$ \n ($m < 3e16~\mathrm{g}$)")
        fig.tight_layout()
        fig.suptitle("$\Delta={:.1f}$".format(Deltas[j]))
        
        
#%% Tests of the results obtained using different power-law exponents in f_max at low masses 
# Constraints from 2202.07483 (Voyager-1 electron / positron detections).

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    mc_values = np.logspace(14, 20, 120)

    # Array of power law exponents to use at masses below 1e15g
    exponents_PL_lower = [0, 2, 3, 4]
    
    linestyles = ["solid", "dashed", "dashdot", "dotted"]
    
    # Boolean determines which propagation model to load data from
    prop_A = True
    prop_B = not prop_A
    
    if prop_A:
        m_delta_values, f_max = load_data("1807.03075/1807.03075_prop_A_bkg.csv")
        prop_string = "prop_A"
    elif prop_B:  
        m_delta_values, f_max = load_data("1807.03075/1807.03075_prop_B_bkg.csv")
        prop_string = "prop_B"
    
    for j in range(len(Deltas)):
        
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
            
        ax0.plot(m_delta_values, f_max, color="tab:grey")
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
                ax.set_xlim(1e16, 7e18)
               
            ax.set_ylim(1e-6, 1)
            ax.set_ylabel("$f_\mathrm{PBH}$")
            ax.set_xlabel("$m_p~[\mathrm{g}]$")
            ax.set_xscale("log")
            ax.set_yscale("log")

        ax0.legend(fontsize="x-small", title="PL exponent in $f_\mathrm{max}$ \n " + "($m < {:.0e}".format(min(m_delta_values)) + "~\mathrm{g}$)")        
        ax1.legend(fontsize="x-small", title="PL exponent in $f_\mathrm{max}$ \n " + "($m < {:.0e}".format(min(m_delta_values)) + "~\mathrm{g}$)")
        fig.tight_layout()
        fig.suptitle("$\Delta={:.1f}$".format(Deltas[j]))



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
    
    LN = False
    SLN = True
    CC3 = False
        
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
            if LN:
                ax.plot(mp_LN_unevolved, f_PBH_LN_unevolved[i], label=constraints_names[i], color=colors_evap[i])
            elif SLN:
                ax.plot(mp_SLN_unevolved, f_PBH_SLN_unevolved[i], label=constraints_names[i], color=colors_evap[i])
            elif CC3:
                ax.plot(mc_values_old, f_PBH_CC3_unevolved[i], label=constraints_names[i], color=colors_evap[i])
                
            # Load and plot results for the unevolved mass functions
            data_filename_LN_unevolved_approx = "./Data-tests/unevolved" + "/LN_GC_%s" % constraints_names_short[i] + "_Carr_Delta={:.1f}_approx_unevolved.txt".format(Deltas[j])
            data_filename_SLN_unevolved_approx = "./Data-tests/unevolved" + "/SLN_GC_%s" % constraints_names_short[i]  + "_Carr_Delta={:.1f}_approx_unevolved.txt".format(Deltas[j])
            data_filename_CC3_unevolved_approx = "./Data-tests/unevolved" + "/CC3_GC_%s" % constraints_names_short[i]  + "_Carr_Delta={:.1f}_approx_unevolved.txt".format(Deltas[j])

            mc_LN_unevolved_approx, f_PBH_LN_unevolved_approx = np.genfromtxt(data_filename_LN_unevolved_approx, delimiter="\t")
            mc_SLN_unevolved_approx, f_PBH_SLN_unevolved_approx = np.genfromtxt(data_filename_SLN_unevolved_approx, delimiter="\t")
            mp_CC3_unevolved_approx, f_PBH_CC3_unevolved_approx = np.genfromtxt(data_filename_CC3_unevolved_approx, delimiter="\t")
            
            mp_SLN_unevolved_approx = [m_max_SLN(m_c, sigma=sigmas_SLN[j], alpha=alphas_SLN[j], log_m_factor=3, n_steps=1000) for m_c in mc_SLN_unevolved_approx]
            mp_LN_unevolved_approx = mc_LN_unevolved_approx * np.exp(-sigmas_LN[j]**2)
            
            if LN:
                ax.plot(mp_LN_unevolved_approx, f_PBH_LN_unevolved_approx, color=colors_evap[i], linestyle="None", marker="x")
            elif SLN:
                ax.plot(mp_SLN_unevolved_approx, f_PBH_SLN_unevolved_approx, color=colors_evap[i], linestyle="None", marker="x")
            elif CC3:
                ax.plot(mp_CC3_unevolved_approx, f_PBH_CC3_unevolved_approx, color=colors_evap[i], linestyle="None", marker="x")                

            # Load and plot results for the 'evolved' mass functions evaluated at the initial time t_init = 0
            data_filename_LN_t_init_approx = "./Data-tests/t_initial" + "/LN_GC_%s" % constraints_names_short[i] + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[j])
            data_filename_SLN_t_init_approx = "./Data-tests/t_initial" + "/SLN_GC_%s" % constraints_names_short[i]  + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[j])
            data_filename_CC3_t_init_approx = "./Data-tests/t_initial" + "/CC3_GC_%s" % constraints_names_short[i]  + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[j])

            mc_LN_t_init_approx, f_PBH_LN_t_init_approx = np.genfromtxt(data_filename_LN_t_init_approx, delimiter="\t")
            mc_SLN_t_init_approx, f_PBH_SLN_t_init_approx = np.genfromtxt(data_filename_SLN_t_init_approx, delimiter="\t")
            mp_CC3_t_init_approx, f_PBH_CC3_t_init_approx = np.genfromtxt(data_filename_CC3_t_init_approx, delimiter="\t")
            
            mp_SLN_t_init_approx = [m_max_SLN(m_c, sigma=sigmas_SLN[j], alpha=alphas_SLN[j], log_m_factor=3, n_steps=1000) for m_c in mc_SLN_t_init_approx]
            mp_LN_t_init_approx = mc_LN_t_init_approx * np.exp(-sigmas_LN[j]**2)
            
            if LN:
                ax.plot(mp_LN_t_init_approx, f_PBH_LN_t_init_approx, color=colors_evap[i], linestyle="None", marker="+")    
                ax1.plot(mp_LN_unevolved, np.interp(mp_LN_unevolved, mp_LN_unevolved_approx, f_PBH_SLN_unevolved_approx) / f_PBH_LN_unevolved[i] - 1, color=colors_evap[i], linestyle="None", marker="+")
            elif SLN:
                ax.plot(mp_SLN_t_init_approx, f_PBH_SLN_t_init_approx, color=colors_evap[i], linestyle="None", marker="+")   
                ax1.plot(mp_SLN_unevolved, np.interp(mp_SLN_unevolved, mp_SLN_unevolved_approx, f_PBH_SLN_unevolved_approx) / f_PBH_SLN_unevolved[i] - 1, color=colors_evap[i], linestyle="None", marker="+")
            elif CC3:
                ax.plot(mp_CC3_t_init_approx, f_PBH_CC3_t_init_approx, color=colors_evap[i], linestyle="None", marker="+")
                ax1.plot(mc_values_old, np.interp(mc_values_old, mp_CC3_unevolved_approx, f_PBH_CC3_unevolved_approx) / f_PBH_CC3_unevolved[i] - 1, color=colors_evap[i], linestyle="None", marker="+")
                
            print(f_PBH_LN_unevolved_approx[0:5])
            print(f_PBH_LN_t_init_approx[0:5])
            
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

                # Load constraints for an evolved extended mass function obtained from each instrument
                data_filename_LN = data_folder + "/LN_GC_%s" % constraints_names_short[i] + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[j])
                data_filename_SLN = data_folder + "/SLN_GC_%s" % constraints_names_short[i]  + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[j])
                data_filename_CC3 = data_folder + "/CC3_GC_%s" % constraints_names_short[i]  + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[j])
                    
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
    
            ax0.legend(fontsize="x-small", title="PL exponent in $f_\mathrm{max}$ \n ($m < 10^{13}~\mathrm{g}$)")        
            ax1.legend(fontsize="x-small", title="PL exponent in $f_\mathrm{max}$ \n ($m < 10^{13}~\mathrm{g}$)")
            fig.tight_layout()
            fig.suptitle("$\Delta={:.1f}$".format(Deltas[j]))


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
        fig.tight_layout()
        fig.suptitle("$\Delta={:.1f}$".format(Deltas[j]))


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
        fig.tight_layout()
        fig.suptitle("$\Delta={:.1f}$".format(Deltas[j]))


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
        fig.tight_layout()
        fig.suptitle("$\Delta={:.1f}$".format(Deltas[i]))


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
        fig.tight_layout()
        fig.suptitle("Convergence test ($\Delta={:.1f}$)".format(Deltas[j]))


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
                data_filename_LN = data_folder + "/LN_GC_%s" % constraints_names_short[i] + "_Carr_Delta={:.1f}_approx_mmin=1e15g.txt".format(Deltas[j])
                data_filename_SLN = data_folder + "/SLN_GC_%s" % constraints_names_short[i]  + "_Carr_Delta={:.1f}_approx_mmin=1e15g.txt".format(Deltas[j])
                data_filename_CC3 = data_folder + "/CC3_GC_%s" % constraints_names_short[i]  + "_Carr_Delta={:.1f}_approx_mmin=1e15g.txt".format(Deltas[j])
                    
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


