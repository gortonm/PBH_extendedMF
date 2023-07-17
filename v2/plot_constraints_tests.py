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
plt.style.use('tableau-colorblind10')

#%% Tests evaluating the mass functions at the initial time (or unevolved mass functions), and comparing to results obtained before June 2023.

if "__main__" == __name__:
    
    # If True, plot the evaporation constraints used by Isatis (from COMPTEL, INTEGRAL, EGRET and Fermi-LAT)
    plot_GC_Isatis = False
    # If True, plot the evaporation constraints shown in Korwar & Profumo (2023) [2302.04408]
    plot_KP23 = not plot_GC_Isatis
    # If True, use extended MF constraint calculated from the delta-function MF extrapolated down to 1e11g using a power-law fit
    include_extrapolated = True
    if not plot_KP23:
        include_extrapolated = False
    
    # Choose colors to match those from Fig. 5 of 2009.03204
    colors = ['tab:grey', 'r', 'b', 'g', 'k']
    
    # Parameters used for convergence tests in Galactic Centre constraints.
    cutoff = 1e-4
    delta_log_m = 1e-3
    E_number = 500    
    
    if E_number < 1e3:
        energies_string = "E{:.0f}".format(E_number)
    else:
        energies_string = "E{:.0f}".format(np.log10(E_number))
            
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
        
        if plot_GC_Isatis:
            
            plt.suptitle("Galactic Centre photon constraints (Isatis), $\Delta={:.1f}$".format(Deltas[i]), fontsize="small")
            
            # Monochromatic MF constraints
            m_mono_evap = np.logspace(11, 21, 1000)

            constraints_names_evap, f_PBHs_GC_mono = load_results_Isatis(modified=True)
            f_PBH_mono_evap = envelope(f_PBHs_GC_mono)
            m_mono_Subaru, f_PBH_mono_Subaru = load_data("2007.12697/Subaru-HSC_2007.12697_dx=5.csv")

            mc_values = np.logspace(14, 20, 120)
            
            # Load constraints from Galactic Centre photons.
            mc_values_old = np.logspace(14, 19, 100)
            fname_base_CC3 = "CC_D={:.1f}_dm{:.0f}_".format(Deltas[i], -np.log10(delta_log_m)) + energies_string + "_c{:.0f}".format(-np.log10(cutoff))
            fname_base_SLN = "SL_D={:.1f}_dm{:.0f}_".format(Deltas[i], -np.log10(delta_log_m)) + energies_string + "_c{:.0f}".format(-np.log10(cutoff))
            fname_base_LN = "LN_D={:.1f}_dm{:.0f}_".format(Deltas[i], -np.log10(delta_log_m)) + energies_string + "_c{:.0f}".format(-np.log10(cutoff))

            constraints_names_evap, f_PBHs_GC_SLN = load_results_Isatis(mf_string=fname_base_SLN, modified=True)
            constraints_names_evap, f_PBHs_GC_CC3 = load_results_Isatis(mf_string=fname_base_CC3, modified=True)
            constraints_names, f_PBHs_GC_LN = load_results_Isatis(mf_string=fname_base_LN, modified=True)
   
            f_PBH_GC_SLN = envelope(f_PBHs_GC_SLN)
            f_PBH_GC_CC3 = envelope(f_PBHs_GC_CC3)
            f_PBH_GC_LN = envelope(f_PBHs_GC_LN)
            
            
            mc_GC_SLN_t_init, f_PBH_GC_SLN_t_init = np.genfromtxt("./Data-tests/t_initial/SLN_GC_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
            mp_GC_CC3_t_init, f_PBH_GC_CC3_t_init = np.genfromtxt("./Data-tests/t_initial/CC3_GC_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
            mc_GC_LN_t_init, f_PBH_GC_LN_t_init = np.genfromtxt("./Data-tests/t_initial/LN_GC_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
            
            mc_GC_SLN_unevolved, f_PBH_GC_SLN_unevolved = np.genfromtxt("./Data-tests/unevolved/SLN_GC_Carr_Delta={:.1f}_unevolved.txt".format(Deltas[i]), delimiter="\t")
            mp_GC_CC3_unevolved, f_PBH_GC_CC3_unevolved = np.genfromtxt("./Data-tests/unevolved/CC3_GC_Carr_Delta={:.1f}_unevolved.txt".format(Deltas[i]), delimiter="\t")
            mc_GC_LN_unevolved, f_PBH_GC_LN_unevolved = np.genfromtxt("./Data-tests/unevolved/LN_GC_Carr_Delta={:.1f}_unevolved.txt".format(Deltas[i]), delimiter="\t")


            # Estimate peak mass of skew-lognormal MF
            mp_SLN_evap_old = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_values_old]
            mp_GC_SLN_t_init = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_GC_SLN_t_init]
            mp_GC_SLN_unevolved = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_GC_SLN_unevolved]
            mp_Subaru_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_Carr_SLN]
            
            ax0.plot(mp_SLN_evap_old, f_PBH_GC_SLN, color=colors[2], linestyle=(0, (5, 7)))
            ax0.plot(mc_values_old, f_PBH_GC_CC3, color=colors[3], linestyle="dashed")
            ax1.plot(mp_SLN_evap_old, f_PBH_GC_SLN, color=colors[2], linestyle=(0, (5, 7)))
            ax1.plot(mc_values_old, f_PBH_GC_CC3, color=colors[3], linestyle="dashed")

            ax0.plot(mp_GC_SLN_t_init, f_PBH_GC_SLN_t_init, color=colors[2], linestyle="None", marker="x")
            ax0.plot(mp_GC_CC3_t_init, f_PBH_GC_CC3_t_init, color=colors[3], linestyle="None", marker="x")
            ax1.plot(mp_GC_SLN_t_init, f_PBH_GC_SLN_t_init, color=colors[2], linestyle="None", marker="x")
            ax1.plot(mp_GC_CC3_t_init, f_PBH_GC_CC3_t_init, color=colors[3], linestyle="None", marker="x")

            ax0.plot(mp_GC_SLN_unevolved, f_PBH_GC_SLN_unevolved, color=colors[2], linestyle="None", marker="+")
            ax0.plot(mp_GC_CC3_unevolved, f_PBH_GC_CC3_unevolved, color=colors[3], linestyle="None", marker="+")
            ax1.plot(mp_GC_SLN_unevolved, f_PBH_GC_SLN_unevolved, color=colors[2], linestyle="None", marker="+")
            ax1.plot(mp_GC_CC3_unevolved, f_PBH_GC_CC3_unevolved, color=colors[3], linestyle="None", marker="+")
                                                                   
            ax0.plot(mc_values_old * np.exp(-sigmas_LN[i]**2), f_PBH_GC_LN, color=colors[1], dashes=[6, 2], label="LN")
            ax0.plot(mc_GC_LN_t_init * np.exp(-sigmas_LN[i]**2), f_PBH_GC_LN_t_init, color=colors[1], linestyle="None", marker="+")
            ax0.plot(mc_GC_LN_unevolved * np.exp(-sigmas_LN[i]**2), f_PBH_GC_LN_unevolved, color=colors[1], linestyle="None", marker="x")

            ax1.plot(mc_values_old * np.exp(-sigmas_LN[i]**2), f_PBH_GC_LN, color=colors[1], dashes=[6, 2])
            
            ax0.plot(0, 0, linestyle="None", color="k", marker="x", label="Test: $t=0$")
            ax0.plot(0, 0, linestyle="None", color="k", marker="+", label="Test: unevolved")

        
        elif plot_KP23:
            
            if include_extrapolated:
                fig.suptitle("Using 511 keV line constraints (Korwar \& Profumo 2023), $\Delta={:.1f}$ \n $f_".format(Deltas[i]) + "\mathrm{max}(m)$" + " extrapolated below " + "$m=10^{16}" + "~\mathrm{g}$", fontsize="small")
            else:
                fig.suptitle("Using 511 keV line constraints (Korwar \& Profumo 2023), $\Delta={:.1f}$".format(Deltas[i]))
           
            # Monochromatic MF constraints
            m_mono_evap, f_PBH_mono_evap = load_data("2302.04408/2302.04408_MW_diffuse_SPI.csv")
            m_mono_Subaru, f_PBH_mono_Subaru = load_data("2007.12697/Subaru-HSC_2007.12697_dx=5.csv")

            # Load constraints from Galactic Centre 511 keV line emission (from 2302.04408).            
            if include_extrapolated:
                mc_KP23_SLN, f_PBH_KP23_SLN = np.genfromtxt("./Data-old/SLN_2302.04408_Carr_Delta={:.1f}_extrapolated.txt".format(Deltas[i]), delimiter="\t")
                mp_KP23_CC3, f_PBH_KP23_CC3 = np.genfromtxt("./Data-old/CC3_2302.04408_Carr_Delta={:.1f}_extrapolated.txt".format(Deltas[i]), delimiter="\t")
                mc_KP23_LN, f_PBH_KP23_LN = np.genfromtxt("./Data-old/LN_2302.04408_Carr_Delta={:.1f}_extrapolated.txt".format(Deltas[i]), delimiter="\t")

                mc_KP23_SLN_t_init, f_PBH_KP23_SLN_t_init = np.genfromtxt("./Data-tests/t_initial/SLN_2302.04408_Carr_Delta={:.1f}_extrapolated.txt".format(Deltas[i]), delimiter="\t")
                mp_KP23_CC3_t_init, f_PBH_KP23_CC3_t_init = np.genfromtxt("./Data-tests/t_initial/CC3_2302.04408_Carr_Delta={:.1f}_extrapolated.txt".format(Deltas[i]), delimiter="\t")
                mc_KP23_LN_t_init, f_PBH_KP23_LN_t_init = np.genfromtxt("./Data-tests/t_initial/LN_2302.04408_Carr_Delta={:.1f}_extrapolated.txt".format(Deltas[i]), delimiter="\t")
                
                mc_KP23_SLN_unevolved, f_PBH_KP23_SLN_unevolved = np.genfromtxt("./Data-tests/unevolved/SLN_2302.04408_Carr_Delta={:.1f}_extrapolated.txt".format(Deltas[i]), delimiter="\t")
                mp_KP23_CC3_unevolved, f_PBH_KP23_CC3_unevolved = np.genfromtxt("./Data-tests/unevolved/CC3_2302.04408_Carr_Delta={:.1f}_extrapolated.txt".format(Deltas[i]), delimiter="\t")
                mc_KP23_LN_unevolved, f_PBH_KP23_LN_unevolved = np.genfromtxt("./Data-tests/unevolved/LN_2302.04408_Carr_Delta={:.1f}_extrapolated.txt".format(Deltas[i]), delimiter="\t")

            else:
                mc_KP23_SLN, f_PBH_KP23_SLN = np.genfromtxt("./Data-old/SLN_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
                mp_KP23_CC3, f_PBH_KP23_CC3 = np.genfromtxt("./Data-old/CC3_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
                mc_KP23_LN, f_PBH_KP23_LN = np.genfromtxt("./Data-old/LN_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")

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
            xmin_evap, xmax_evap = 1e16, 2.5e17
            xmin_HSC, xmax_HSC = 1e21, 1e29
            ymin, ymax = 1e-4, 1
            
            if plot_KP23:
                xmin_evap, xmax_evap = 1e16, 7e17
                ymin, ymax = 1e-5, 1
        
        else:
            xmin_evap, xmax_evap = 1e16, 7e17
            xmin_HSC, xmax_HSC = 9e18, 1e29
            ymin, ymax = 3e-6, 1
            
            if plot_KP23:
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
       
        for ax in [ax0, ax1, ax2]:
            ax.set_xlabel("$m_p~[\mathrm{g}]$")
            ax.plot(m_mono_evap, f_PBH_mono_evap, color=colors[0], label="Delta function", linestyle="dotted", linewidth=2)
            ax.plot(m_mono_Subaru, f_PBH_mono_Subaru, color=colors[0], linestyle="dotted", linewidth=2)
            ax.set_ylabel("$f_\mathrm{PBH}$")
            ax.set_xscale("log")
            ax.set_yscale("log")

        ax0.legend(fontsize="xx-small")
        fig.tight_layout()
        
        if plot_GC_Isatis:
            fig.savefig("./Results/Figures/fPBH_Delta={:.1f}_GC_Isatis.pdf".format(Deltas[i]))
            fig.savefig("./Results/Figures/fPBH_Delta={:.1f}_GC_Isatis.png".format(Deltas[i]))
            
        elif plot_KP23:
            if include_extrapolated:
                fig.savefig("./Results/Figures/fPBH_Delta={:.1f}_KP23_extrapolated.pdf".format(Deltas[i]))
                fig.savefig("./Results/Figures/fPBH_Delta={:.1f}_KP23_extrapolated.png".format(Deltas[i]))
            else:
                fig.savefig("./Results/Figures/fPBH_Delta={:.1f}_KP23.pdf".format(Deltas[i]))
                fig.savefig("./Results/Figures/fPBH_Delta={:.1f}_KP23.png".format(Deltas[i]))
                

#%% Tests of the results obtained using different power-law slopes in f_max at low masses (Korwar & Profumo (2023))

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    mc_values = np.logspace(14, 20, 120)

    # Array of power law exponents to use at masses below 1e15g
    slopes_PL_lower = [2, 3, 4]
    
    linestyles = ["dashed", "dashdot", "dotted"]
    
    for j in range(len(Deltas)):
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        ax0 = axes[0][0]
        ax1 = axes[0][1]
        ax2 = axes[1][0]
        ax3 = axes[1][1]
            
        for k, slope_PL_lower in enumerate(slopes_PL_lower):
            
            data_folder = "./Data-tests/PL_slope_{:.0f}".format(slope_PL_lower) 
    
            data_filename_LN = data_folder + "/LN_2302.04408_Carr_Delta={:.1f}_extrapolated_slope{:.0f}.txt".format(Deltas[j], slope_PL_lower)
            data_filename_SLN = data_folder + "/SLN_2302.04408_Carr_Delta={:.1f}_extrapolated_slope{:.0f}.txt".format(Deltas[j], slope_PL_lower)
            data_filename_CC3 = data_folder + "/CC3_2302.04408_Carr_Delta={:.1f}_extrapolated_slope{:.0f}.txt".format(Deltas[j], slope_PL_lower)
            
            mc_KP23_LN_evolved, f_PBH_KP23_LN_evolved = np.genfromtxt(data_filename_LN, delimiter="\t")
            mc_KP23_SLN_evolved, f_PBH_KP23_SLN_evolved = np.genfromtxt(data_filename_SLN, delimiter="\t")
            mp_KP23_CC3_evolved, f_PBH_KP23_CC3_evolved = np.genfromtxt(data_filename_CC3, delimiter="\t")
            mp_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[j], alpha=alphas_SLN[j], log_m_factor=3, n_steps=1000) for m_c in mc_KP23_SLN_evolved]
            mp_LN = mc_KP23_LN_evolved * np.exp(-sigmas_LN[j]**2)
            
            m_mono_values, f_max = load_data("2302.04408/2302.04408_MW_diffuse_SPI.csv")
                                
            # Power-law slope to use between 1e15g and 1e16g (motivated by mass-dependence of the positron spectrum emitted over energy)
            slope_PL_upper = 2.0
            
            m_mono_extrapolated_upper = np.logspace(15, 16, 11)
            m_mono_extrapolated_lower = np.logspace(11, 15, 41)
            f_max_extrapolated_upper = min(f_max) * np.power(m_mono_extrapolated_upper / min(m_mono_values), slope_PL_upper)
            f_max_extrapolated_lower = min(f_max_extrapolated_upper) * np.power(m_mono_extrapolated_lower / min(m_mono_extrapolated_upper), slope_PL_lower)
                        
            ax1.plot(mp_LN, f_PBH_KP23_LN_evolved, linestyle=linestyles[k], color="r", marker="None")
            ax2.plot(mp_SLN, f_PBH_KP23_SLN_evolved, linestyle=linestyles[k], color="b", marker="None")
            ax3.plot(mp_KP23_CC3_evolved, f_PBH_KP23_CC3_evolved, linestyle=linestyles[k], color="g", marker="None")
            
            ax1.set_title("LN")
            ax2.set_title("SLN")
            ax3.set_title("CC3")
            
            ax1.plot(0, 0, marker="None", linestyle=linestyles[k], color="k", label="{:.0f}".format(slope_PL_lower))
            ax0.plot(m_mono_extrapolated_lower, f_max_extrapolated_lower, color=(0.5294, 0.3546, 0.7020), linestyle=linestyles[k], label="{:.0f}".format(slope_PL_lower))
            
        ax0.plot(np.concatenate((m_mono_extrapolated_upper, m_mono_values)), np.concatenate((f_max_extrapolated_upper, f_max)), color=(0.5294, 0.3546, 0.7020))
        ax0.set_xlabel("$m$ [g]")
        ax0.set_ylabel("$f_\mathrm{max}$")
        ax0.set_xscale("log")
        ax0.set_yscale("log")
        ax0.set_xlim(min(m_mono_extrapolated_lower), 1e18)
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

        ax0.legend(fontsize="x-small", title="PL slope in $f_\mathrm{max}$ \n ($m < 10^{15}~\mathrm{g}$)")        
        ax1.legend(fontsize="x-small", title="PL slope in $f_\mathrm{max}$ \n ($m < 10^{15}~\mathrm{g}$)")
        fig.tight_layout()
        fig.suptitle("$\Delta={:.1f}$".format(Deltas[j]))
