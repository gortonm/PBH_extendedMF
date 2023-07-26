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
    
    # If True, plot the evaporation constraints used by Isatis (from COMPTEL, INTEGRAL, EGRET and Fermi-LAT)
    plot_GC_Isatis = True
    # If True, plot the evaporation constraints shown in Korwar & Profumo (2023) [2302.04408]
    plot_KP23 = not plot_GC_Isatis
    # If True, use extended MF constraint calculated from the delta-function MF extrapolated down to 1e11g using a power-law fit
    include_extrapolated = False
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
            
            # Load constraints from Galactic Centre photons
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
            #ax0.plot(mc_GC_LN_t_init * np.exp(-sigmas_LN[i]**2), f_PBH_GC_LN_t_init, color=colors[1], linestyle="None", marker="+")
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

                mc_KP23_SLN_t_init, f_PBH_KP23_SLN_t_init = np.genfromtxt("./Data-tests/t_initial/SLN_2302.04408_Carr_Delta={:.1f}_extrapolated_upper.txt".format(Deltas[i]), delimiter="\t")
                mp_KP23_CC3_t_init, f_PBH_KP23_CC3_t_init = np.genfromtxt("./Data-tests/t_initial/CC3_2302.04408_Carr_Delta={:.1f}_extrapolated_upper.txt".format(Deltas[i]), delimiter="\t")
                mc_KP23_LN_t_init, f_PBH_KP23_LN_t_init = np.genfromtxt("./Data-tests/t_initial/LN_2302.04408_Carr_Delta={:.1f}_extrapolated_upper.txt".format(Deltas[i]), delimiter="\t")
                
                mc_KP23_SLN_unevolved, f_PBH_KP23_SLN_unevolved = np.genfromtxt("./Data-tests/unevolved/SLN_2302.04408_Carr_Delta={:.1f}_extrapolated_upper.txt".format(Deltas[i]), delimiter="\t")
                mp_KP23_CC3_unevolved, f_PBH_KP23_CC3_unevolved = np.genfromtxt("./Data-tests/unevolved/CC3_2302.04408_Carr_Delta={:.1f}_extrapolated_upper.txt".format(Deltas[i]), delimiter="\t")
                mc_KP23_LN_unevolved, f_PBH_KP23_LN_unevolved = np.genfromtxt("./Data-tests/unevolved/LN_2302.04408_Carr_Delta={:.1f}_extrapolated_upper.txt".format(Deltas[i]), delimiter="\t")

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
    slopes_PL_lower = [0, 2, 3, 4]
    
    linestyles = ["solid", "dashed", "dashdot", "dotted"]
    
    for j in range(len(Deltas)):
        
        fig, axes = plt.subplots(2, 2, figsize=(13, 13))
        
        ax0 = axes[0][0]
        ax1 = axes[0][1]
        ax2 = axes[1][0]
        ax3 = axes[1][1]
        
        m_mono_values, f_max = load_data("2302.04408/2302.04408_MW_diffuse_SPI.csv")
                            
        # Power-law slope to use between 1e15g and 1e16g (motivated by mass-dependence of the positron spectrum emitted over energy)
        slope_PL_upper = 2.0
        
        m_mono_extrapolated_upper = np.logspace(15, 16, 11)
        m_mono_extrapolated_lower = np.logspace(11, 15, 41)
        f_max_extrapolated_upper = min(f_max) * np.power(m_mono_extrapolated_upper / min(m_mono_values), slope_PL_upper)
            
        for k, slope_PL_lower in enumerate(slopes_PL_lower):
            
            if k == 0:
                data_folder = "./Data-old/"
                data_filename_LN = data_folder + "/LN_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[j])
                data_filename_SLN = data_folder + "/SLN_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[j])
                data_filename_CC3 = data_folder + "/CC3_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[j])
                """
                mc_KP23_SLN_unevolved, f_PBH_KP23_SLN_unevolved = np.genfromtxt("./Data-tests/unevolved/SLN_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[j]), delimiter="\t")
                mp_KP23_CC3_unevolved, f_PBH_KP23_CC3_unevolved = np.genfromtxt("./Data-tests/unevolved/CC3_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[j]), delimiter="\t")
                mc_KP23_LN_unevolved, f_PBH_KP23_LN_unevolved = np.genfromtxt("./Data-tests/unevolved/LN_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[j]), delimiter="\t")
                """
            else:
                data_folder = "./Data-tests/PL_slope_{:.0f}".format(slope_PL_lower) 
                data_filename_LN = data_folder + "/LN_2302.04408_Carr_Delta={:.1f}_extrapolated_slope{:.0f}.txt".format(Deltas[j], slope_PL_lower)
                data_filename_SLN = data_folder + "/SLN_2302.04408_Carr_Delta={:.1f}_extrapolated_slope{:.0f}.txt".format(Deltas[j], slope_PL_lower)
                data_filename_CC3 = data_folder + "/CC3_2302.04408_Carr_Delta={:.1f}_extrapolated_slope{:.0f}.txt".format(Deltas[j], slope_PL_lower)
            
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
                ax1.plot(0, 0, marker="None", linestyle=linestyles[k], color="k", label="{:.0f}".format(slope_PL_lower))
                f_max_extrapolated_lower = min(f_max_extrapolated_upper) * np.power(m_mono_extrapolated_lower / min(m_mono_extrapolated_upper), slope_PL_lower)
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


#%% Tests of the results obtained using different power-law slopes in f_max at low masses (Galactic Centre photon constraints)

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    mc_values = np.logspace(14, 20, 120)

    # Array of power law exponents to use at masses below 1e15g
    slopes_PL_lower = [0, 2, 4]
    
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
        
        
        for k, slope_PL_lower in enumerate(slopes_PL_lower):
            data_folder = "./Data-tests/PL_slope_{:.0f}".format(slope_PL_lower)
            
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
                f_max_extrapolated = f_max_loaded_truncated[0] * np.power(m_delta_extrapolated / 1e13, slope_PL_lower)
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

            ax0.plot(0, 0, style_markers[k], color="k", label="{:.0f}".format(slope_PL_lower))            
            ax1.plot(0, 0, style_markers[k], color="k", label="{:.0f}".format(slope_PL_lower))
                
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
    
            ax0.legend(fontsize="x-small", title="PL slope in $f_\mathrm{max}$ \n ($m < 10^{13}~\mathrm{g}$)")        
            ax1.legend(fontsize="x-small", title="PL slope in $f_\mathrm{max}$ \n ($m < 10^{13}~\mathrm{g}$)")
            fig.tight_layout()
            fig.suptitle("$\Delta={:.1f}$".format(Deltas[j]))


#%% Tests of the results obtained using different power-law slopes in f_max at low masses (Prospective evaporation constraints from GECCO [2101.01370])

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    mc_values = np.logspace(14, 20, 120)

    # Array of power law exponents to use at masses below 1e15g
    slopes_PL_lower = [0, 2, 4]
    
    linestyles = ["dashed", "dashdot", "dotted"]
    
    for j in range(len(Deltas)):
        
        fig, axes = plt.subplots(2, 2, figsize=(13, 13))
        
        ax0 = axes[0][0]
        ax1 = axes[0][1]
        ax2 = axes[1][0]
        ax3 = axes[1][1]
        
        m_mono_values, f_max = load_data("2101.01370/2101.01370_Fig9_GC_Einasto.csv")
        m_mono_extrapolated = np.logspace(11, 15, 41)
                
        for k, slope_PL_lower in enumerate(slopes_PL_lower):
            
            data_folder = "./Data-tests/PL_slope_{:.0f}".format(slope_PL_lower) 
            data_filename_LN = data_folder + "/LN_2101.01370_Carr_Delta={:.1f}_Einasto_extrapolated_slope{:.0f}.txt".format(Deltas[j], slope_PL_lower)
            data_filename_SLN = data_folder + "/SLN_2101.01370_Carr_Delta={:.1f}_Einasto_extrapolated_slope{:.0f}.txt".format(Deltas[j], slope_PL_lower)
            data_filename_CC3 = data_folder + "/CC3_2101.01370_Carr_Delta={:.1f}_Einasto_extrapolated_slope{:.0f}.txt".format(Deltas[j], slope_PL_lower)
            
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
            
            ax1.plot(0, 0, marker="None", linestyle=linestyles[k], color="k", label="{:.0f}".format(slope_PL_lower))
            f_max_extrapolated = min(f_max) * np.power(m_mono_extrapolated / min(m_mono_values), slope_PL_lower)
            ax0.plot(m_mono_extrapolated, f_max_extrapolated, color=(0.5294, 0.3546, 0.7020), linestyle=linestyles[k], label="{:.0f}".format(slope_PL_lower))
            
        ax0.plot(np.concatenate((m_mono_extrapolated, m_mono_values)), np.concatenate((f_max_extrapolated, f_max)), color=(0.5294, 0.3546, 0.7020))
        ax0.set_xlabel("$m$ [g]")
        ax0.set_ylabel("$f_\mathrm{max}$")
        ax0.set_xscale("log")
        ax0.set_yscale("log")
        ax0.set_xlim(min(m_mono_extrapolated), 1e18)
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

        ax0.legend(fontsize="x-small", title="PL slope in $f_\mathrm{max}$ \n ($m < 10^{15}~\mathrm{g}$)")        
        ax1.legend(fontsize="x-small", title="PL slope in $f_\mathrm{max}$ \n ($m < 10^{15}~\mathrm{g}$)")
        fig.tight_layout()
        fig.suptitle("$\Delta={:.1f}$".format(Deltas[j]))

