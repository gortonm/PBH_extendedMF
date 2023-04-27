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
from extended_MF_checks import envelope, load_results_Isatis
from preliminaries import load_data, m_max_SLN

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

#%%

if "__main__" == __name__:
    
    # If True, plot the evaporation constraints used by Isatis (from COMPTEL, INTEGRAL, EGRET and Fermi-LAT)
    plot_GC_Isatis = False
    # If True, plot the evaporation constraints shown in Korwar & Profumo (2023) [2302.04408]
    plot_KP23 = True
    # If True, use extended MF constraint calculated from the delta-function MF extrapolated down to 5e14g using a power-law fit
    include_extrapolated = False
    # If True, plot results obtained using the numerical MF from Fig. 5 of 2009.03204
    plot_numeric = False
    
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
                
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        ax0 = axes[0]
        ax1 = axes[1]
        ax2 = axes[2]
        
        # Loading constraints from Subaru-HSC.
        mc_Carr_SLN, f_PBH_Carr_SLN = np.genfromtxt("./Data/SLN_HSC_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
        mp_Subaru_CC3, f_PBH_Carr_CC3 = np.genfromtxt("./Data/CC3_HSC_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
        mc_Carr_LN, f_PBH_Carr_LN = np.genfromtxt("./Data/LN_HSC_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
        
        if plot_GC_Isatis:
            
            plt.suptitle("Galactic Centre photon constraints (Isatis), $\Delta={:.1f}$".format(Deltas[i]), fontsize="small")
            
            # Monochromatic MF constraints
            m_mono_evap = np.logspace(11, 21, 1000)

            constraints_names_evap, f_PBHs_GC_mono = load_results_Isatis(modified=True)
            f_PBH_mono_evap = envelope(f_PBHs_GC_mono)
            m_mono_Subaru, f_PBH_mono_Subaru = load_data("2007.12697/Subaru-HSC_2007.12697_dx=5.csv")

            mc_values_evap = np.logspace(14, 19, 100)
            
            # Load constraints from Galactic Centre photons.
            fname_base_CC3 = "CC_D={:.1f}_dm{:.0f}_".format(Deltas[i], -np.log10(delta_log_m)) + energies_string + "_c{:.0f}".format(-np.log10(cutoff))
            fname_base_SLN = "SL_D={:.1f}_dm{:.0f}_".format(Deltas[i], -np.log10(delta_log_m)) + energies_string + "_c{:.0f}".format(-np.log10(cutoff))
            fname_base_LN = "LN_D={:.1f}_dm{:.0f}_".format(Deltas[i], -np.log10(delta_log_m)) + energies_string + "_c{:.0f}".format(-np.log10(cutoff))

            constraints_names_evap, f_PBHs_GC_SLN = load_results_Isatis(mf_string=fname_base_SLN, modified=True)
            constraints_names_evap, f_PBHs_GC_CC3 = load_results_Isatis(mf_string=fname_base_CC3, modified=True)
            constraints_names, f_PBHs_GC_LN = load_results_Isatis(mf_string=fname_base_LN, modified=True)
   
            f_PBH_GC_SLN = envelope(f_PBHs_GC_SLN)
            f_PBH_GC_CC3 = envelope(f_PBHs_GC_CC3)
            f_PBH_GC_LN = envelope(f_PBHs_GC_LN)
           
            # Estimate peak mass of skew-lognormal MF
            mp_SLN_evap = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_values_evap]
            mp_Subaru_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_Carr_SLN]
            
            ax0.plot(mp_SLN_evap, f_PBH_GC_SLN, color=colors[2], linestyle=(0, (5, 7)))
            ax0.plot(mc_values_evap, f_PBH_GC_CC3, color=colors[3], linestyle="dashed")
            ax1.plot(mp_SLN_evap, f_PBH_GC_SLN, color=colors[2], linestyle=(0, (5, 7)))
            ax1.plot(mc_values_evap, f_PBH_GC_CC3, color=colors[3], linestyle="dashed")
                                                                    
            ax0.plot(mc_values_evap * np.exp(-sigmas_LN[i]**2), f_PBH_GC_LN, color=colors[1], dashes=[6, 2])
            ax1.plot(mc_values_evap * np.exp(-sigmas_LN[i]**2), f_PBH_GC_LN, color=colors[1], dashes=[6, 2])
        
        elif plot_KP23:
            
            if include_extrapolated:
                fig.suptitle("Using 511 keV line constraints (Korwar \& Profumo 2023), $\Delta={:.1f}$ \n $f_".format(Deltas[i]) + "\mathrm{max}(m)$" + " extrapolated below " + "$m=10^{16}" + "~\mathrm{g}$", fontsize="small")
            
            # Monochromatic MF constraints
            m_mono_evap, f_PBH_mono_evap = load_data("2302.04408/2302.04408_MW_diffuse_SPI.csv")
            m_mono_Subaru, f_PBH_mono_Subaru = load_data("2007.12697/Subaru-HSC_2007.12697_dx=5.csv")

            # Load constraints from Galactic Centre 511 keV line emission (from 2302.04408).            
            if include_extrapolated:
                mc_KP23_SLN, f_PBH_KP23_SLN = np.genfromtxt("./Data/SLN_2302.04408_Carr_Delta={:.1f}_extrapolated.txt".format(Deltas[i]), delimiter="\t")
                mp_KP23_CC3, f_PBH_KP23_CC3 = np.genfromtxt("./Data/CC3_2302.04408_Carr_Delta={:.1f}_extrapolated.txt".format(Deltas[i]), delimiter="\t")
                mc_KP23_LN, f_PBH_KP23_LN = np.genfromtxt("./Data/LN_2302.04408_Carr_Delta={:.1f}_extrapolated.txt".format(Deltas[i]), delimiter="\t")
                mp_KP23_numeric, f_PBH_KP23_numeric = np.genfromtxt("./Data/numeric_2302.04408_Carr_Delta={:.1f}_extrapolated.txt".format(Deltas[i]), delimiter="\t")
               
            else:
                mc_KP23_SLN, f_PBH_KP23_SLN = np.genfromtxt("./Data/SLN_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
                mp_KP23_CC3, f_PBH_KP23_CC3 = np.genfromtxt("./Data/CC3_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
                mc_KP23_LN, f_PBH_KP23_LN = np.genfromtxt("./Data/LN_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
                mp_KP23_numeric, f_PBH_KP23_numeric = np.genfromtxt("./Data/numeric_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")

            # Estimate peak mass of skew-lognormal MF
            mp_KP23_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_KP23_SLN]
            mp_Subaru_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_Carr_SLN]
            
            ax0.plot(mp_KP23_SLN, f_PBH_KP23_SLN, color=colors[2], linestyle=(0, (5, 7)))
            ax0.plot(mp_KP23_CC3, f_PBH_KP23_CC3, color=colors[3], linestyle="dashed")
            ax1.plot(mp_KP23_SLN, f_PBH_KP23_SLN, color=colors[2], linestyle=(0, (5, 7)))
            ax1.plot(mp_KP23_CC3, f_PBH_KP23_CC3, color=colors[3], linestyle="dashed")
            ax0.plot(mc_KP23_LN * np.exp(-sigmas_LN[i]**2), f_PBH_KP23_LN, color=colors[1], dashes=[6, 2], label="LN")
            ax1.plot(mc_KP23_LN * np.exp(-sigmas_LN[i]**2), f_PBH_KP23_LN, color=colors[1], dashes=[6, 2])
            
            mc_Carr_LN, f_PBH_Carr_LN = np.genfromtxt("./Data/LN_HSC_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")

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
        
        # Loading constraints from numeric mass function (Subaru-HSC).
        mp_Carr_numeric_HSC, f_PBH_Carr_numeric_HSC = np.genfromtxt("./Data/numeric_HSC_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
        ax0.plot(mp_Carr_numeric_HSC, f_PBH_Carr_numeric_HSC, color=colors[4], label="numeric")
        ax2.plot(mp_Carr_numeric_HSC, f_PBH_Carr_numeric_HSC, color=colors[4])

        
        if plot_numeric:
            if i in [0, 1, 4, 5, 6]:                
                # Loading constraints from numeric mass function (Galactic Centre photons).
                mp_Carr_numeric_evap, f_PBH_Carr_numeric_evap = np.genfromtxt("./Data/numeric_GC_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
                ax0.plot(mp_Carr_numeric_evap, f_PBH_Carr_numeric_evap, color=colors[4])
                ax1.plot(mp_Carr_numeric_evap, f_PBH_Carr_numeric_evap, color=colors[4])
          
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