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
        
    # Monochromatic MF constraints
    m_mono_GC = np.logspace(11, 21, 1000)

    constraints_names_GC, f_PBHs_GC_mono = load_results_Isatis(modified=True)
    f_PBH_mono_GC = envelope(f_PBHs_GC_mono)
    m_mono_Subaru, f_PBH_mono_Subaru = load_data("2007.12697/Subaru-HSC_2007.12697_dx=5.csv")
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)

    mc_values_GC = np.logspace(14, 19, 100)
        
    for i in range(len(Deltas)):
                
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        ax0 = axes[0]
        ax1 = axes[1]
        ax2 = axes[2]
        
        # Load constraints from Galactic Centre photons.
        fname_base_CC3 = "CC_D={:.1f}_dm{:.0f}_".format(Deltas[i], -np.log10(delta_log_m)) + energies_string + "_c{:.0f}".format(-np.log10(cutoff))
        fname_base_SLN = "SL_D={:.1f}_dm{:.0f}_".format(Deltas[i], -np.log10(delta_log_m)) + energies_string + "_c{:.0f}".format(-np.log10(cutoff))

        constraints_names_GC, f_PBHs_GC_SLN = load_results_Isatis(mf_string=fname_base_SLN, modified=True)
        constraints_names_GC, f_PBHs_GC_CC3 = load_results_Isatis(mf_string=fname_base_CC3, modified=True)

        f_PBH_GC_SLN = envelope(f_PBHs_GC_SLN)
        f_PBH_GC_CC3 = envelope(f_PBHs_GC_CC3)
        
        # Loading constraints from Subaru-HSC.
        mc_Carr_SLN, f_PBH_Carr_SLN = np.genfromtxt("./Data/SLN_HSC_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
        mp_Carr_CC3, f_PBH_Carr_CC3 = np.genfromtxt("./Data/CC3_HSC_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
        
        # Estimate peak mass of skew-lognormal MF
        mp_SLN_GC = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_values_GC]
        mp_Carr_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_Carr_SLN]
        
        ax0.plot(mp_SLN_GC, f_PBH_GC_SLN, color=colors[2], linestyle=(0, (5, 7)))
        ax0.plot(mc_values_GC, f_PBH_GC_CC3, color=colors[3], linestyle="dashed")
        ax1.plot(mp_SLN_GC, f_PBH_GC_SLN, color=colors[2], linestyle=(0, (5, 7)))
        ax1.plot(mc_values_GC, f_PBH_GC_CC3, color=colors[3], linestyle="dashed")


        # When defined, load and plot constraints for log-normal mass function
        if Deltas[i] < 5:
            
            fname_base_LN = "LN_D={:.1f}_dm{:.0f}_".format(Deltas[i], -np.log10(delta_log_m)) + energies_string + "_c{:.0f}".format(-np.log10(cutoff))
            constraints_names, f_PBHs_GC_LN = load_results_Isatis(mf_string=fname_base_LN, modified=True)
            f_PBH_GC_LN = envelope(f_PBHs_GC_LN)
                                    
            mc_Carr_LN, f_PBH_Carr_LN = np.genfromtxt("./Data/LN_HSC_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
            ax0.plot(mc_values_GC * np.exp(-sigmas_LN[i]**2), f_PBH_GC_LN, color=colors[1], dashes=[6, 2])
            ax1.plot(mc_values_GC * np.exp(-sigmas_LN[i]**2), f_PBH_GC_LN, color=colors[1], dashes=[6, 2])

            #ax1.plot(mc_Carr_LN, f_PBH_Carr_LN, color=colors[1], label="LN")
            ax0.plot(mc_Carr_LN * np.exp(-sigmas_LN[i]**2), f_PBH_Carr_LN, color=colors[1], dashes=[6, 2], label="LN")
            ax2.plot(mc_Carr_LN * np.exp(-sigmas_LN[i]**2), f_PBH_Carr_LN, color=colors[1], dashes=[6, 2], label="LN")
            
            xmin_GC, xmax_GC = 1e16, 2.5e17
            xmin_HSC, xmax_HSC = 1e21, 1e29
            ymin, ymax = 1e-4, 1
        
        else:
            xmin_GC, xmax_GC = 1e16, 7e17
            xmin_HSC, xmax_HSC = 9e18, 1e29
            ymin, ymax = 3e-6, 1

        #ax1.plot(mc_Carr_SLN, f_PBH_Carr_SLN, color=colors[2], label="SLN", linestyle="dashed")
        ax0.plot(mp_Carr_SLN, f_PBH_Carr_SLN, color=colors[2], label="SLN", linestyle=(0, (5, 7)))
        ax0.plot(mp_Carr_CC3, f_PBH_Carr_CC3, color=colors[3], label="CC3", linestyle="dashed")
        ax2.plot(mp_Carr_SLN, f_PBH_Carr_SLN, color=colors[2], label="SLN", linestyle=(0, (5, 7)))
        ax2.plot(mp_Carr_CC3, f_PBH_Carr_CC3, color=colors[3], label="CC3", linestyle="dashed")
        
        
        if i in [0, 1, 4, 5, 6]:
            # Loading constraints from numeric mass function (Subaru-HSC).
            mp_Carr_numeric_HSC, f_PBH_Carr_numeric_HSC = np.genfromtxt("./Data/numeric_HSC_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
            ax0.plot(mp_Carr_numeric_HSC, f_PBH_Carr_numeric_HSC, color=colors[4], label="numeric")
            ax2.plot(mp_Carr_numeric_HSC, f_PBH_Carr_numeric_HSC, color=colors[4])
            
            # Loading constraints from numeric mass function (Galactic Centre photons).
            mp_Carr_numeric_GC, f_PBH_Carr_numeric_GC = np.genfromtxt("./Data/numeric_GC_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
            ax0.plot(mp_Carr_numeric_GC, f_PBH_Carr_numeric_GC, color=colors[4])
            ax1.plot(mp_Carr_numeric_GC, f_PBH_Carr_numeric_GC, color=colors[4])
          
        
        ax0.set_xlim(xmin_GC, xmax_HSC)
        ax0.set_ylim(ymin, ymax)
        ax1.set_xlim(xmin_GC, xmax_GC)
        ax1.set_ylim(ymin, ymax)
        ax2.set_xlim(xmin_HSC, xmax_HSC)
        ax2.set_ylim(4e-3, 1)
       
        for ax in [ax0, ax1, ax2]:
            ax.set_xlabel("$m_p~[\mathrm{g}]$")
            ax.plot(m_mono_GC, f_PBH_mono_GC, color=colors[0], label="Delta function", linestyle="dotted", linewidth=2)
            ax.plot(m_mono_Subaru, f_PBH_mono_Subaru, color=colors[0], linestyle="dotted", linewidth=2)
            ax.set_ylabel("$f_\mathrm{PBH}$")
            ax.set_xscale("log")
            ax.set_yscale("log")
            
        ax0.legend(fontsize="xx-small")
        fig.tight_layout()
        fig.suptitle("$\Delta={:.1f}$".format(Deltas[i]))
        fig.savefig("./Results/Figures/fPBH_Delta={:.1f}.pdf".format(Deltas[i]))
        fig.savefig("./Results/Figures/fPBH_Delta={:.1f}.png".format(Deltas[i]))