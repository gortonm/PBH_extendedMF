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

#%% Existing constraints

if "__main__" == __name__:
    
    # If True, plot the evaporation constraints used by Isatis (from COMPTEL, INTEGRAL, EGRET and Fermi-LAT)
    plot_GC_Isatis = False
    # If True, plot the evaporation constraints shown in Korwar & Profumo (2023) [2302.04408]
    plot_KP23 = not plot_GC_Isatis
    # If True, use extended MF constraint calculated from the delta-function MF extrapolated down to 5e14g using a power-law fit
    include_extrapolated = True
    
    # Choose colors to match those from Fig. 5 of 2009.03204
    colors = ['silver', 'r', 'b', 'g', 'k']
    
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
                        
        fig, axes = plt.subplots(1, 3, figsize=(17, 6))
        ax0 = axes[0]
        ax1 = axes[1]
        ax2 = axes[2]
        
        # Loading constraints from Subaru-HSC.
        mc_Carr_SLN, f_PBH_Carr_SLN = np.genfromtxt("./Data/SLN_HSC_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
        mp_Subaru_CC3, f_PBH_Carr_CC3 = np.genfromtxt("./Data/CC3_HSC_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
        mc_Carr_LN, f_PBH_Carr_LN = np.genfromtxt("./Data/LN_HSC_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
        
        if plot_GC_Isatis:
            
            
            # Plot constraints obtained with unevolved MF
            mc_values_GC = np.logspace(14, 19, 100)
            # Load constraints from Galactic Centre photons.
            fname_base_CC3 = "CC_D={:.1f}_dm{:.0f}_".format(Deltas[i], -np.log10(delta_log_m)) + energies_string + "_c{:.0f}".format(-np.log10(cutoff))
            fname_base_SLN = "SL_D={:.1f}_dm{:.0f}_".format(Deltas[i], -np.log10(delta_log_m)) + energies_string + "_c{:.0f}".format(-np.log10(cutoff))
            
            constraints_names_GC, f_PBHs_GC_SLN = load_results_Isatis(mf_string=fname_base_SLN, modified=True)
            constraints_names_GC, f_PBHs_GC_CC3 = load_results_Isatis(mf_string=fname_base_CC3, modified=True)
            
            f_PBH_GC_SLN = envelope(f_PBHs_GC_SLN)
            f_PBH_GC_CC3 = envelope(f_PBHs_GC_CC3)
                        
            # Estimate peak mass of skew-lognormal MF
            mp_SLN_GC = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_values_GC]
            
            ax0.plot(mp_SLN_GC, f_PBH_GC_SLN, color=colors[2], alpha=0.4)
            ax0.plot(mc_values_GC, f_PBH_GC_CC3, color=colors[3], alpha=0.4)
            ax1.plot(mp_SLN_GC, f_PBH_GC_SLN, color=colors[2], alpha=0.4)
            ax1.plot(mc_values_GC, f_PBH_GC_CC3, color=colors[3], alpha=0.4)
            
            
            plt.suptitle("Existing constraints (showing Galactic Centre photon constraints (Isatis)), $\Delta={:.1f}$".format(Deltas[i]), fontsize="small")
            
            # Monochromatic MF constraints
            m_delta_evap = np.logspace(11, 21, 1000)

            constraints_names_evap, f_PBHs_GC_delta = load_results_Isatis(modified=True)
            f_PBH_delta_evap = envelope(f_PBHs_GC_delta)
            m_delta_Subaru, f_PBH_delta_Subaru = load_data("2007.12697/Subaru-HSC_2007.12697_dx=5.csv")

            mc_values_evap = np.logspace(14, 20, 120)
                        
            # Load constraints from Galactic Centre photons.
            
            exponent_PL_lower = 2
            constraints_names_short = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
            data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower)
            
            f_PBH_instrument_LN = []
            f_PBH_instrument_SLN = []
            f_PBH_instrument_CC3 = []

            for k in range(len(constraints_names_short)):
                # Load constraints for an evolved extended mass function obtained from each instrument
                data_filename_LN = data_folder + "/LN_GC_%s" % constraints_names_short[k] + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[i])
                data_filename_SLN = data_folder + "/SLN_GC_%s" % constraints_names_short[k]  + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[i])
                data_filename_CC3 = data_folder + "/CC3_GC_%s" % constraints_names_short[k]  + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[i])
                    
                mc_LN_evolved, f_PBH_LN_evolved = np.genfromtxt(data_filename_LN, delimiter="\t")
                mc_SLN_evolved, f_PBH_SLN_evolved = np.genfromtxt(data_filename_SLN, delimiter="\t")
                mp_CC3_evolved, f_PBH_CC3_evolved = np.genfromtxt(data_filename_CC3, delimiter="\t")
                
                # Compile constraints from all instruments
                f_PBH_instrument_LN.append(f_PBH_LN_evolved)
                f_PBH_instrument_SLN.append(f_PBH_SLN_evolved)
                f_PBH_instrument_CC3.append(f_PBH_CC3_evolved)
 
            f_PBH_GC_LN = envelope(f_PBH_instrument_LN)
            f_PBH_GC_SLN = envelope(f_PBH_instrument_SLN)
            f_PBH_GC_CC3 = envelope(f_PBH_instrument_CC3)
           
            # Estimate peak mass of skew-lognormal MF
            mp_SLN_evap = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_values_evap]
            mp_Subaru_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_Carr_SLN]
                        
            for ax in [ax0, ax1, ax2]:                
                ax.plot(m_delta_evap, f_PBH_delta_evap, color=colors[0], label="Delta function", linewidth=2)
                ax.plot(m_delta_Subaru, f_PBH_delta_Subaru, color=colors[0], linewidth=2)
                
                ax.set_xlabel("$m_p~[\mathrm{g}]$")                
                ax.plot(mp_SLN_evap, f_PBH_GC_SLN, color=colors[2], linestyle=(0, (5, 7)))
                ax.plot(mc_values_evap, f_PBH_GC_CC3, color=colors[3], linestyle="dashed")
                ax.plot(mc_values_evap * np.exp(-sigmas_LN[i]**2), f_PBH_GC_LN, color=colors[1], dashes=[6, 2])
                ax.set_ylabel("$f_\mathrm{PBH}$")
                ax.set_xscale("log")
                ax.set_yscale("log")
                
                # set x-axis and y-axis ticks
                # see https://stackoverflow.com/questions/30887920/how-to-show-minor-tick-labels-on-log-scale-with-matplotlib
                
                x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
                ax.xaxis.set_major_locator(x_major)
                x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 5)
                ax.xaxis.set_minor_locator(x_minor)
                ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

                y_major = mpl.ticker.LogLocator(base = 10.0, numticks = 10)
                ax.yaxis.set_major_locator(y_major)
                y_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
                ax.yaxis.set_minor_locator(y_minor)
                ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
                
                ax.plot(mp_Subaru_SLN, f_PBH_Carr_SLN, color=colors[2], label="SLN", linestyle=(0, (5, 7)))
                ax.plot(mp_Subaru_CC3, f_PBH_Carr_CC3, color=colors[3], label="CC3", linestyle="dashed")
                ax.plot(mc_Carr_LN * np.exp(-sigmas_LN[i]**2), f_PBH_Carr_LN, color=colors[1], label="LN", dashes=[6, 2])

        
        elif plot_KP23:
            
            exponent_PL_lower = 2
            data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower) 
            
            fig.suptitle("Existing constraints (showing 511 keV line constraints (Korwar \& Profumo 2023)), $\Delta={:.1f}$".format(Deltas[i]), fontsize="small")
            
            # Monochromatic MF constraints
            m_delta_evap, f_PBH_delta_evap = load_data("2302.04408/2302.04408_MW_diffuse_SPI.csv")
            m_delta_Subaru, f_PBH_delta_Subaru = load_data("2007.12697/Subaru-HSC_2007.12697_dx=5.csv")
            
            data_filename_LN = data_folder + "/LN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[i], exponent_PL_lower)
            data_filename_SLN = data_folder + "/SLN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[i], exponent_PL_lower)
            data_filename_CC3 = data_folder + "/CC3_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[i], exponent_PL_lower)
            
            mc_KP23_LN, f_PBH_KP23_LN = np.genfromtxt(data_filename_LN, delimiter="\t")
            mc_KP23_SLN, f_PBH_KP23_SLN = np.genfromtxt(data_filename_SLN, delimiter="\t")
            mp_KP23_CC3, f_PBH_KP23_CC3 = np.genfromtxt(data_filename_CC3, delimiter="\t")
            
            # Estimate peak mass of skew-lognormal MF
            mp_KP23_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_KP23_SLN]
            mp_Subaru_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_Carr_SLN]
            
            for ax in [ax0, ax1, ax2]:
                ax.set_xlabel("$m_p~[\mathrm{g}]$")
                ax.plot(m_delta_evap, f_PBH_delta_evap, color=colors[0], label="Delta function", linewidth=2)
                ax.plot(m_delta_Subaru, f_PBH_delta_Subaru, color=colors[0], linewidth=2)
                ax.set_ylabel("$f_\mathrm{PBH}$")
                ax.set_xscale("log")
                ax.set_yscale("log")
                
                ax.plot(mp_KP23_SLN, f_PBH_KP23_SLN, color=colors[2], linestyle=(0, (5, 7)))
                ax.plot(mp_KP23_CC3, f_PBH_KP23_CC3, color=colors[3], linestyle="dashed")
                ax.plot(mc_KP23_LN * np.exp(-sigmas_LN[i]**2), f_PBH_KP23_LN, color=colors[1], dashes=[6, 2])
                
                ax.plot(mp_Subaru_SLN, f_PBH_Carr_SLN, color=colors[2], label="SLN", linestyle=(0, (5, 7)))
                ax.plot(mp_Subaru_CC3, f_PBH_Carr_CC3, color=colors[3], label="CC3", linestyle="dashed")
                ax.plot(mc_Carr_LN * np.exp(-sigmas_LN[i]**2), f_PBH_Carr_LN, color=colors[1], label="LN", dashes=[6, 2])

            # Plot constraint obtained with unevolved MF
            
            mc_KP23_SLN, f_PBH_KP23_SLN = np.genfromtxt("./Data-old/SLN_2302.04408_Carr_Delta={:.1f}_extrapolated.txt".format(Deltas[i]), delimiter="\t")
            mp_KP23_CC3, f_PBH_KP23_CC3 = np.genfromtxt("./Data-old/CC3_2302.04408_Carr_Delta={:.1f}_extrapolated.txt".format(Deltas[i]), delimiter="\t")
            mc_KP23_LN, f_PBH_KP23_LN = np.genfromtxt("./Data-old/LN_2302.04408_Carr_Delta={:.1f}_extrapolated.txt".format(Deltas[i]), delimiter="\t")
           
            # Estimate peak mass of skew-lognormal MF
            mp_KP23_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_KP23_SLN]
            mp_Subaru_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_Carr_SLN]
        
            ax0.plot(mp_KP23_SLN, f_PBH_KP23_SLN, color=colors[2], alpha=0.4)
            ax0.plot(mp_KP23_CC3, f_PBH_KP23_CC3, color=colors[3], alpha=0.4)
            ax1.plot(mp_KP23_SLN, f_PBH_KP23_SLN, color=colors[2], alpha=0.4)
            ax1.plot(mp_KP23_CC3, f_PBH_KP23_CC3, color=colors[3], alpha=0.4)
            ax0.plot(mc_KP23_LN * np.exp(-sigmas_LN[i]**2), f_PBH_KP23_LN, color=colors[1], alpha=0.4)
            ax1.plot(mc_KP23_LN * np.exp(-sigmas_LN[i]**2), f_PBH_KP23_LN, color=colors[1], alpha=0.4)
            

        # Set axis limits
        if Deltas[i] < 5:
            xmin_evap, xmax_evap = 1e16, 2.5e17
            xmin_HSC, xmax_HSC = 1e21, 1e29
            ymin, ymax = 3e-5, 1
            
            if plot_KP23:
                xmin_evap, xmax_evap = 1e16, 7e17
                ymin, ymax = 3e-5, 1
        
        else:
            xmin_evap, xmax_evap = 1e16, 7e17
            xmin_HSC, xmax_HSC = 9e18, 1e29
            ymin, ymax = 3e-5, 1
            """
            
            xmin_GC, xmax_GC = 1e16, 7e17
            xmin_HSC, xmax_HSC = 9e18, 1e29
            ymin, ymax = 3e-6, 1
            
            """
            if plot_KP23:
                xmin_evap, xmax_evap = 1e16, 2e18
                ymin, ymax = 3e-5, 1

        ax0.set_xlim(xmin_evap, 1e24)
        ax0.set_ylim(ymin, ymax)
        ax1.set_xlim(xmin_evap, xmax_evap)
        ax1.set_ylim(ymin, ymax)
        ax2.set_xlim(xmin_HSC, xmax_HSC)
        ax2.set_ylim(4e-3, 1)
       
        ax0.legend(fontsize="xx-small")
        fig.tight_layout()
        
        if plot_GC_Isatis:
            fig.savefig("./Results/Figures/fPBH_Delta={:.1f}_existing_GC_Isatis.pdf".format(Deltas[i]))
            fig.savefig("./Results/Figures/fPBH_Delta={:.1f}_existing_GC_Isatis.png".format(Deltas[i]))
            
        elif plot_KP23:
            fig.savefig("./Results/Figures/fPBH_Delta={:.1f}_existing_KP23.pdf".format(Deltas[i]))
            fig.savefig("./Results/Figures/fPBH_Delta={:.1f}_existing_KP23.png".format(Deltas[i]))
                
#%% Prospective constraints

if "__main__" == __name__:
        
    # Choose colors to match those from Fig. 5 of 2009.03204
    colors = ['silver', 'r', 'b', 'g', 'k']
    
    # Parameters used for convergence tests in Galactic Centre constraints.
    cutoff = 1e-4
    delta_log_m = 1e-3
    E_number = 500    
                
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
        
    for i in range(len(Deltas)):
                        
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
        ax0 = axes[0]
        ax1 = axes[1]
        ax2 = axes[2]
        
        # Load prospective extended MF constraints from the white dwarf microlensing survey proposed in Sugiyama et al. (2020) [1905.06066].
        mc_micro_SLN, f_PBH_micro_SLN = np.genfromtxt("./Data/SLN_Sugiyama20_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
        mp_micro_CC3, f_PBH_micro_CC3 = np.genfromtxt("./Data/CC3_Sugiyama20_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
        mc_micro_LN, f_PBH_micro_LN = np.genfromtxt("./Data/LN_Sugiyama20_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
        
        # Load prospective extended MF constraints from GECCO. 
        exponent_PL_lower = 2
        data_folder = "./Data-tests/PL_exp_{:.0f}/".format(exponent_PL_lower) 
        
        data_filename_LN_NFW = data_folder + "LN_2101.01370_Carr_Delta={:.1f}_NFW_extrapolated_exp{:.0f}.txt".format(Deltas[i], exponent_PL_lower) 
        data_filename_SLN_NFW = data_folder + "SLN_2101.01370_Carr_Delta={:.1f}_NFW_extrapolated_exp{:.0f}.txt".format(Deltas[i], exponent_PL_lower) 
        data_filename_CC3_NFW = data_folder + "CC3_2101.01370_Carr_Delta={:.1f}_NFW_extrapolated_exp{:.0f}.txt".format(Deltas[i], exponent_PL_lower) 
        mc_GECCO_LN_NFW, f_PBH_GECCO_LN_NFW = np.genfromtxt(data_filename_LN_NFW, delimiter="\t")
        mc_GECCO_SLN_NFW, f_PBH_GECCO_SLN_NFW = np.genfromtxt(data_filename_SLN_NFW, delimiter="\t")
        mp_GECCO_CC3_NFW, f_PBH_GECCO_CC3_NFW = np.genfromtxt(data_filename_CC3_NFW, delimiter="\t")
           
        data_filename_LN_Einasto = data_folder + "LN_2101.01370_Carr_Delta={:.1f}_Einasto_extrapolated_exp{:.0f}.txt".format(Deltas[i], exponent_PL_lower) 
        data_filename_SLN_Einasto = data_folder + "SLN_2101.01370_Carr_Delta={:.1f}_Einasto_extrapolated_exp{:.0f}.txt".format(Deltas[i], exponent_PL_lower) 
        data_filename_CC3_Einasto = data_folder + "CC3_2101.01370_Carr_Delta={:.1f}_Einasto_extrapolated_exp{:.0f}.txt".format(Deltas[i], exponent_PL_lower) 
        mc_GECCO_LN_Einasto, f_PBH_GECCO_LN_Einasto = np.genfromtxt(data_filename_LN_Einasto, delimiter="\t")
        mc_GECCO_SLN_Einasto, f_PBH_GECCO_SLN_Einasto = np.genfromtxt(data_filename_SLN_Einasto, delimiter="\t")
        mp_GECCO_CC3_Einasto, f_PBH_GECCO_CC3_Einasto = np.genfromtxt(data_filename_CC3_Einasto, delimiter="\t")
        
        
        # Estimate peak mass of skew-lognormal MF
        mp_GECCO_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_GECCO_SLN_NFW]
        mp_micro_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_micro_SLN]
        

        # Load delta-function MF constraints
        m_delta_evap, f_PBH_delta_evap = load_data("1905.06066/1905.06066_Fig8_finite+wave.csv")
        m_delta_micro_NFW, f_PBH_delta_micro_NFW = load_data("2101.01370/2101.01370_Fig9_GC_NFW.csv")
        m_delta_micro_Einasto, f_PBH_delta_micro_Einasto = load_data("2101.01370/2101.01370_Fig9_GC_Einasto.csv")
        
        
        # Plot constraints
        
        for ax in [ax0, ax1, ax2]:
            ax.set_xlabel("$m_p~[\mathrm{g}]$")
            ax.plot(m_delta_evap, f_PBH_delta_evap, color=colors[0], label="Delta function", linewidth=2)
            #ax.plot(m_delta_micro_NFW, f_PBH_delta_micro_NFW, color=colors[0], linewidth=2)
            ax.plot(m_delta_micro_Einasto, f_PBH_delta_micro_Einasto, color=colors[0], linewidth=2)
            ax.set_ylabel("$f_\mathrm{PBH}$")
            ax.set_xscale("log")
            ax.set_yscale("log")

            # plot Einasto profile results
            ax.plot(mp_GECCO_SLN, f_PBH_GECCO_SLN_Einasto, color=colors[2], linestyle=(0, (5, 7)))
            ax.plot(mp_GECCO_CC3_Einasto, f_PBH_GECCO_CC3_Einasto, color=colors[3], linestyle="dashed")
            ax.plot(mc_GECCO_LN_Einasto * np.exp(-sigmas_LN[i]**2), f_PBH_GECCO_LN_Einasto, color=colors[1], dashes=[6, 2])
            
            """
            #Uncomment to plot NFW results
            ax.plot(mp_GECCO_SLN, f_PBH_GECCO_SLN_NFW, color=colors[2], linestyle=(0, (5, 7)))
            ax.plot(mp_GECCO_CC3_NFW, f_PBH_GECCO_CC3_NFW, color=colors[3], linestyle="dashed")
            ax.plot(mc_GECCO_LN_NFW * np.exp(-sigmas_LN[i]**2), f_PBH_GECCO_LN_NFW, color=colors[1], dashes=[6, 2], label="LN")
            """
            ax.plot(mp_micro_SLN, f_PBH_micro_SLN, color=colors[2], label="SLN", linestyle=(0, (5, 7)))
            ax.plot(mp_micro_CC3, f_PBH_micro_CC3, color=colors[3], label="CC3", linestyle="dashed")
            ax.plot(mc_micro_LN * np.exp(-sigmas_LN[i]**2), f_PBH_micro_LN, color=colors[1], dashes=[6, 2], label="LN")
            
            # set x-axis and y-axis ticks
            # see https://stackoverflow.com/questions/30887920/how-to-show-minor-tick-labels-on-log-scale-with-matplotlib
            
            x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
            ax.xaxis.set_major_locator(x_major)
            x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 5)
            ax.xaxis.set_minor_locator(x_minor)
            ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

            y_major = mpl.ticker.LogLocator(base = 10.0, numticks = 10)
            ax.yaxis.set_major_locator(y_major)
            y_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
            ax.yaxis.set_minor_locator(y_minor)
            ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

        # Set axis limits
        if Deltas[i] < 5:
            xmin_evap, xmax_evap = 1e16, 2e18
            xmin_micro, xmax_micro = 2e20, 5e23
            ymin, ymax = 1e-5, 1
        else:
            xmin_evap, xmax_evap = 1e16, 5e18
            xmin_micro, xmax_micro = 2e17, 5e23
            ymin, ymax = 1e-5, 1

        ax0.set_xlim(xmin_evap, 1e22)
        ax0.set_ylim(ymin, ymax)
        ax1.set_xlim(xmin_evap, xmax_evap)
        ax1.set_ylim(ymin, ymax)
        ax2.set_xlim(xmin_micro, xmax_micro)
        ax2.set_ylim(1e-3, 1)
       
        ax0.legend(fontsize="xx-small")
        
        plt.suptitle("Prospective constraints, $\Delta={:.1f}$".format(Deltas[i]), fontsize="small")
        fig.tight_layout()
        fig.savefig("./Results/Figures/fPBH_Delta={:.1f}_prospective.pdf".format(Deltas[i]))
        fig.savefig("./Results/Figures/fPBH_Delta={:.1f}_prospective.png".format(Deltas[i]))
            

#%% Existing constraints: try plotting 511 keV line and Galactic Centre photon constraints on the same plot

if "__main__" == __name__:
        
    # Choose colors to match those from Fig. 5 of 2009.03204
    colors = ['tab:gray', 'r', 'b', 'g', 'k']
    
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
                        
        fig, axes = plt.subplots(1, 3, figsize=(17, 7))
        ax0 = axes[0]
        ax1 = axes[1]
        ax2 = axes[2]
        
        # Loading constraints from Subaru-HSC.
        mc_Carr_SLN, f_PBH_Carr_SLN = np.genfromtxt("./Data/SLN_HSC_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
        mp_Subaru_CC3, f_PBH_Carr_CC3 = np.genfromtxt("./Data/CC3_HSC_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
        mc_Carr_LN, f_PBH_Carr_LN = np.genfromtxt("./Data/LN_HSC_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
        
        plt.suptitle("$\Delta={:.1f}$".format(Deltas[i]), fontsize="small")
        
        # Monochromatic MF constraints
        m_delta_evap = np.logspace(11, 21, 1000)

        constraints_names_evap, f_PBHs_GC_delta = load_results_Isatis(modified=True)
        f_PBH_delta_evap = envelope(f_PBHs_GC_delta)
        m_delta_Subaru, f_PBH_delta_Subaru = load_data("2007.12697/Subaru-HSC_2007.12697_dx=5.csv")

        mc_values_evap = np.logspace(14, 20, 120)
                
        # Load constraints from Galactic Centre photons.
        exponent_PL_lower = 2
        constraints_names_short = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
        data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower)
        
        f_PBH_instrument_LN = []
        f_PBH_instrument_SLN = []
        f_PBH_instrument_CC3 = []

        for k in range(len(constraints_names_short)):
            # Load constraints for an evolved extended mass function obtained from each instrument
            data_filename_LN = data_folder + "/LN_GC_%s" % constraints_names_short[k] + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[i])
            data_filename_SLN = data_folder + "/SLN_GC_%s" % constraints_names_short[k]  + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[i])
            data_filename_CC3 = data_folder + "/CC3_GC_%s" % constraints_names_short[k]  + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[i])
                
            mc_LN_evolved, f_PBH_LN_evolved = np.genfromtxt(data_filename_LN, delimiter="\t")
            mc_SLN_evolved, f_PBH_SLN_evolved = np.genfromtxt(data_filename_SLN, delimiter="\t")
            mp_CC3_evolved, f_PBH_CC3_evolved = np.genfromtxt(data_filename_CC3, delimiter="\t")
            
            # Compile constraints from all instruments
            f_PBH_instrument_LN.append(f_PBH_LN_evolved)
            f_PBH_instrument_SLN.append(f_PBH_SLN_evolved)
            f_PBH_instrument_CC3.append(f_PBH_CC3_evolved)
 
        f_PBH_GC_LN = envelope(f_PBH_instrument_LN)
        f_PBH_GC_SLN = envelope(f_PBH_instrument_SLN)
        f_PBH_GC_CC3 = envelope(f_PBH_instrument_CC3)
       
        # Estimate peak mass of skew-lognormal MF
        mp_SLN_evap = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_values_evap]
        mp_Subaru_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_Carr_SLN]
                        
        exponent_PL_lower = 2
        data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower) 
               
        # Monochromatic MF constraints
        m_delta_KP23, f_PBH_delta_KP23 = load_data("2302.04408/2302.04408_MW_diffuse_SPI.csv")
        m_delta_Subaru, f_PBH_delta_Subaru = load_data("2007.12697/Subaru-HSC_2007.12697_dx=5.csv")
        
        data_filename_LN = data_folder + "/LN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[i], exponent_PL_lower)
        data_filename_SLN = data_folder + "/SLN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[i], exponent_PL_lower)
        data_filename_CC3 = data_folder + "/CC3_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[i], exponent_PL_lower)
        
        mc_KP23_LN, f_PBH_KP23_LN = np.genfromtxt(data_filename_LN, delimiter="\t")
        mc_KP23_SLN, f_PBH_KP23_SLN = np.genfromtxt(data_filename_SLN, delimiter="\t")
        mp_KP23_CC3, f_PBH_KP23_CC3 = np.genfromtxt(data_filename_CC3, delimiter="\t")
           
        # Estimate peak mass of skew-lognormal MF
        mp_KP23_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_KP23_SLN]
        mp_Subaru_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_Carr_SLN]
        
        for ax in [ax0, ax1, ax2]:
            
            ax.set_xlabel("$m_p~[\mathrm{g}]$")
            ax.plot(m_delta_evap, f_PBH_delta_evap, color=colors[0], linestyle="dashed", label="Delta function", linewidth=2)
            ax.plot(m_delta_KP23, f_PBH_delta_KP23, color=colors[0], linewidth=2, alpha=0.4)
            ax.plot(m_delta_Subaru, f_PBH_delta_Subaru, color=colors[0], linewidth=2)
            ax.set_ylabel("$f_\mathrm{PBH}$")
            ax.set_xscale("log")
            ax.set_yscale("log")
            
            ax.plot(mp_SLN_evap, f_PBH_GC_SLN, color=colors[2], linestyle=(0, (5, 7)))
            ax.plot(mc_values_evap, f_PBH_GC_CC3, color=colors[3], linestyle="dashed")
            ax.plot(mc_values_evap * np.exp(-sigmas_LN[i]**2), f_PBH_GC_LN, color=colors[1], dashes=[6, 2])
            
            """
            ax.plot(mp_KP23_SLN, f_PBH_KP23_SLN, color=colors[2], linestyle=(0, (5, 7)), alpha=0.5)
            ax.plot(mp_KP23_CC3, f_PBH_KP23_CC3, color=colors[3], linestyle="dashed", alpha=0.5)
            ax.plot(mc_KP23_LN * np.exp(-sigmas_LN[i]**2), f_PBH_KP23_LN, color=colors[1], dashes=[6, 2], alpha=0.5)
            """
            
            ax.plot(mp_KP23_SLN, f_PBH_KP23_SLN, color=colors[2], alpha=0.4)
            ax.plot(mp_KP23_CC3, f_PBH_KP23_CC3, color=colors[3], alpha=0.4)
            ax.plot(mc_KP23_LN * np.exp(-sigmas_LN[i]**2), f_PBH_KP23_LN, color=colors[1], alpha=0.4)
            
            
            ax.plot(mp_Subaru_SLN, f_PBH_Carr_SLN, color=colors[2], label="SLN", linestyle=(0, (5, 7)))
            ax.plot(mp_Subaru_CC3, f_PBH_Carr_CC3, color=colors[3], label="CC3", linestyle="dashed")
            ax.plot(mc_Carr_LN * np.exp(-sigmas_LN[i]**2), f_PBH_Carr_LN, color=colors[1], label="LN", dashes=[6, 2])
        
        # Set axis limits
        if Deltas[i] < 5:
            xmin_HSC, xmax_HSC = 1e21, 1e29            
            xmin_evap, xmax_evap = 1e16, 7e17
            ymin, ymax = 1e-4, 1
        
        else:
            xmin_HSC, xmax_HSC = 9e18, 1e29
            xmin_evap, xmax_evap = 1e16, 2e18
            ymin, ymax = 1e-4, 1
                      
        ax0.set_xlim(xmin_evap, 1e25)
        ax0.set_ylim(ymin, ymax)
        ax1.set_xlim(xmin_evap, xmax_evap)
        ax1.set_ylim(ymin, ymax)
        ax2.set_xlim(xmin_HSC, xmax_HSC)
        ax2.set_ylim(4e-3, 1)
       
        ax0.legend(fontsize="xx-small")
        fig.tight_layout()
        

#%% Plot constraints for different Delta on the same plot

if "__main__" == __name__:
            
    exponent_PL_lower = 2
    data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower) 
        
    # Monochromatic MF constraints
    m_delta_evap, f_PBH_delta_evap = load_data("2302.04408/2302.04408_MW_diffuse_SPI.csv")
    
    colors=["tab:blue", "tab:orange", "tab:green"]
    
    fig, ax = plt.subplots(figsize=(6,6))
    
    for i, Delta_index in enumerate([0, 5, 6]):
        
        data_filename_LN = data_folder + "/LN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)
        data_filename_SLN = data_folder + "/SLN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)
        data_filename_CC3 = data_folder + "/CC3_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)
    
        mc_KP23_LN, f_PBH_KP23_LN = np.genfromtxt(data_filename_LN, delimiter="\t")
        mc_KP23_SLN, f_PBH_KP23_SLN = np.genfromtxt(data_filename_SLN, delimiter="\t")
        mp_KP23_CC3, f_PBH_KP23_CC3 = np.genfromtxt(data_filename_CC3, delimiter="\t")
        
        # Estimate peak mass of skew-lognormal MF
        mp_KP23_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_KP23_SLN]

        ax.plot(mp_KP23_SLN, f_PBH_KP23_SLN, color=colors[i], linestyle=(0, (5, 7)), label="{:.0f}".format(Deltas[Delta_index]))
        #ax.plot(mp_KP23_CC3, f_PBH_KP23_CC3, color=colors[i], linestyle="dashed")
        #ax.plot(mc_KP23_LN * np.exp(-sigmas_LN[i]**2), f_PBH_KP23_LN, color=colors[1], dashes=[6, 2])

        # Plot constraint obtained with unevolved MF
        mc_KP23_SLN, f_PBH_KP23_SLN = np.genfromtxt("./Data-old/SLN_2302.04408_Carr_Delta={:.1f}_extrapolated.txt".format(Deltas[Delta_index]), delimiter="\t")
        mp_KP23_CC3, f_PBH_KP23_CC3 = np.genfromtxt("./Data-old/CC3_2302.04408_Carr_Delta={:.1f}_extrapolated.txt".format(Deltas[Delta_index]), delimiter="\t")
        mc_KP23_LN, f_PBH_KP23_LN = np.genfromtxt("./Data-old/LN_2302.04408_Carr_Delta={:.1f}_extrapolated.txt".format(Deltas[Delta_index]), delimiter="\t")
                          
        # Estimate peak mass of skew-lognormal MF
        mp_KP23_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_KP23_SLN]
    
        ax.plot(mp_KP23_SLN, f_PBH_KP23_SLN, color=colors[i], alpha=0.4)
        #ax.plot(mp_KP23_CC3, f_PBH_KP23_CC3, color=colors[i], alpha=0.4)
        # ax.plot(mc_KP23_LN * np.exp(-sigmas_LN[i]**2), f_PBH_KP23_LN, color=colors[1], alpha=0.4)
        

    ax.set_xlabel("$m_p~[\mathrm{g}]$")
    ax.plot(m_delta_evap, f_PBH_delta_evap, color="tab:gray", label="Delta func.", linewidth=2)
    ax.legend(title="$\Delta$", fontsize="x-small")
    ax.set_ylabel("$f_\mathrm{PBH}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e16, 5e18)
    ax.set_ylim(1e-5, 1)
    fig.suptitle("511 keV line constraints (Korwar \& Profumo 2023))\n SLN", fontsize="x-small")
    fig.tight_layout()
        
