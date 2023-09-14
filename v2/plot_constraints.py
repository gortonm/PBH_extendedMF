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
        
    # Load Subaru-HSC delta-function MF constraint
    m_delta_Subaru, f_PBH_delta_Subaru = load_data("2007.12697/Subaru-HSC_2007.12697_dx=5.csv")
    
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
            
            # Delta-function MF constraints
            m_delta_evap = np.logspace(11, 21, 1000)

            constraints_names_evap, f_PBHs_GC_delta = load_results_Isatis(modified=True)
            f_PBH_delta_evap = envelope(f_PBHs_GC_delta)

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
            
            # Delta-function MF constraints
            m_delta_values_loaded, f_max_loaded = load_data("./2302.04408/2302.04408_MW_diffuse_SPI.csv")
            # Power-law exponent to use between 1e15g and 1e16g.
            exponent_PL_upper = 2.0
            # Power-law exponent to use between 1e11g and 1e15g.
            exponent_PL_lower = 2.0
            
            m_delta_extrapolated_upper = np.logspace(15, 16, 11)
            m_delta_extrapolated_lower = np.logspace(11, 15, 41)
            
            f_max_extrapolated_upper = min(f_max_loaded) * np.power(m_delta_extrapolated_upper / min(m_delta_values_loaded), exponent_PL_upper)
            f_max_extrapolated_lower = min(f_max_extrapolated_upper) * np.power(m_delta_extrapolated_lower / min(m_delta_extrapolated_upper), exponent_PL_lower)
        
            m_pbh_values_upper = np.concatenate((m_delta_extrapolated_upper, m_delta_values_loaded))
            f_max_upper = np.concatenate((f_max_extrapolated_upper, f_max_loaded))
            
            f_PBH_delta_evap = np.concatenate((f_max_extrapolated_lower, f_max_extrapolated_upper, f_max_loaded))
            m_delta_evap = np.concatenate((m_delta_extrapolated_lower, m_delta_extrapolated_upper, m_delta_values_loaded))


            # Path to extended MF constraints
            exponent_PL_lower = 2
            data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower) 
            
            fig.suptitle("Existing constraints (showing Korwar \& Profumo 2023 constraints), $\Delta={:.1f}$".format(Deltas[i]), fontsize="small")
            
            data_filename_LN = data_folder + "/LN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[i], exponent_PL_lower)
            data_filename_SLN = data_folder + "/SLN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[i], exponent_PL_lower)
            data_filename_CC3 = data_folder + "/CC3_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[i], exponent_PL_lower)
            
            mc_KP23_LN, f_PBH_KP23_LN = np.genfromtxt(data_filename_LN, delimiter="\t")
            mc_KP23_SLN, f_PBH_KP23_SLN = np.genfromtxt(data_filename_SLN, delimiter="\t")
            mp_KP23_CC3, f_PBH_KP23_CC3 = np.genfromtxt(data_filename_CC3, delimiter="\t")
            
            # print(f_PBH_KP23_SLN[20:30])
            
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
            

#%% Existing constraints: try plotting Korwar & Profumo (2023) constraints and Galactic Centre photon constraints on the same plot

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
        
        # Delta-function MF constraints
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
               
        # Delta-function MF constraints
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
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
            
    exponent_PL_lower = 2
    data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower)
    
    plot_LN = True
    plot_SLN = False
    plot_CC3 = False
    
    plot_unevolved = True
    
    fig, ax = plt.subplots(figsize=(6,6))
    fig1, ax1 = plt.subplots(figsize=(6,6))
       
    # Delta-function MF constraints
    m_delta_values_loaded, f_max_loaded = load_data("2302.04408/2302.04408_MW_diffuse_SPI.csv")
    
    # Power-law exponent to use between 1e15g and 1e16g.
    exponent_PL_upper = 2.0
    # Power-law exponent to use between 1e11g and 1e15g.
    exponent_PL_lower = 2.0
    
    m_delta_extrapolated_upper = np.logspace(15, 16, 11)
    m_delta_extrapolated_lower = np.logspace(11, 15, 41)
    
    f_max_extrapolated_upper = min(f_max_loaded) * np.power(m_delta_extrapolated_upper / min(m_delta_values_loaded), exponent_PL_upper)
    f_max_extrapolated_lower = min(f_max_extrapolated_upper) * np.power(m_delta_extrapolated_lower / min(m_delta_extrapolated_upper), exponent_PL_lower)

    m_pbh_values_upper = np.concatenate((m_delta_extrapolated_upper, m_delta_values_loaded))
    f_max_upper = np.concatenate((f_max_extrapolated_upper, f_max_loaded))
    
    f_PBH_delta_evap = np.concatenate((f_max_extrapolated_lower, f_max_extrapolated_upper, f_max_loaded))
    m_delta_evap = np.concatenate((m_delta_extrapolated_lower, m_delta_extrapolated_upper, m_delta_values_loaded))

    ax.plot(m_delta_evap, f_PBH_delta_evap, color="tab:gray", label="Delta func.", linewidth=2)

    colors=["tab:blue", "tab:orange", "tab:green", "tab:red"]
        
    for i, Delta_index in enumerate([0, 5, 6]):
    #for i, Delta_index in enumerate([1, 2, 3, 4]):
        
        data_filename_LN = data_folder + "/LN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)
        data_filename_SLN = data_folder + "/SLN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)
        data_filename_CC3 = data_folder + "/CC3_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)
    
        mc_KP23_LN, f_PBH_KP23_LN = np.genfromtxt(data_filename_LN, delimiter="\t")
        mc_KP23_SLN, f_PBH_KP23_SLN = np.genfromtxt(data_filename_SLN, delimiter="\t")
        mp_KP23_CC3, f_PBH_KP23_CC3 = np.genfromtxt(data_filename_CC3, delimiter="\t")

        if plot_LN:
            mp_LN = mc_KP23_LN * np.exp(-sigmas_LN[Delta_index]**2)
            ax.plot(mp_LN, f_PBH_KP23_LN, color=colors[i], dashes=[6, 2], label="{:.1f}".format(Deltas[Delta_index]))
                
            f_max_interpolated = 10**np.interp(np.log10(mp_LN), np.log10(m_delta_evap), np.log10(f_PBH_delta_evap))
            frac_diff = (f_PBH_KP23_LN / f_max_interpolated) - 1
            ax1.plot(mp_LN, np.abs(frac_diff), color=colors[i], label="{:.1f}".format(Deltas[Delta_index]))

            
        elif plot_SLN:
            # Estimate peak mass of skew-lognormal MF
            mp_KP23_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index], log_m_factor=3, n_steps=1000) for m_c in mc_KP23_SLN]
            ax.plot(mp_KP23_SLN, f_PBH_KP23_SLN, color=colors[i], linestyle=(0, (5, 7)), label="{:.1f}".format(Deltas[Delta_index]))
            
            f_max_interpolated = 10**np.interp(np.log10(mp_KP23_SLN), np.log10(m_delta_evap), np.log10(f_PBH_delta_evap))
            frac_diff = (f_PBH_KP23_SLN / f_max_interpolated) - 1
            ax1.plot(mp_KP23_SLN, np.abs(frac_diff), color=colors[i], label="{:.1f}".format(Deltas[Delta_index]))
        
        elif plot_CC3:
            ax.plot(mp_KP23_CC3, f_PBH_KP23_CC3, color=colors[i], linestyle="dashed", label="{:.1f}".format(Deltas[Delta_index]))
            
            f_max_interpolated = 10**np.interp(np.log10(mp_KP23_CC3), np.log10(m_delta_evap), np.log10(f_PBH_delta_evap))
            frac_diff = (f_PBH_KP23_CC3 / f_max_interpolated) - 1
            ax1.plot(mp_KP23_CC3, np.abs(frac_diff), color=colors[i], label="{:.1f}".format(Deltas[Delta_index]))               

            
        # Plot constraint obtained with unevolved MF
        if plot_unevolved:
            """
            # Load constraints calculated for the unevolved MF extrapolated down to 5e14g using a power-law with exponent 2 calculated using GC_constraints_Carr.py (before May 2023)
            mc_KP23_SLN, f_PBH_KP23_SLN = np.genfromtxt("./Data-old/SLN_2302.04408_Carr_Delta={:.1f}_extrapolated.txt".format(Deltas[Delta_index]), delimiter="\t")
            mp_KP23_CC3, f_PBH_KP23_CC3 = np.genfromtxt("./Data-old/CC3_2302.04408_Carr_Delta={:.1f}_extrapolated.txt".format(Deltas[Delta_index]), delimiter="\t")
            mc_KP23_LN, f_PBH_KP23_LN = np.genfromtxt("./Data-old/LN_2302.04408_Carr_Delta={:.1f}_extrapolated.txt".format(Deltas[Delta_index]), delimiter="\t")
            """
            """
            # Load constraints calculated for the unevolved MF extrapolated down to 5e14g using a power-law with exponent 2 calculated using GC_constraints_Carr.py (August 2023)
            mc_KP23_SLN, f_PBH_KP23_SLN = np.genfromtxt("./Data-tests/unevolved/upper_PL_exp_2/SLN_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[Delta_index]), delimiter="\t")
            mp_KP23_CC3, f_PBH_KP23_CC3 = np.genfromtxt("./Data-tests/unevolved/upper_PL_exp_2/CC3_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[Delta_index]), delimiter="\t")
            mc_KP23_LN, f_PBH_KP23_LN = np.genfromtxt("./Data-tests/unevolved/upper_PL_exp_2/LN_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[Delta_index]), delimiter="\t")
            """
            # Load constraints calculated for the unevolved MF extrapolated down to 1e11g using a power-law with exponent 2 calculated using GC_constraints_Carr.py (August 2023)
            mc_KP23_SLN, f_PBH_KP23_SLN = np.genfromtxt("./Data-tests/unevolved/PL_exp_2/SLN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp2.txt".format(Deltas[Delta_index]), delimiter="\t")
            mp_KP23_CC3, f_PBH_KP23_CC3 = np.genfromtxt("./Data-tests/unevolved/PL_exp_2/CC3_2302.04408_Carr_Delta={:.1f}_extrapolated_exp2.txt".format(Deltas[Delta_index]), delimiter="\t")
            mc_KP23_LN, f_PBH_KP23_LN = np.genfromtxt("./Data-tests/unevolved/PL_exp_2/LN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp2.txt".format(Deltas[Delta_index]), delimiter="\t")


            if plot_LN:
                mp_LN = mc_KP23_LN * np.exp(-sigmas_LN[Delta_index]**2)                
                ax.plot(mp_LN, f_PBH_KP23_LN, color=colors[i], alpha=0.4)
                                
            elif plot_SLN:
                # Estimate peak mass of skew-lognormal MF
                mp_KP23_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index], log_m_factor=3, n_steps=1000) for m_c in mc_KP23_SLN]            
                ax.plot(mp_KP23_SLN, f_PBH_KP23_SLN, color=colors[i], alpha=0.4)
     
            elif plot_CC3:
                ax.plot(mp_KP23_CC3, f_PBH_KP23_CC3, color=colors[i], alpha=0.4)
        
    ax.set_ylabel("$f_\mathrm{PBH}$")
    ax1.set_ylabel("$|f_\mathrm{PBH} / f_\mathrm{max} - 1|$")
    
    for a in [ax, ax1]:
        a.set_xlabel("$m_p~[\mathrm{g}]$")
        a.legend(title="$\Delta$", fontsize="x-small")
        a.set_xscale("log")
        a.set_yscale("log")
        
    ax.set_xlim(1e16, 2e18)
    ax.set_ylim(1e-5, 1)
    ax1.set_xlim(1e15, max(m_delta_evap))
    ax1.set_ylim(1e-2, 2)
    
    for f in [fig, fig1]:
        if plot_LN:
            f.suptitle("Korwar \& Profumo 2023 constraints (LN)", fontsize="small")
        elif plot_SLN:
            f.suptitle("Korwar \& Profumo 2023 constraints (SLN)", fontsize="small")
        elif plot_CC3:
            f.suptitle("Korwar \& Profumo 2023 constraints (CC3)", fontsize="small")
        f.tight_layout()
 
    x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
    ax1.xaxis.set_major_locator(x_major)
    x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 5)
    ax1.xaxis.set_minor_locator(x_minor)
    ax1.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())


    
#%% Plot the most stringent GC photon constraint

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
    plot_SLN = False
    plot_CC3 = True
        
    for i in range(len(Deltas)):
        
        fig, ax = plt.subplots(figsize=(8, 8))
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
            
            # Estimate peak mass of skew-lognormal MF
            mp_SLN_evap = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_SLN_evolved]
            
            if plot_LN:
                ax.plot(mc_LN_evolved * np.exp(-sigmas_LN[i]**2), f_PBH_LN_evolved, color=colors_evap[k], marker="x", label=constraints_names_short[k])
            elif plot_SLN:
                ax.plot(mp_SLN_evap, f_PBH_SLN_evolved, color=colors_evap[k], marker="x", label=constraints_names_short[k])
            elif plot_CC3:
                ax.plot(mp_CC3_evolved, f_PBH_CC3_evolved, color=colors_evap[k], marker="x", label=constraints_names_short[k])

        ax.set_xlim(1e16, 1e18)
        ax.set_ylim(10**(-6), 1)
        ax.set_xlabel("$m_p~[\mathrm{g}]$")
        ax.set_ylabel("$f_\mathrm{PBH}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize="x-small")
        fig.tight_layout()
        fig.suptitle("$\Delta={:.1f}$".format(Deltas[i]))

        
#%% Plot the GC photon, Korwar & Profumo (2023) constraints and Boudaud & Cirelli (2019) constraints on the same plot

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
                
    plot_LN = False
    plot_SLN = False
    plot_CC3 = True
    
    exponent_PL_lower = 2
    data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower)
    
    # For Boudaud & Cirelli (2019) constraints
    BC19_colours = ["b", "r"]
    linestyles = ["solid", "dashed"]
    
    prop_A = True
    with_bkg = False
    
    # For Galactic Centre photon constraints
    constraints_names_short = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
    mc_values_evap = np.logspace(14, 20, 120)


    for Delta_index in range(len(Deltas)):
        
        fig, ax = plt.subplots(figsize=(8, 8))
                
        for colour_index, prop in enumerate([True, True]):
            
            prop_B = not prop_A
            
            for linestyle_index, with_bkg in enumerate([False, True]):
            
                label=""
                
                if prop_A:
                    prop_string = "prop_A"
                    label = "Prop A"
        
                elif prop_B:
                    prop_string = "prop_B"
                    label = "Prop B"
                    
                if not with_bkg:
                    prop_string += "_nobkg"
                    label += " w/o bkg subtraction "
                else:
                    label += " w/ bkg subtraction"
                    
                data_filename_LN = data_folder + "/LN_1807.03075_Carr_" + prop_string + "_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)
                data_filename_SLN = data_folder + "/SLN_1807.03075_Carr_" + prop_string + "_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)
                data_filename_CC3 = data_folder + "/CC3_1807.03075_Carr_" + prop_string + "_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)
                
                mc_LN_evolved, f_PBH_LN_evolved = np.genfromtxt(data_filename_LN, delimiter="\t")
                mc_SLN_evolved, f_PBH_SLN_evolved = np.genfromtxt(data_filename_SLN, delimiter="\t")
                mp_CC3_evolved, f_PBH_CC3_evolved = np.genfromtxt(data_filename_CC3, delimiter="\t")
                
                if plot_LN:
                    mp_LN = mc_LN_evolved * np.exp(-sigmas_LN[Delta_index]**2)
                    ax.plot(mp_LN, f_PBH_LN_evolved, color=BC19_colours[colour_index], linestyle=linestyles[linestyle_index], label=label)
                elif plot_SLN:
                    mp_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index], log_m_factor=3, n_steps=1000) for m_c in mc_SLN_evolved]
                    ax.plot(mp_SLN, f_PBH_SLN_evolved, color=BC19_colours[colour_index], linestyle=linestyles[linestyle_index], label=label)
                elif plot_CC3:
                    ax.plot(mp_CC3_evolved, f_PBH_CC3_evolved, color=BC19_colours[colour_index], linestyle=linestyles[linestyle_index], label="BC '19 (" + label + ")")
                    
                with_bkg = not with_bkg
                   
            prop_A = not prop_A

        data_filename_LN = data_folder + "/LN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)
        data_filename_SLN = data_folder + "/SLN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)
        data_filename_CC3 = data_folder + "/CC3_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)
    
        mc_KP23_LN, f_PBH_KP23_LN = np.genfromtxt(data_filename_LN, delimiter="\t")
        mc_KP23_SLN, f_PBH_KP23_SLN = np.genfromtxt(data_filename_SLN, delimiter="\t")
        mp_KP23_CC3, f_PBH_KP23_CC3 = np.genfromtxt(data_filename_CC3, delimiter="\t")
            
    
        # Load constraints from Galactic Centre photons.
        f_PBH_instrument_LN = []
        f_PBH_instrument_SLN = []
        f_PBH_instrument_CC3 = []

        for k in range(len(constraints_names_short)):
            # Load constraints for an evolved extended mass function obtained from each instrument
            data_filename_LN = data_folder + "/LN_GC_%s" % constraints_names_short[k] + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[Delta_index])
            data_filename_SLN = data_folder + "/SLN_GC_%s" % constraints_names_short[k]  + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[Delta_index])
            data_filename_CC3 = data_folder + "/CC3_GC_%s" % constraints_names_short[k]  + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[Delta_index])
                
            mc_GC_LN, f_PBH_LN_evolved = np.genfromtxt(data_filename_LN, delimiter="\t")
            mc_GC_SLN, f_PBH_SLN_evolved = np.genfromtxt(data_filename_SLN, delimiter="\t")
            mp_GC_CC3, f_PBH_CC3_evolved = np.genfromtxt(data_filename_CC3, delimiter="\t")
            
            # Compile constraints from all instruments
            f_PBH_instrument_LN.append(f_PBH_LN_evolved)
            f_PBH_instrument_SLN.append(f_PBH_SLN_evolved)
            f_PBH_instrument_CC3.append(f_PBH_CC3_evolved)
 
        f_PBH_GC_LN = envelope(f_PBH_instrument_LN)
        f_PBH_GC_SLN = envelope(f_PBH_instrument_SLN)
        f_PBH_GC_CC3 = envelope(f_PBH_instrument_CC3)
        
        if plot_LN:
            ax.plot(mc_KP23_LN * np.exp(-sigmas_LN[Delta_index]**2), f_PBH_KP23_LN, color="k", label="KP '23")
            #ax.plot(mc_values_evap * np.exp(-sigmas_LN[Delta_index]**2), f_PBH_GC_LN, color="tab:grey", label="GC photons")
                
        elif plot_SLN:
            # Estimate peak mass of skew-lognormal MF
            mp_KP23_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index], log_m_factor=3, n_steps=1000) for m_c in mc_KP23_SLN]
            ax.plot(mp_KP23_SLN, f_PBH_KP23_SLN, color="k", label="KP '23")
            mp_GC_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index], log_m_factor=3, n_steps=1000) for m_c in mc_values_evap]
            #ax.plot(mp_GC_SLN, f_PBH_GC_SLN, color="tab:grey", label="GC photons")
        
        elif plot_CC3:
            ax.plot(mc_values_evap, f_PBH_KP23_CC3, color="k", label="KP '23")
            #ax.plot(mp_GC_CC3, f_PBH_GC_CC3, color="tab:grey", label="GC photons")

        ax.set_xlim(1e16, 5e18)
        ax.set_ylim(10**(-6), 1)
        ax.set_xlabel("$m_p~[\mathrm{g}]$")
        ax.set_ylabel("$f_\mathrm{PBH}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize="x-small")
        fig.tight_layout()
        fig.suptitle("$\Delta={:.1f}$".format(Deltas[Delta_index]))
    
        
#%% Test: log-normal plots. Aim is to understand why the extended MF constraints shown in Fig. 20 of 2002.12778 differ so much from the delta-function MF constraints, compared to the difference I'm seeing when plotting against the peak mass.
from preliminaries import constraint_Carr, LN

def g_to_Solmass(m):
    return 1.989e33 * m

def Solmass_to_g(m):
    return m / 1.989e33

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
            
    # Power-law exponent to use between 1e15g and 1e16g.
    exponent_PL_upper = 2.0
    # Power-law exponent to use between 1e11g and 1e15g.
    exponent_PL_lower = 2.0
    
    data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower)
    
    plot_unevolved = True
    plot_against_mc = False
    
    fig, ax = plt.subplots(figsize=(6,5))
       
    # Delta-function MF constraints
    m_delta_values_loaded, f_max_loaded = load_data("2302.04408/2302.04408_MW_diffuse_SPI.csv")
        
    m_delta_extrapolated_upper = np.logspace(15, 16, 11)
    m_delta_extrapolated_lower = np.logspace(11, 15, 41)
    
    f_max_extrapolated_upper = min(f_max_loaded) * np.power(m_delta_extrapolated_upper / min(m_delta_values_loaded), exponent_PL_upper)
    f_max_extrapolated_lower = min(f_max_extrapolated_upper) * np.power(m_delta_extrapolated_lower / min(m_delta_extrapolated_upper), exponent_PL_lower)

    m_pbh_values_upper = np.concatenate((m_delta_extrapolated_upper, m_delta_values_loaded))
    f_max_upper = np.concatenate((f_max_extrapolated_upper, f_max_loaded))
    
    f_PBH_delta_evap = np.concatenate((f_max_extrapolated_lower, f_max_extrapolated_upper, f_max_loaded))
    m_delta_evap = np.concatenate((m_delta_extrapolated_lower, m_delta_extrapolated_upper, m_delta_values_loaded))
    
    # Calculate extended MF constraint for a log-normal with sigma = 2
    sigma_Carr21 = 2
    mc_Carr21 = np.logspace(14, 22, 1000)
    f_PBH_sigma2_evolved = constraint_Carr(mc_Carr21, m_delta_evap, f_PBH_delta_evap, LN, [sigma_Carr21], evolved=True)
    f_PBH_sigma2_unevolved = constraint_Carr(mc_Carr21, m_delta_evap, f_PBH_delta_evap, LN, [sigma_Carr21], evolved=False)

    colors=["tab:blue", "tab:orange", "tab:green", "tab:red"]
    
    ax.plot(m_delta_evap, f_PBH_delta_evap, color="tab:gray", label="Delta func.", linewidth=2)
    ax1 = ax.secondary_xaxis('top', functions=(Solmass_to_g, g_to_Solmass)) 
        
    for i, Delta_index in enumerate([6]):
        
        data_filename_LN = data_folder + "/LN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)    
        mc_KP23_LN, f_PBH_KP23_LN = np.genfromtxt(data_filename_LN, delimiter="\t")
        if plot_against_mc:
            ax.plot(mc_KP23_LN, f_PBH_KP23_LN, color=colors[i], dashes=[6, 2], label="{:.1f}".format(sigmas_LN[Delta_index]))
        else:
            ax.plot(mc_KP23_LN * np.exp(-sigmas_LN[Delta_index]**2), f_PBH_KP23_LN, color=colors[i], dashes=[6, 2], label="{:.1f}".format(sigmas_LN[Delta_index]))
                        
        # Plot constraint obtained with unevolved MF
        if plot_unevolved:
            # Load constraints calculated for the unevolved MF extrapolated down to 1e11g using a power-law with exponent 2 calculated using GC_constraints_Carr.py (August 2023)
            mc_KP23_LN, f_PBH_KP23_LN = np.genfromtxt("./Data-tests/unevolved/PL_exp_2/LN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp2.txt".format(Deltas[Delta_index]), delimiter="\t")

            mp_LN = mc_KP23_LN * np.exp(-sigmas_LN[Delta_index]**2)     
            if plot_against_mc:
                ax.plot(mc_KP23_LN, f_PBH_KP23_LN, color=colors[i], alpha=0.4)
            else:
                ax.plot(mc_KP23_LN * np.exp(-sigmas_LN[Delta_index]**2), f_PBH_KP23_LN, color=colors[i], alpha=0.4)
                
    if plot_against_mc:
        ax.plot(mc_Carr21, f_PBH_sigma2_evolved, color="tab:orange", dashes=[6, 2], label="{:.1f}".format(2))
        if plot_unevolved:
            ax.plot(mc_Carr21, f_PBH_sigma2_unevolved, color="tab:orange", alpha=0.4)
        ax.set_xlabel("$m_c = m_p\exp(\sigma^2)~[\mathrm{g}]$")
        ax1.set_xlabel("$m_c~[M_\odot]$")
    else:
        ax.plot(mc_Carr21 * np.exp(-sigma_Carr21**2), f_PBH_sigma2_evolved, color="tab:orange", dashes=[6, 2], label="{:.1f}".format(2))
        if plot_unevolved:
            ax.plot(mc_Carr21 * np.exp(-sigma_Carr21**2), f_PBH_sigma2_unevolved, color="tab:orange", alpha=0.4)
        ax.set_xlabel("$m_p~[\mathrm{g}]$")
        ax1.set_xlabel("$m_p~[M_\odot]$")
       
    ax.set_ylabel("$f_\mathrm{PBH}$")
    ax.legend(title="$\sigma$", fontsize="x-small")
    ax.set_xscale("log")
    ax.set_yscale("log")
        
    ax.set_xlim(1e16, 1e20)
    ax.set_ylim(1e-6, 1)
    
    x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 10)
    ax.xaxis.set_major_locator(x_major)
    x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax.xaxis.set_minor_locator(x_minor)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        
    fig.tight_layout()
    
#%% Test: log-normal plots. Aim is to understand why the extended MF constraints shown in Fig. 20 of 2002.12778 differ so much from the delta-function MF constraints compared to the versions Im using.
    
    # Plot the constraints shown in Fig. 20 of 2002.12778
    m_min = 1e11
    m_max = 1e20
    epsilon = 0.4
    m_star = 5e14
    
    def f_PBH_beta(m_values, beta):
        # masses m_values must be in grams
        return 1.7e8 * beta * np.power(m_values / 1.989e33, -1/2)
    
    def beta(beta_prime, gamma=0.2):
        return beta_prime / np.sqrt(gamma) 
 
    def beta_prime_lower(m_values, epsilon=0.4):
        
        beta_prime_values = []
        
        for m in m_values:
            if m < m_star:
                beta_prime_values.append(5e-28 * np.power(m/m_star, -5/2-2*epsilon))
            else:
                beta_prime_values.append(5e-26 * np.power(m/m_star, 7/2+epsilon))
        
        return beta_prime_values
    
    m_pbh_values = 10**np.arange(np.log10(m_min), np.log10(m_max), 0.1)
    f_max_values = f_PBH_beta(m_pbh_values, beta(beta_prime_lower(m_pbh_values)))
    #f_max_values /= 3
    mc_values = np.logspace(15, 22, 70)
    
    sigma = 2
    
    fig, ax = plt.subplots(figsize=(6,5))
    ax1 = ax.secondary_xaxis('top', functions=(Solmass_to_g, g_to_Solmass)) 

    f_PBH_values = constraint_Carr(mc_values, m_pbh_values, f_max_values, LN, [sigma], evolved=False)

    ax.plot(m_pbh_values, f_max_values, color="k", label="Delta func. [repr.]", linestyle="dashed")
    
    m_delta_values_loaded, f_max_loaded = load_data("./2002.12778/Carr+21_mono_RH.csv")
    mc_LN_values_loaded, f_PBH_loaded = load_data("./2002.12778/Carr+21_Gamma_ray_LN_RH.csv")
    
    ax.plot(m_delta_values_loaded * 1.989e33, f_max_loaded, color="tab:grey", label="Delta func.")
    ax.plot(mc_LN_values_loaded * 1.989e33, f_PBH_loaded, color="lime", label="LN ($\sigma={:.1f}$)".format(sigma))
    ax.plot(mc_values, f_PBH_values, color="tab:green", label="LN ($\sigma={:.1f}$) [repr.]".format(sigma), linestyle="dashed")

    ax.set_ylabel("$f_\mathrm{PBH}$")
    ax.set_xlabel("$m_c~[\mathrm{g}]$")
    ax1.set_xlabel("$m_c~[M_\odot]$")
    ax.legend(title="$\sigma$", fontsize="x-small")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e14, 1e20)
    ax.set_ylim(1e-4, 1)
    fig.tight_layout()    
    

    
