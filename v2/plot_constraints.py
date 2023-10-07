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


def frac_diff(y1, y2, x1, x2, interp_log = True):
    """
    Find the fractional difference between two arrays (y1, y2), evaluated
    at (x1, x2), of the form (y1/y2 - 1).
    
    In the calculation, interpolation (logarithmic or linear) is used to 
    evaluate the array y2 at x-axis values x1.

    Parameters
    ----------
    y1 : Array-like
        Array to find fractional difference of, evaluated at x1.
    y2 : Array-like
        Array to find fractional difference of, evaluated at x2.
    x1 : Array-like
        x-axis values that y1 is evaluated at.
    x2 : Array-like
        x-axis values that y2 is evaluated at.
    interp_log : Boolean, optional
        If True, use logarithmic interpolation to evaluate y1 at x2. The default is True.

    Returns
    -------
    Array-like
        Fractional difference between y1 and y2.

    """
    if interp_log:
        return y1 / 10**np.interp(np.log10(x1), np.log10(x2), np.log10(y2)) - 1
    else:
        return np.interp(x1, x2, y2) / y1 - 1

#%% Tests of the method frac_diff:
    
if "__main__" == __name__:
    
    # Constant fractional difference
    y1, y2 = 1.5*np.ones(10), np.ones(10)
    x1 = x2 = np.linspace(0, 10, 10)
    print(frac_diff(y1, y2, x1, x2))
    
    y1, y2 = 1.5*np.ones(10), np.ones(30)
    x1 = np.linspace(0, 10, 10)
    x2 = np.linspace(-10, 20, 30)
    print(frac_diff(y1, y2, x1, x2))
    
    # x-dependent fractional difference
    x1 = x2 = np.linspace(1, 10, 10)
    y1 = 1 + x1**2
    y2 = np.ones(10)
    print(frac_diff(y1, y2, x1, x2))
       

#%% Existing constraints

if "__main__" == __name__:
    
    # If True, plot the evaporation constraints used by Isatis (from COMPTEL, INTEGRAL, EGRET and Fermi-LAT)
    plot_GC_Isatis = True
    # If True, plot the evaporation constraints shown in Korwar & Profumo (2023) [2302.04408]
    plot_KP23 = False
    # If True, plot the evaporation constraints from Boudaud & Cirelli (2019) [1807.03075]
    plot_BC19 = False
    # If True, use extended MF constraint calculated from the delta-function MF extrapolated down to 5e14g using a power-law fit
    include_extrapolated = False
    # If True, plot unevolved MF constraint
    plot_unevolved = True
    # If True, plot the fractional difference between evolved and unevolved MF results
    plot_fracdiff = True
    
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
                        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Loading constraints from Subaru-HSC.
        mc_Subaru_SLN, f_PBH_Subaru_SLN = np.genfromtxt("./Data/SLN_HSC_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
        mp_Subaru_CC3, f_PBH_Subaru_CC3 = np.genfromtxt("./Data/CC3_HSC_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
        mc_Subaru_LN, f_PBH_Subaru_LN = np.genfromtxt("./Data/LN_HSC_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
        mp_Subaru_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_Subaru_SLN]

        if plot_GC_Isatis:

            # If required, plot unevolved MF constraints.
            if plot_unevolved:            
                constraints_names_short = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
                f_PBH_instrument_LN_unevolved = []
                f_PBH_instrument_SLN_unevolved = []
                f_PBH_instrument_CC3_unevolved = []
    
                for k in range(len(constraints_names_short)):
    
                    # Plot constraints obtained with unevolved MF
                    # Load and plot results for the unevolved mass functions
                    data_filename_LN_unevolved_approx = "./Data-tests/unevolved" + "/LN_GC_%s" % constraints_names_short[k] + "_Carr_Delta={:.1f}_approx_unevolved.txt".format(Deltas[i])
                    data_filename_SLN_unevolved_approx = "./Data-tests/unevolved" + "/SLN_GC_%s" % constraints_names_short[k]  + "_Carr_Delta={:.1f}_approx_unevolved.txt".format(Deltas[i])
                    data_filename_CC3_unevolved_approx = "./Data-tests/unevolved" + "/CC3_GC_%s" % constraints_names_short[k]  + "_Carr_Delta={:.1f}_approx_unevolved.txt".format(Deltas[i])
        
                    mc_LN_unevolved_approx, f_PBH_LN_unevolved_approx = np.genfromtxt(data_filename_LN_unevolved_approx, delimiter="\t")
                    mc_SLN_unevolved_approx, f_PBH_SLN_unevolved_approx = np.genfromtxt(data_filename_SLN_unevolved_approx, delimiter="\t")
                    mp_CC3_unevolved_approx, f_PBH_CC3_unevolved_approx = np.genfromtxt(data_filename_CC3_unevolved_approx, delimiter="\t")
                                    
                    # Compile constraints from all instruments
                    f_PBH_instrument_LN_unevolved.append(f_PBH_LN_unevolved_approx)
                    f_PBH_instrument_SLN_unevolved.append(f_PBH_SLN_unevolved_approx)
                    f_PBH_instrument_CC3_unevolved.append(f_PBH_CC3_unevolved_approx)
                
                f_PBH_GC_LN_unevolved = envelope(f_PBH_instrument_LN_unevolved)
                f_PBH_GC_SLN_unevolved = envelope(f_PBH_instrument_SLN_unevolved)
                f_PBH_GC_CC3_unevolved = envelope(f_PBH_instrument_CC3_unevolved)
            
                mp_SLN_unevolved_approx = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_SLN_unevolved_approx]
                mp_LN_unevolved_approx = mc_LN_unevolved_approx * np.exp(-sigmas_LN[i]**2)
                
                ax.plot(mp_LN_unevolved_approx, f_PBH_GC_LN_unevolved, color=colors[1], alpha=0.4)
                ax.plot(mp_SLN_unevolved_approx, f_PBH_GC_SLN_unevolved, color=colors[2], alpha=0.4)
                ax.plot(mp_CC3_unevolved_approx, f_PBH_GC_CC3_unevolved, color=colors[3], alpha=0.4)
            
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
                        
            ax.plot(m_delta_evap, f_PBH_delta_evap, color=colors[0], linewidth=2)
            ax.plot(m_delta_Subaru, f_PBH_delta_Subaru, color=colors[0], linewidth=2)
            
            ax.set_xlabel("$m_p~[\mathrm{g}]$")                
            ax.plot(mp_SLN_evap, f_PBH_GC_SLN, color=colors[2], linestyle=(0, (5, 7)))
            ax.plot(mc_values_evap, f_PBH_GC_CC3, color=colors[3], linestyle="dashed")
            ax.plot(mc_values_evap * np.exp(-sigmas_LN[i]**2), f_PBH_GC_LN, color=colors[1], dashes=[6, 2])
            ax.set_ylabel("$f_\mathrm{PBH}$")
            ax.set_xscale("log")
            ax.set_yscale("log")
                                    
            if plot_unevolved and plot_fracdiff:
                fig1, ax1a = plt.subplots(figsize=(6,6))
                ax1a.plot(mc_values_evap * np.exp(-sigmas_LN[i]**2), np.abs(frac_diff(f_PBH_GC_LN_unevolved, f_PBH_GC_LN, mp_LN_unevolved_approx, mc_values_evap * np.exp(-sigmas_LN[i]**2))), label="LN", color="r")
                ax1a.plot(mp_SLN_evap, np.abs(frac_diff(f_PBH_GC_SLN_unevolved, f_PBH_GC_SLN, mp_SLN_unevolved_approx, mp_SLN_evap)), label="SLN", color="b")
                ax1a.plot(mc_values_evap, np.abs(frac_diff(f_PBH_GC_CC3_unevolved, f_PBH_GC_CC3, mp_CC3_unevolved_approx, mc_values_evap)), label="CC3", color="g")
                ax1a.set_ylabel("$|\Delta f_\mathrm{PBH} / f_\mathrm{PBH}|$")
                ax1a.set_xlabel("$m_p~[\mathrm{g}]$")
                ax1a.set_xscale("log")
                ax1a.set_yscale("log")
                ax1a.set_title("$\Delta={:.1f}$".format(Deltas[i]))
                ax1a.legend(title="Unevolved/evolved - 1", fontsize="x-small")
                ax1a.set_xlim(xmin=1e16)
                ax1a.set_ylim(ymax=1e2)
                ax1a.grid()
                fig1.tight_layout()

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
                        
            # Estimate peak mass of skew-lognormal MF
            mp_KP23_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_KP23_SLN]
            
            ax.set_xlabel("$m_p~[\mathrm{g}]$")
            ax.plot(m_delta_evap, f_PBH_delta_evap, color=colors[0], linewidth=2)
            ax.plot(m_delta_Subaru, f_PBH_delta_Subaru, color=colors[0], linewidth=2)
            ax.set_ylabel("$f_\mathrm{PBH}$")
            ax.set_xscale("log")
            ax.set_yscale("log")
            
            ax.plot(mp_KP23_SLN, f_PBH_KP23_SLN, color=colors[2], linestyle=(0, (5, 7)))
            ax.plot(mp_KP23_CC3, f_PBH_KP23_CC3, color=colors[3], linestyle="dashed")
            ax.plot(mc_KP23_LN * np.exp(-sigmas_LN[i]**2), f_PBH_KP23_LN, color=colors[1], dashes=[6, 2])
            
            # If required, plot the fractional difference from the delta-function MF constraint
            if plot_fracdiff:
                fig1, ax1a = plt.subplots(figsize=(6,6))
                ax1a.plot(mc_KP23_LN * np.exp(-sigmas_LN[i]**2), np.abs(frac_diff(f_PBH_KP23_LN, f_PBH_delta_evap, mc_KP23_LN * np.exp(-sigmas_LN[i]**2), m_delta_evap)), label="LN", color="r")
                ax1a.plot(mp_KP23_SLN, np.abs(frac_diff(f_PBH_KP23_SLN, f_PBH_delta_evap, mp_KP23_SLN, m_delta_evap)), label="SLN", color="b")
                ax1a.plot(mp_KP23_CC3, np.abs(frac_diff(f_PBH_KP23_CC3, f_PBH_delta_evap, mp_KP23_CC3, m_delta_evap)), label="CC3", color="g")
                ax1a.set_ylabel("$|\Delta f_\mathrm{PBH} / f_\mathrm{PBH}|$")
                ax1a.set_xlabel("$m_p~[\mathrm{g}]$")
                ax1a.set_xscale("log")
                ax1a.set_yscale("log")
                ax1a.set_title("$\Delta={:.1f}$".format(Deltas[i]))
                ax1a.legend(title="Delta func./extended MF - 1", fontsize="x-small")
                ax1a.set_xlim(xmin=5e14)
                ax1a.set_ylim(1e-4, 10)
                ax1a.grid()
                fig1.tight_layout()
                
            # If required, plot constraints obtained with unevolved MF
            if plot_unevolved:
                """
                mc_KP23_SLN, f_PBH_KP23_SLN = np.genfromtxt("./Data-old/SLN_2302.04408_Carr_Delta={:.1f}_extrapolated.txt".format(Deltas[i]), delimiter="\t")
                mp_KP23_CC3, f_PBH_KP23_CC3 = np.genfromtxt("./Data-old/CC3_2302.04408_Carr_Delta={:.1f}_extrapolated.txt".format(Deltas[i]), delimiter="\t")
                mc_KP23_LN, f_PBH_KP23_LN = np.genfromtxt("./Data-old/LN_2302.04408_Carr_Delta={:.1f}_extrapolated.txt".format(Deltas[i]), delimiter="\t")
                """
                
                # Load constraints calculated for the unevolved MF extrapolated down to 1e11g using a power-law with exponent 2 calculated using GC_constraints_Carr.py (August 2023)
                mc_KP23_SLN, f_PBH_KP23_SLN = np.genfromtxt("./Data-tests/unevolved/PL_exp_2/SLN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp2.txt".format(Deltas[i]), delimiter="\t")
                mp_KP23_CC3, f_PBH_KP23_CC3 = np.genfromtxt("./Data-tests/unevolved/PL_exp_2/CC3_2302.04408_Carr_Delta={:.1f}_extrapolated_exp2.txt".format(Deltas[i]), delimiter="\t")
                mc_KP23_LN, f_PBH_KP23_LN = np.genfromtxt("./Data-tests/unevolved/PL_exp_2/LN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp2.txt".format(Deltas[i]), delimiter="\t")                

                # Estimate peak mass of skew-lognormal MF
                mp_KP23_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_KP23_SLN]
            
                ax.plot(mp_KP23_SLN, f_PBH_KP23_SLN, color=colors[2], alpha=0.4)
                ax.plot(mp_KP23_CC3, f_PBH_KP23_CC3, color=colors[3], alpha=0.4)
                ax.plot(mc_KP23_LN * np.exp(-sigmas_LN[i]**2), f_PBH_KP23_LN, color=colors[1], alpha=0.4)
        
        elif plot_BC19:
            
            exponent_PL_lower = 2
            # Boolean determines which propagation model to load data from
            for prop_A in [True, False]:
                prop_B = not prop_A
            
                # Boolean determines whether to load constraint obtained with a background or without a background
                for with_bkg in [True, False]:
                            
                    if prop_A:
                        prop_string = "prop_A"
                        if with_bkg:
                            m_delta_values, f_max = load_data("1807.03075/1807.03075_prop_A_bkg.csv")
                        else:
                            m_delta_values, f_max = load_data("1807.03075/1807.03075_prop_A_nobkg.csv")
                
                    elif prop_B:
                        prop_string = "prop_B"
                        if with_bkg:
                            m_delta_values, f_max = load_data("1807.03075/1807.03075_prop_B_bkg.csv")
                        else:
                            m_delta_values, f_max = load_data("1807.03075/1807.03075_prop_B_nobkg.csv")
                                
                    if not with_bkg:
                        prop_string += "_nobkg"
                        linestyle = "dashed"

                    else:
                        linestyle = "dotted"                                                    
                
                    data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower) 
                    data_filename_LN = data_folder + "/LN_1807.03075_Carr_" + prop_string + "_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[i], exponent_PL_lower)
                    data_filename_SLN = data_folder + "/SLN_1807.03075_Carr_" + prop_string + "_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[i], exponent_PL_lower)
                    data_filename_CC3 = data_folder + "/CC3_1807.03075_Carr_" + prop_string + "_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[i], exponent_PL_lower)
                        
                    mc_BC19_LN, f_PBH_BC19_LN = np.genfromtxt(data_filename_LN, delimiter="\t")
                    mc_BC19_SLN, f_PBH_BC19_SLN = np.genfromtxt(data_filename_SLN, delimiter="\t")
                    mp_BC19_CC3, f_PBH_BC19_CC3 = np.genfromtxt(data_filename_CC3, delimiter="\t")
                    
                    # Estimate peak mass of skew-lognormal MF
                    mp_BC19_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_BC19_SLN]
                    
                    ax.set_xlabel("$m_p~[\mathrm{g}]$")
                    ax.plot(m_delta_values, f_max, color=colors[0], linewidth=2, linestyle=linestyle)
                    ax.set_ylabel("$f_\mathrm{PBH}$")
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    
                    ax.plot(mp_BC19_SLN, f_PBH_BC19_SLN, color=colors[2], linestyle=linestyle)
                    ax.plot(mp_BC19_CC3, f_PBH_BC19_CC3, color=colors[3], linestyle=linestyle)
                    ax.plot(mc_BC19_LN * np.exp(-sigmas_LN[i]**2), f_PBH_BC19_LN, color=colors[1], linestyle=linestyle)
                       
            plt.suptitle("Existing constraints (showing Voyager 1 constraints), $\Delta={:.1f}$".format(Deltas[i]), fontsize="small")

        # Plot delta-function and extended MF constraints from Subaru-HSC observations
        ax.plot(m_delta_Subaru, f_PBH_delta_Subaru, color=colors[0], linewidth=2, label="Delta function")
        ax.plot(mp_Subaru_SLN, f_PBH_Subaru_SLN, color=colors[2], label="SLN", linestyle=(0, (5, 7)))
        ax.plot(mp_Subaru_CC3, f_PBH_Subaru_CC3, color=colors[3], label="CC3", linestyle="dashed")
        ax.plot(mc_Subaru_LN * np.exp(-sigmas_LN[i]**2), f_PBH_Subaru_LN, color=colors[1], label="LN", dashes=[6, 2])

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
        
        ax.grid()

        # Set axis limits
        
        ymin, ymax = 1e-3, 1

        if Deltas[i] < 5:
            xmin_evap, xmax_evap = 1e16, 2.5e17
            xmin_HSC, xmax_HSC = 1e21, 1e29
            
            if plot_KP23:
                xmin_evap, xmax_evap = 1e16, 7e17
                
            elif plot_BC19:
                xmin_evap, xmax_evap = 1e16, 3e17
       
        else:
            xmin_evap, xmax_evap = 1e16, 7e17
            xmin_HSC, xmax_HSC = 9e18, 1e29

            if plot_KP23:
                xmin_evap, xmax_evap = 1e16, 2e18
                
            elif plot_BC19:
                xmin_evap, xmax_evap = 1e16, 1e19

        ax.set_xlim(xmin_evap, 1e24)
        ax.set_ylim(ymin, ymax)
       
        ax.legend(fontsize="xx-small")
        fig.tight_layout()
        """
        if plot_GC_Isatis:
            fig.savefig("./Results/Figures/fPBH_Delta={:.1f}_existing_GC_Isatis.pdf".format(Deltas[i]))
            fig.savefig("./Results/Figures/fPBH_Delta={:.1f}_existing_GC_Isatis.png".format(Deltas[i]))
            
        elif plot_KP23:
            fig.savefig("./Results/Figures/fPBH_Delta={:.1f}_existing_KP23.pdf".format(Deltas[i]))
            fig.savefig("./Results/Figures/fPBH_Delta={:.1f}_existing_KP23.png".format(Deltas[i]))

        elif plot_BC19:
            fig.savefig("./Results/Figures/fPBH_Delta={:.1f}_existing_BC19.pdf".format(Deltas[i]))
            fig.savefig("./Results/Figures/fPBH_Delta={:.1f}_existing_BC19.png".format(Deltas[i]))
        """
                
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
    colors = ['tab:gray', 'crimson', 'tab:blue', 'lime', 'k']
    
    # Linestyles for different constraints
    linestyles = ["dashdot", "dashed", "dotted"]
    
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
        
        # Loading constraints from Subaru-HSC
        m_delta_Subaru, f_PBH_delta_Subaru = load_data("2007.12697/Subaru-HSC_2007.12697_dx=5.csv")
        mc_Carr_SLN, f_PBH_Carr_SLN = np.genfromtxt("./Data/SLN_HSC_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
        mp_Subaru_CC3, f_PBH_Carr_CC3 = np.genfromtxt("./Data/CC3_HSC_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
        mc_Carr_LN, f_PBH_Carr_LN = np.genfromtxt("./Data/LN_HSC_Carr_Delta={:.1f}.txt".format(Deltas[i]), delimiter="\t")
        
        plt.suptitle("$\Delta={:.1f}$".format(Deltas[i]), fontsize="small")
        
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
            ax.plot(m_delta_Subaru, f_PBH_delta_Subaru, color=colors[0], linewidth=2, linestyle=linestyles[0])
            ax.plot(m_delta_evap, f_PBH_delta_evap, color=colors[0], label="Delta function", linewidth=2, linestyle=linestyles[1])
            ax.plot(m_delta_KP23, f_PBH_delta_KP23, color=colors[0], linewidth=2, linestyle=linestyles[2])
            ax.set_ylabel("$f_\mathrm{PBH}$")
            ax.set_xscale("log")
            ax.set_yscale("log")
            
            # Subaru-HSC constraints
            ax.plot(mp_Subaru_SLN, f_PBH_Carr_SLN, color=colors[2], label="SLN", linestyle=linestyles[0])
            ax.plot(mp_Subaru_CC3, f_PBH_Carr_CC3, color=colors[3], label="CC3", linestyle=linestyles[0])
            ax.plot(mc_Carr_LN * np.exp(-sigmas_LN[i]**2), f_PBH_Carr_LN, color=colors[1], label="LN", linestyle=linestyles[0])
            
            # Galactic Centre photon constraints
            ax.plot(mp_SLN_evap, f_PBH_GC_SLN, color=colors[2], linestyle=linestyles[1])
            ax.plot(mc_values_evap, f_PBH_GC_CC3, color=colors[3], linestyle=linestyles[1])
            ax.plot(mc_values_evap * np.exp(-sigmas_LN[i]**2), f_PBH_GC_LN, color=colors[1], linestyle=linestyles[1])
            
            # Korwar & Profumo (2023) constraints
            ax.plot(mp_KP23_SLN, f_PBH_KP23_SLN, color=colors[2], linestyle=linestyles[2])
            ax.plot(mp_KP23_CC3, f_PBH_KP23_CC3, color=colors[3], linestyle=linestyles[2])
            ax.plot(mc_KP23_LN * np.exp(-sigmas_LN[i]**2), f_PBH_KP23_LN, color=colors[1], linestyle=linestyles[2])
                    
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




#%% Plot constraints for different Delta on the same plot

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
            
    exponent_PL_lower = 2
    data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower)
            
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, hspace=0, tight_layout = {'pad': 0}, figsize=(10,15))
       
    # Linestyles for different constraints
    linestyles = ["dashdot", "dashed", "dotted"]
       
    # Delta-function MF constraints
    m_delta_GC = np.logspace(11, 21, 1000)
    constraints_names_GC, f_PBHs_GC_delta = load_results_Isatis(modified=True)
    f_PBH_delta_GC = envelope(f_PBHs_GC_delta)
    m_delta_Subaru, f_PBH_delta_Subaru = load_data("2007.12697/Subaru-HSC_2007.12697_dx=5.csv")

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
    f_PBH_delta_KP23 = np.concatenate((f_max_extrapolated_lower, f_max_extrapolated_upper, f_max_loaded))
    m_delta_KP23 = np.concatenate((m_delta_extrapolated_lower, m_delta_extrapolated_upper, m_delta_values_loaded))

    colors_LN = ["k", "tab:red", "pink"]
    colors_SLN = ["k", "tab:blue", "deepskyblue"]
    colors_CC3 = ["k", "tab:green", "lime"]
    
    # Opacities and line widths for different Delta:
    linewidth_values = [2, 1.5, 1]
    
    for ax in [ax0, ax1, ax2]:
        ax.plot(0, 0, color="tab:gray", linewidth=2, label="$\delta$ func.")
        ax.plot(m_delta_Subaru, f_PBH_delta_Subaru, color="tab:gray", linewidth=2, linestyle=linestyles[0])
        ax.plot(m_delta_GC, f_PBH_delta_GC, color="tab:gray", linewidth=2, linestyle=linestyles[1])
        ax.plot(m_delta_KP23, f_PBH_delta_KP23, color="tab:gray", linewidth=2, linestyle=linestyles[2])

    
    for i, Delta_index in enumerate([0, 5, 6]):
        
        # Loading constraints from Subaru-HSC
        mc_Subaru_SLN, f_PBH_Subaru_SLN = np.genfromtxt("./Data/SLN_HSC_Carr_Delta={:.1f}.txt".format(Deltas[Delta_index]), delimiter="\t")
        mp_Subaru_CC3, f_PBH_Subaru_CC3 = np.genfromtxt("./Data/CC3_HSC_Carr_Delta={:.1f}.txt".format(Deltas[Delta_index]), delimiter="\t")
        mc_Subaru_LN, f_PBH_Subaru_LN = np.genfromtxt("./Data/LN_HSC_Carr_Delta={:.1f}.txt".format(Deltas[Delta_index]), delimiter="\t")
        
        # Galactic Centre photon constraints, from Isatis
        mc_values_GC = np.logspace(14, 20, 120)

        exponent_PL_lower = 2
        constraints_names_short = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
        data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower)
        
        f_PBH_instrument_LN = []
        f_PBH_instrument_SLN = []
        f_PBH_instrument_CC3 = []

        for k in range(len(constraints_names_short)):
            # Load constraints for an evolved extended mass function obtained from each instrument
            data_filename_LN = data_folder + "/LN_GC_%s" % constraints_names_short[k] + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[Delta_index])
            data_filename_SLN = data_folder + "/SLN_GC_%s" % constraints_names_short[k]  + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[Delta_index])
            data_filename_CC3 = data_folder + "/CC3_GC_%s" % constraints_names_short[k]  + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[Delta_index])
                
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

        # Korwar & Profumo (2023) constraints
        data_filename_LN = data_folder + "/LN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)
        data_filename_SLN = data_folder + "/SLN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)
        data_filename_CC3 = data_folder + "/CC3_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)
        mc_KP23_LN, f_PBH_KP23_LN = np.genfromtxt(data_filename_LN, delimiter="\t")
        mc_KP23_SLN, f_PBH_KP23_SLN = np.genfromtxt(data_filename_SLN, delimiter="\t")
        mp_KP23_CC3, f_PBH_KP23_CC3 = np.genfromtxt(data_filename_CC3, delimiter="\t")

        # Plot LN MF results
        mp_KP23_LN = mc_KP23_LN * np.exp(-sigmas_LN[Delta_index]**2)
        mp_GC_LN = mc_values_GC * np.exp(-sigmas_LN[Delta_index]**2)
        mp_Subaru_LN = mc_Subaru_SLN * np.exp(-sigmas_LN[Delta_index]**2) 
        ax0.plot(mp_Subaru_LN, f_PBH_Subaru_LN, color=colors_LN[i], linestyle=linestyles[0], linewidth=linewidth_values[i])
        ax0.plot(mp_GC_LN, f_PBH_GC_LN, color=colors_LN[i], linestyle=linestyles[1], linewidth=linewidth_values[i])
        ax0.plot(mp_KP23_LN, f_PBH_KP23_LN, color=colors_LN[i], linestyle=linestyles[2], linewidth=linewidth_values[i])
        ax0.plot(0,0, color=colors_LN[i], label="$\Delta={:.0f}$".format(Deltas[Delta_index]))
        ax0a = ax0.twinx()
        ax0a.legend(title="LN", loc="upper right", fontsize="x-small")
        ax0a.axis("off")
        
        # Plot SLN MF results
        mp_KP23_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index], log_m_factor=3, n_steps=1000) for m_c in mc_KP23_SLN]
        mp_GC_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index], log_m_factor=3, n_steps=1000) for m_c in mc_values_GC]
        mp_Subaru_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index], log_m_factor=3, n_steps=1000) for m_c in mc_Subaru_SLN]
        ax1.plot(mp_Subaru_SLN, f_PBH_Subaru_SLN, color=colors_SLN[i], linestyle=linestyles[0], linewidth=linewidth_values[i])
        ax1.plot(mp_GC_SLN, f_PBH_GC_SLN, color=colors_SLN[i], linestyle=linestyles[1], linewidth=linewidth_values[i])
        ax1.plot(mp_KP23_SLN, f_PBH_KP23_SLN, color=colors_SLN[i], linestyle=linestyles[2], linewidth=linewidth_values[i])
        ax1.plot(0,0, color=colors_SLN[i], label="$\Delta={:.0f}$".format(Deltas[Delta_index]))
        ax1a = ax1.twinx()
        ax1a.legend(title="SLN", loc="upper right", fontsize="x-small")
        ax1a.axis("off")
        
        # Plot CC3 MF results
        mp_GC_CC3 = mc_values_GC
        ax2.plot(mp_Subaru_CC3, f_PBH_Subaru_CC3, color=colors_CC3[i], linestyle=linestyles[0], linewidth=linewidth_values[i])
        ax2.plot(mp_GC_CC3, f_PBH_GC_CC3, color=colors_CC3[i], linestyle=linestyles[1], linewidth=linewidth_values[i])
        ax2.plot(mp_KP23_CC3, f_PBH_KP23_CC3, color=colors_CC3[i], linestyle=linestyles[2], linewidth=linewidth_values[i])
        ax2.plot(0,0, color=colors_CC3[i], label="$\Delta={:.0f}$".format(Deltas[Delta_index]))
        ax2a = ax2.twinx()
        ax2a.legend(title="CC3", loc="upper right", fontsize="x-small")
        ax2a.axis("off")
        
    for ax in[ax0, ax1, ax2]:
        ax.set_ylabel("$f_\mathrm{PBH}$")

        ax.set_xscale("log")
        ax.set_yscale("log")  
        ax.set_xlim(1e16, 1e24)
        ax.set_ylim(1e-3, 1)
        ax.legend(fontsize="x-small")
            
        x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 10)
        ax.xaxis.set_major_locator(x_major)
        x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
        ax.xaxis.set_minor_locator(x_minor)
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.grid()
        
    ax2.set_xlabel("$m_p~[\mathrm{g}]$")
    #fig.tight_layout()
 
    
#%% Plot the most stringent GC photon constraint (calculated using the method from 1705.05567 with the delta-function MF constraint from Isatis).

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    colors_evap = ["tab:orange", "tab:green", "tab:red", "tab:blue"]
    constraints_names_short = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
    
    # if True, use the constraints obtained for evolved MFs. Otherwise, use constraints obtained using the unevolved MFs.
    evolved = False
    
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
        
    for i in range(len(Deltas)):
        
        fig, ax = plt.subplots(figsize=(8, 8))
        # Load constraints from Galactic Centre photons.
        
        exponent_PL_lower = 2
        constraints_names_short = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
        if evolved:
            data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower)
        else:
            data_folder = "./Data-tests/unevolved/PL_exp_{:.0f}".format(exponent_PL_lower)
        f_PBH_instrument_LN = []
        f_PBH_instrument_SLN = []
        f_PBH_instrument_CC3 = []

        for k in range(len(constraints_names_short)):
            # Load constraints for an evolved extended mass function obtained from each instrument
            if evolved:
                data_filename_LN = data_folder + "/LN_GC_%s" % constraints_names_short[k] + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[i])
                data_filename_SLN = data_folder + "/SLN_GC_%s" % constraints_names_short[k]  + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[i])
                data_filename_CC3 = data_folder + "/CC3_GC_%s" % constraints_names_short[k]  + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[i])
            else:
                data_filename_LN = data_folder + "/LN_GC_%s" % constraints_names_short[k] + "_Carr_Delta={:.1f}_approx_unevolved.txt".format(Deltas[i])
                data_filename_SLN = data_folder + "/SLN_GC_%s" % constraints_names_short[k]  + "_Carr_Delta={:.1f}_approx_unevolved.txt".format(Deltas[i])
                data_filename_CC3 = data_folder + "/CC3_GC_%s" % constraints_names_short[k]  + "_Carr_Delta={:.1f}_approx_unevolved.txt".format(Deltas[i])

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
    
    exponent_PL_lower = 3
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
            
        """
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
        """
        if plot_LN:
            ax.plot(mc_KP23_LN * np.exp(-sigmas_LN[Delta_index]**2), f_PBH_KP23_LN, color="k", label="KP '23")
            #ax.plot(mc_values_evap * np.exp(-sigmas_LN[Delta_index]**2), f_PBH_GC_LN, color="tab:grey", label="GC photons")
                
        elif plot_SLN:
            # Estimate peak mass of skew-lognormal MF
            mp_KP23_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index], log_m_factor=3, n_steps=1000) for m_c in mc_KP23_SLN]
            ax.plot(mp_KP23_SLN, f_PBH_KP23_SLN, color="k", label="KP '23")
            #mp_GC_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index], log_m_factor=3, n_steps=1000) for m_c in mc_values_evap]
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
    
        
#%% Plot extended MF constraints shown in Fig. 20 of 2002.12778 on the same axes as in that figure.
from preliminaries import constraint_Carr, LN


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
            
    # Power-law exponent to use between 1e15g and 1e16g.
    exponent_PL_upper = 2.0
    # Power-law exponent to use between 1e11g and 1e15g.
    exponent_PL_lower = 2.0
    
    data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower)
    
    plot_unevolved = True
    plot_against_mc = False
    
    fig, ax = plt.subplots(figsize=(7,6))
       
    
    # Delta-function MF constraints from Korwar & Profumo (2023)
    m_delta_KP23_loaded, f_max_KP23_loaded = load_data("2302.04408/2302.04408_MW_diffuse_SPI.csv")
        
    m_delta_extrapolated_upper = np.logspace(15, 16, 11)
    m_delta_extrapolated_lower = np.logspace(11, 15, 41)
    
    f_max_extrapolated_upper = min(f_max_KP23_loaded) * np.power(m_delta_extrapolated_upper / min(m_delta_KP23_loaded), exponent_PL_upper)
    f_max_extrapolated_lower = min(f_max_extrapolated_upper) * np.power(m_delta_extrapolated_lower / min(m_delta_extrapolated_upper), exponent_PL_lower)

    m_delta_Carr21_upper = np.concatenate((m_delta_extrapolated_upper, m_delta_KP23_loaded))
    f_max_upper = np.concatenate((f_max_extrapolated_upper, f_max_KP23_loaded))
    
    f_max_KP23 = np.concatenate((f_max_extrapolated_lower, f_max_extrapolated_upper, f_max_KP23_loaded))
    m_delta_KP23 = np.concatenate((m_delta_extrapolated_lower, m_delta_extrapolated_upper, m_delta_KP23_loaded))
    
    
    # Width of log-normal MF used in Carr et al. (2021)
    sigma_Carr21 = 2
    # Range of characteristic masses for obtaining constraints.
    mc_Carr21 = np.logspace(14, 22, 1000)

    colors=["tab:blue", "tab:orange"]
    
    ax.plot(m_delta_KP23, f_max_KP23, color="tab:gray", label="$\delta$ func.", linewidth=2)
    ax1 = ax.secondary_xaxis('top', functions=(g_to_Solmass, Solmass_to_g)) 
    
    
    # Calculate extended MF constraints obtained for a log-normal with the delta-function MF constraint from Korwar & Profumo (2023).
    for i, sigma in enumerate([sigma_Carr21]):
        
        f_PBH_evolved = constraint_Carr(mc_Carr21, m_delta_KP23, f_max_KP23, LN, [sigma], evolved=True)
        f_PBH_unevolved = constraint_Carr(mc_Carr21, m_delta_KP23, f_max_KP23, LN, [sigma], evolved=False)
        
        if plot_against_mc:
            ax.plot(mc_Carr21, f_PBH_evolved, color=colors[i], dashes=[6, 2], label="LN ($\sigma={:.2f})$".format(sigma))
        else:
            ax.plot(mc_Carr21 * np.exp(-sigma**2), f_PBH_evolved, color=colors[i], dashes=[6, 2], label="LN ($\sigma={:.2f})$".format(sigma))
                        
        # Plot constraint obtained with unevolved MF
        if plot_unevolved:

            if plot_against_mc:
                ax.plot(mc_Carr21, f_PBH_unevolved, color=colors[i], alpha=0.4)
            else:
                ax.plot(mc_Carr21 * np.exp(-sigma**2), f_PBH_unevolved, color=colors[i], alpha=0.4)
            
    # Calculate constraints shown in Fig. 20 of 2002.12778
    m_min = 1e11
    m_max = 1e20
    epsilon = 0.4
    m_star = 5.1e14

    # Calculate delta-function MF constraints, using Eqs. 32-33 of 2002.12778 for beta_prime, and Eq. 57 to convert to f_PBH
    m_delta_Carr21 = 10**np.arange(np.log10(m_min), np.log10(m_max), 0.1)
    f_max_Carr21 = f_PBH_beta_prime(m_delta_Carr21, beta_prime_gamma_rays(m_delta_Carr21))
                
    f_PBH_Carr21 = constraint_Carr(mc_Carr21, m_delta_Carr21, f_max_Carr21, LN, params=[sigma_Carr21], evolved=False)
    
    m_delta_Carr21_loaded, f_max_Carr21_loaded = load_data("./2002.12778/Carr+21_mono_RH.csv")
    mc_Carr21_LN_loaded, fPBH_Carr21_LN_loaded = load_data("./2002.12778/Carr+21_Gamma_ray_LN_RH.csv")
    
    if plot_against_mc:
        ax.set_xlabel("$m_c = m_p\exp(\sigma^2)~[\mathrm{g}]$")
        #ax1.set_xlabel("$m_c~[M_\odot]$")
        
        ax2 = ax.twinx()        
        #ax2.plot(Solmass_to_g(m_delta_Carr21_loaded), f_max_Carr21_loaded, color="k", label="$\delta$ func.")
        ax2.plot(m_delta_Carr21, f_max_Carr21, color="k", label="$\delta$ func.", linestyle="dotted")
        #ax2.plot(Solmass_to_g(mc_Carr21_LN_loaded), fPBH_Carr21_LN_loaded, color="lime", label="LN ($\sigma={:.1f}$)".format(sigma_Carr21))
        ax2.plot(mc_Carr21, f_PBH_Carr21, color="tab:green", label="LN ($\sigma={:.2f}$)".format(sigma_Carr21))
        ax2.legend(title="Carr+ '21", fontsize="x-small", loc=(0.55, 0.02))
        
    else:
        ax.set_xlabel("$m_p~[\mathrm{g}]$")
        #ax1.set_xlabel("$m_p~[M_\odot]$")
        
        ax2 = ax.twinx()        
        #ax2.plot(Solmass_to_g(m_delta_Carr21_loaded), f_max_Carr21_loaded, color="k", label="$\delta$ func.")
        ax2.plot(m_delta_Carr21, f_max_Carr21, color="k", label="$\delta$ func.", linestyle="dotted")
        #ax2.plot(Solmass_to_g(mc_Carr21_LN_loaded) / np.exp(sigma_Carr21**2), fPBH_Carr21_LN_loaded, color="lime", label="LN ($\sigma={:.1f}$)".format(sigma_Carr21))
        ax2.plot(mc_Carr21 * np.exp(-sigma_Carr21**2), f_PBH_Carr21, color="tab:green", label="LN ($\sigma={:.2f}$)".format(sigma_Carr21))
        ax2.legend(title="Carr+ '21", fontsize="x-small", loc=(0.55, 0.02))
            
    ax.set_ylabel("$f_\mathrm{PBH}$")
    
    ax.legend(title="KP' 23", fontsize="x-small", loc=(0.55, 0.35))
    
    for a in [ax, ax2]:
        a.set_xscale("log")
        a.set_yscale("log")
        
        if plot_against_mc:
            a.set_xlim(1e16, 1e20)
        else:
            a.set_xlim(1e16, 3e18)
        a.set_ylim(1e-6, 1)
        
        x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 10)
        a.xaxis.set_major_locator(x_major)
        x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
        a.xaxis.set_minor_locator(x_minor)
        a.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    
    #ax1.set_axis_off()
    ax2.set_axis_off()
    fig.tight_layout()
    
    #%%
    # Plot psi_N, f_max and the integrand in the same figure window
    for sigma_Carr21 in [sigmas_LN[5], 2]:
        m_p = 1e17
        m_c = m_p * np.exp(sigma_Carr21**2) 
        
        fig1, axes = plt.subplots(3, 1, figsize=(5, 14))
        ax1 = axes[0]
        ax2 = axes[1]
        ax3 = axes[2]
        
        ax1.plot(m_delta_KP23, LN(m_delta_KP23, m_c=m_c, sigma=sigma_Carr21) / max(LN(m_delta_KP23, m_c=m_c, sigma=sigma_Carr21)))
        ax1.set_ylabel("$\psi / \psi_\mathrm{max}$")
        ax1.set_ylim(1e-5, 2)
    
        ax2.plot(m_delta_KP23, 1/f_max_KP23, color=(0.5294, 0.3546, 0.7020), label="KP '23")
        ax2.plot(m_delta_Carr21, 1/f_max_Carr21, color="k", label="Carr+ '21")
        ax2.legend(fontsize="x-small")
        ax2.set_ylabel("$1 / f_\mathrm{max}$")
        ax2.set_ylim(1e-2, 1e10)
       
        ax3.plot(m_delta_KP23, LN(m_delta_KP23, m_c=m_c, sigma=sigma_Carr21)/f_max_KP23, color=(0.5294, 0.3546, 0.7020), label="KP '23")
        ax3.plot(m_delta_Carr21, LN(m_delta_Carr21, m_c=m_c, sigma=sigma_Carr21)/f_max_Carr21, color="k", label="Carr+ '21")
        ax3.legend(fontsize="x-small")
        ax3.set_ylabel("$\psi_\mathrm{N}/f_\mathrm{max}~[\mathrm{g}^{-1}]$")
        ax3.set_ylim(1e-20, 1e-8)
    
        for ax in [ax1, ax2, ax3]:
            ax.set_xlabel("$m~[\mathrm{g}]$")
            ax.set_xlim(1e14, 1e17)
            ax.set_xscale("log")
            ax.set_yscale("log")
         
        # Plot the integrand on a linear scale
        fig2, ax4 = plt.subplots(figsize=(6,5))
        ax4.plot(m_delta_KP23, LN(m_delta_KP23, m_c=m_c, sigma=sigma_Carr21)/f_max_KP23, color=(0.5294, 0.3546, 0.7020), label="KP '23")
        ax4.plot(m_delta_Carr21, LN(m_delta_Carr21, m_c=m_c, sigma=sigma_Carr21)/f_max_Carr21, color="k", label="Carr+ '21")
        ax4.legend(fontsize="x-small")
        ax4.set_ylabel("$\psi_\mathrm{N}/f_\mathrm{max}~[\mathrm{g}^{-1}]$")
        ax4.set_xlabel("$m~[\mathrm{g}]$")
        ax4.set_xlim(0, 1e15)
        
        for fig in [fig1, fig2]:
            fig.suptitle("$m_p = {:.1e}".format(m_p) + "~\mathrm{g}$" + ", $\sigma={:.2f}$".format(sigma_Carr21), fontsize="small")
            fig.tight_layout()
        
#%%
    # Plot psi_N, f_max and the integrand in the same figure window (sigma=2 case), using constraints from Auffinger (2022) [2201.01265] Fig. 3 RH panel
    m_delta_A22, f_max_A22 = load_data("./2201.01265/2201.01265_Fig3_EGXB.csv")   

    sigma_A22 = 2
    m_p = 1e16
    m_c = m_p * np.exp(sigma_A22**2) 
    
    fig1, axes = plt.subplots(3, 1, figsize=(5, 14))
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]
    
    from preliminaries import psi_evolved, mass_evolved
    t_0 = 13.8e9 * 365.25 * 86400    # Age of Universe, in seconds
    m_pbh_values_formation = np.concatenate((np.logspace(np.log10(m_star) - 3, np.log10(m_star)), np.arange(m_star, m_star*(1+1e-11), 5e2), np.arange(m_star*(1+1e-11), m_star*(1+1e-6), 1e7), np.logspace(np.log10(m_star*(1+1e-4)), np.log10(max(m_delta_KP23))+4, 1000)))
    m_pbh_values_evolved = mass_evolved(m_pbh_values_formation, t_0)
    psi_initial = LN(m_pbh_values_formation, m_c, sigma_A22)
    psi_evolved = psi_evolved(psi_initial, m_pbh_values_evolved, m_pbh_values_formation)
    psi_evolved_interp_A22 = 10**np.interp(np.log10(m_delta_A22), np.log10(m_pbh_values_evolved), np.log10(psi_evolved))
    psi_evolved_interp_KP23 = 10**np.interp(np.log10(m_delta_KP23), np.log10(m_pbh_values_evolved), np.log10(psi_evolved))
   
    ax1.plot(m_delta_KP23, psi_evolved_interp_KP23)
    ax1.set_ylabel("$\psi_\mathrm{N}~[\mathrm{g}^{-1}]$")

    ax2.plot(m_delta_KP23, 1/f_max_KP23, color=(0.5294, 0.3546, 0.7020), label="KP '23")
    ax2.plot(m_delta_A22, 1/f_max_A22, color="k", label="Auffinger '22")
    ax2.legend(fontsize="x-small")
    ax2.set_ylabel("$1 / f_\mathrm{max}$")
    
    ax3.plot(m_delta_KP23, psi_evolved_interp_KP23/f_max_KP23, color=(0.5294, 0.3546, 0.7020), label="KP '23")
    ax3.plot(m_delta_A22, psi_evolved_interp_A22/f_max_A22, color="k", label="Auffinger '22")
    ax3.legend(fontsize="x-small")
    ax3.set_ylabel("$\psi_\mathrm{N}/f_\mathrm{max}~[\mathrm{g}^{-1}]$")

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel("$m~[\mathrm{g}]$")
        ax.set_xlim(1e14, 1e17)
        ax.set_xscale("log")
        ax.set_yscale("log")
     
    # Plot the integrand on a linear scale
    fig2, ax4 = plt.subplots(figsize=(6,5))
    ax4.plot(m_delta_KP23, psi_evolved_interp_KP23/f_max_KP23, color=(0.5294, 0.3546, 0.7020), label="KP '23")
    ax4.plot(m_delta_A22, psi_evolved_interp_A22/f_max_A22, color="k", label="Auffinger '22")
    ax4.legend(fontsize="x-small")
    ax4.set_ylabel("$\psi_\mathrm{N}/f_\mathrm{max}~[\mathrm{g}^{-1}]$")
    ax4.set_xlim(0, 5e16)
    
    for fig in [fig1, fig2]:
        fig.suptitle("$m_p = {:.1e}".format(m_p) + "~\mathrm{g}$", fontsize="small")
        fig.tight_layout()

        
#%% Test: log-normal plots. Aim is to understand why the extended MF constraints shown in Fig. 20 of 2002.12778 differ so much from the delta-function MF constraints compared to the versions Im using.
    
    # Plot the constraints shown in Fig. 20 of 2002.12778
    m_min = 1e11
    m_max = 1e20
    epsilon = 0.4
    m_star = 5.1e14
    
    m_pbh_values = 10**np.arange(np.log10(m_min), np.log10(m_max), 0.1)
    f_max_values = f_PBH_beta_prime(m_pbh_values, beta_prime_gamma_rays(m_pbh_values))
    #f_max_values /= 2
    mc_values = np.logspace(15, 22, 70)
    
    sigma = 2
    
    fig, ax = plt.subplots(figsize=(8,7))
    ax1 = ax.secondary_xaxis('top', functions=(g_to_Solmass, Solmass_to_g)) 

    f_PBH_values = constraint_Carr(mc_values, m_pbh_values, f_max_values, LN, [sigma], evolved=False)

    m_delta_values_loaded, f_max_loaded = load_data("./2002.12778/Carr+21_mono_RH.csv")
    mc_LN_values_loaded, f_PBH_loaded = load_data("./2002.12778/Carr+21_Gamma_ray_LN_RH.csv")
    
    ax.plot(m_delta_values_loaded * 1.989e33, f_max_loaded, color="tab:grey", label="Delta func.")
    ax.plot(m_pbh_values, f_max_values, color="k", label="Delta func. [repr.]", linestyle="dashed")
    ax.plot(mc_LN_values_loaded * 1.989e33, f_PBH_loaded, color="lime", label="LN ($\sigma={:.1f}$)".format(sigma))
    ax.plot(mc_values, f_PBH_values, color="tab:green", label="LN ($\sigma={:.1f}$) [repr.]".format(sigma), linestyle="dashed")

    ax.set_ylabel("$f_\mathrm{PBH}$")
    ax.set_xlabel("$m_c~[\mathrm{g}]$")
    ax1.set_xlabel("$m_c~[M_\odot]$")
    ax.legend(title="$\sigma$", fontsize="x-small")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-18*1.989e33, 5*1.989e33)
    ax.set_ylim(1e-4, 1)
    fig.tight_layout()    
    
    
    fig, ax = plt.subplots(figsize=(8,7))
    ax1 = ax.secondary_xaxis('top', functions=(g_to_Solmass, Solmass_to_g)) 

    f_PBH_values = constraint_Carr(mc_values, m_pbh_values, f_max_values, LN, [sigma], evolved=False)

    m_delta_values_loaded, f_max_loaded = load_data("./2002.12778/Carr+21_mono_RH.csv")
    mc_LN_values_loaded, f_PBH_loaded = load_data("./2002.12778/Carr+21_Gamma_ray_LN_RH.csv")
    
    ax.plot(m_delta_values_loaded * 1.989e33, f_max_loaded, color="tab:grey", label="Delta func.")
    ax.plot(m_pbh_values, f_max_values, color="k", label="Delta func. [repr.]", linestyle="dashed")
    ax.plot(mc_LN_values_loaded * 1.989e33 * np.exp(-sigma**2), f_PBH_loaded, color="lime", label="LN ($\sigma={:.1f}$)".format(sigma))
    ax.plot(mc_values  * np.exp(-sigma**2), f_PBH_values, color="tab:green", label="LN ($\sigma={:.1f}$) [repr.]".format(sigma), linestyle="dashed")

    ax.set_ylabel("$f_\mathrm{PBH}$")
    ax.set_xlabel("$m_p~[\mathrm{g}]$")
    ax1.set_xlabel("$m_p~[M_\odot]$")
    ax.legend(title="$\sigma$", fontsize="x-small")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-18*1.989e33, 5*1.989e33)
    ax.set_ylim(1e-4, 1)
    fig.tight_layout()    

    
