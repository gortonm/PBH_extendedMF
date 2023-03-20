#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:45:21 2023

@author: ppxmg2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from preliminaries import *

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

#%%%

def constraint_Carr(mc_values, m_mono, f_max, mf, params):
    """
    Calculate constraint on f_PBH for an extended mass function, using the method from 1705.05567.

    Parameters
    ----------
    mc_values : Array-like
        Characteristic PBH masses (m_c for a (skew-)lognormal, m_p for CC3)..
    m_mono : Array-like
        Masses at which constraints for a monochromatic PBH mass function are evaluated..
    f_max : Array-like
        Constraints obtained for a monochromatic mass function..
    mf : Function
        PBH mass function.
    params : Array-like
        Parameters of the PBH mass function.

    Returns
    -------
    f_pbh : Array-like
        Constraints on f_PBH.

    """
    f_pbh = []
    
    for m_c in mc_values:
        integral = np.trapz(mf(m_mono, m_c, *params) / f_max, m_mono)
        if integral == 0:
            f_pbh.append(10)
        else:
            f_pbh.append(1/integral)
            
    return f_pbh


def envelope(constraints):
    """
    Calculate the tightest constraint at a given mass, from a set of 
    constraints.

    Parameters
    ----------
    constraints : Array-like
        Constraints on PBH abundance. All should have the same length and be
        evaluated at the same PBH mass.

    Returns
    -------
    tightest : Array-like
        Tightest constraint, from the constraints given in the input.

    """
    tightest = np.ones(len(constraints[0]))
    
    for i in range(len(constraints[0])):
        
        constraints_values = []
        
        for j in range(len(constraints)):
            constraints_values.append(abs(constraints[j][i]))
        
        tightest[i] = min(constraints_values)
    
    return tightest
        


#%% Test: constant constraint from monochromatic MF
if "__main__" == __name__:
    
    n_pbhs = 1000
    f_max = 1e-3
    
    # Monochromatic MF constraint (constant value of f_max = 1e-3 with PBH mass)
    m_mono_values = np.logspace(14, 21, n_pbhs)
    f_max_values = f_max * np.ones(n_pbhs)
    
    # Extended mass function
    n_mc_values = 100
    mc_values = np.logspace(15, 20, n_mc_values)
    
    # Width of log-normal mass function
    sigma_LN = 0.5
    
    # Estimated constraint from extended mass function.
    f_pbh = constraint_Carr(mc_values, m_mono_values, f_max_values, LN, [sigma_LN])
    
    # Exact constraint from extended mass function.
    f_pbh_exact = f_max * np.ones(n_mc_values)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mc_values, f_pbh, marker="x", linestyle="None")
    ax.plot(mc_values, f_pbh_exact, linestyle="dotted", color="r")
    ax.set_xlabel("$M_c$")
    ax.set_ylabel("$f_\mathrm{PBH}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(f_max/10, f_max*10)
    plt.tight_layout()
    
    
#%% Test: linearly decreasing constraint from monochromatic MF
if "__main__" == __name__:
    
    n_pbhs = 1000
    f_max_0 = 1e-3
    m_mono_0 = 1e17
    
    # Monochromatic MF constraint (constant value of f_max = 1e-3 with PBH mass)
    m_mono_values = np.logspace(15, 20, n_pbhs)
    f_max_values = f_max_0 * np.power(m_mono_values / m_mono_0, -1)
    
    # Extended mass function
    n_mc_values = 100
    mc_values = np.logspace(15, 20, n_mc_values)
    
    # Width of log-normal mass function
    sigma_LN = 0.5
    
    # Estimated constraint from extended mass function.
    f_pbh = constraint_Carr(mc_values, m_mono_values, f_max_values, LN, [sigma_LN])
    
    # Exact constraint from extended mass function.
    f_pbh_exact = np.power(-0.5 * (mc_values / m_mono_0) * (1/f_max_0) * np.exp(sigma_LN**2/2) * ( erf( (sigma_LN**2 - np.log(max(m_mono_values)/mc_values)) / (sigma_LN*np.sqrt(2))) - erf( (sigma_LN**2 - np.log(min(m_mono_values)/mc_values)) / (sigma_LN*np.sqrt(2)))), -1)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mc_values, f_pbh, marker="x", linestyle="None")
    ax.plot(mc_values, f_pbh_exact, linestyle="dotted", color="r")
    ax.set_xlabel("$M_c$")
    ax.set_ylabel("$f_\mathrm{PBH}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.tight_layout()
    
    
#%% Plot results for a monochromatic mass function, obtained using Isatis,
# and compare to the results shown in Fig. 3 of 2201.01265.
if "__main__" == __name__:

    m_pbh_values = np.logspace(11, 21, 101)
    constraints_names_unmodified, f_PBH_Isatis_unmodified = load_results_Isatis(modified=False)
    colors_evap = ["tab:orange", "tab:green", "tab:red", "tab:blue"]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for i in range(len(constraints_names_unmodified)):
        ax.plot(m_pbh_values, f_PBH_Isatis_unmodified[i], label=constraints_names_unmodified[i], color=colors_evap[i])
    
    ax.set_xlim(1e14, 1e18)
    ax.set_ylim(10**(-10), 1)
    ax.set_xlabel("$M_\mathrm{PBH}~[\mathrm{g}]$")
    ax.set_ylabel("$f_\mathrm{PBH}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize="small")
    plt.tight_layout()


#%% Plot results for a monochromatic mass function, obtained using Isatis,
# and compare to the results shown in Fig. 3 of 2201.01265.
# Using the modified version of Isatis.
# Includes test of the envelope() function.

if "__main__" == __name__:
    
    m_pbh_values = np.logspace(11, 21, 1000)
    constraints_names, f_PBH_Isatis = load_results_Isatis(modified=True)
    colors_evap = ["tab:orange", "tab:green", "tab:red", "tab:blue"]
    constraints_names_short = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for i in range(len(constraints_names)):
        ax.plot(m_pbh_values, f_PBH_Isatis[i], label=constraints_names[i], color=colors_evap[i])
    
        # Constraints data for each energy bin of each instrument.
        constraints_mono_file = np.transpose(np.genfromtxt("./Data/fPBH_GC_full_all_bins_%s_monochromatic.txt"%(constraints_names_short[i])))
        ax.plot(m_pbh_values, envelope(constraints_mono_file), marker="x", color=colors_evap[i])
    
    ax.set_xlim(1e14, 1e18)
    ax.set_ylim(10**(-10), 1)
    ax.set_xlabel("$M_\mathrm{PBH}~[\mathrm{g}]$")
    ax.set_ylabel("$f_\mathrm{PBH}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize="small")
    plt.tight_layout()


#%% Plot results for a log-normal mass function, obtained using Isatis,
# and compare to the results shown in Fig. 3 of 2201.01265.
# Using the modified version of Isatis.
if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)

    mc_values = np.logspace(14, 19, 100)
    colors_evap = ["tab:orange", "tab:green", "tab:red", "tab:blue"]
    
    m_mono_values = np.logspace(11, 21, 1000)
    
    for j in range(len(sigmas_LN[:-1])):
        
        # Constraints calculated using Isatis.
        constraints_names, f_PBH_Isatis = load_results_Isatis(mf_string="LN_Delta={:.1f}".format(Deltas[j]), modified=True)    
        
        # Load monochromatic MF constraints calculated using Isatis, to use the method from 1705.05567.
        # Using the envelope of constraints for each instrument for the monochromatic MF constraint.
        constraints_names, f_max = load_results_Isatis(modified=True)
        params_LN = [sigmas_LN[j]]
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        for i in range(len(constraints_names)):
            ax.plot(mc_values, f_PBH_Isatis[i], label=constraints_names[i], color=colors_evap[i])
            
            # Calculate constraint using method from 1705.05567, and plot.
            f_PBH_Carr = constraint_Carr(mc_values, m_mono_values, f_max[i], LN, params_LN)
            ax.plot(mc_values, f_PBH_Carr, marker="x", linestyle="None", color=colors_evap[i])
        
        ax.set_xlim(1e14, 1e18)
        ax.set_ylim(10**(-10), 1)
        ax.set_xlabel("$M_c~[\mathrm{g}]$")
        ax.set_ylabel("$f_\mathrm{PBH}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize="small")
        plt.tight_layout()


#%% Plot results for a log-normal mass function, obtained using Isatis,
# and compare to the results shown in Fig. 3 of 2201.01265.
# Using the modified version of Isatis.
# Uses Isatis constraints calculated using the same range of PBH masses as those from 1705.05567.
if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)

    mc_values = np.logspace(14, 19, 100)
    colors_evap = ["tab:orange", "tab:green", "tab:red", "tab:blue"]
    constraints_names_short = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
    
    m_mono_values = np.logspace(11, 21, 1000)
    
    for j in range(len(sigmas_LN[:-1])):
        
        if j==5:
        
            # Constraints calculated using Isatis.
            constraints_names, f_PBH_Isatis = load_results_Isatis(mf_string="LN_Delta={:.1f}".format(Deltas[j]), modified=True, test_mass_range=True)    
            
            # Load monochromatic MF constraints calculated using Isatis, to use the method from 1705.05567.
            # Using each energy bin per instrument individually for the monochromatic MF constraint, then obtaining the tightest constraint from each instrument using envelope().
            constraints_names, f_max = load_results_Isatis(modified=True)
            params_LN = [sigmas_LN[j]]
            
            fig, ax = plt.subplots(figsize=(8, 8))
            
            for i in range(len(constraints_names)):
                ax.plot(mc_values, f_PBH_Isatis[i], label=constraints_names[i], color=colors_evap[i])
                
                # Calculate constraint using method from 1705.05567.
                
                # Constraints data for each energy bin of each instrument.
                constraints_mono_file = np.transpose(np.genfromtxt("./Data/fPBH_GC_full_all_bins_%s_monochromatic.txt"%(constraints_names_short[i])))
                # Constraints for an extended MF, from each instrument.
                energy_bin_constraints = []
                
                for k in range(len(constraints_mono_file)):
                    
                    # Constraint from a particular energy bin
                    constraint_energy_bin = constraints_mono_file[k]
                    
                    # Calculate constraint on f_PBH from each bin
                    f_PBH_k = constraint_Carr(mc_values, m_mono=m_mono_values, f_max=constraint_energy_bin, mf=LN, params=params_LN)
                    energy_bin_constraints.append(f_PBH_k)
                    
                # Calculate constraint using method from 1705.05567, and plot.
                f_PBH_Carr = envelope(energy_bin_constraints)
                ax.plot(mc_values, f_PBH_Carr, marker="x", linestyle="None", color=colors_evap[i])
            
            ax.set_xlim(1e14, 1e18)
            ax.set_ylim(10**(-10), 1)
            ax.set_xlabel("$M_c~[\mathrm{g}]$")
            ax.set_ylabel("$f_\mathrm{PBH}$")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend(fontsize="small")
            plt.tight_layout()
            
#%% Convergence tests: compare results for f_PBH obtained using different numbers of
# PBHs and mass ranges.

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)

    mc_values = np.logspace(14, 19, 100)
    colors_evap = ["tab:orange", "tab:green", "tab:red", "tab:blue"]
    constraints_names_short = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]

    # Cutoff in the PBH mass function, compared to the peak mass
    cutoffs = [1e-3, 1e-5]
    # PBH mass spacing, in log10(PBH mass / grams)
    dm_values = [1e-4]
    
    log_normal = True
    SLN_bool = False
    CC3_bool = False
    
    plot_constraint = True

    # If True, use cutoff in terms of the mass function scaled to its peak value.
    MF_cutoff = True
    # If True, use cutoff in terms of the integrand appearing in Galactic Centre photon constraints.
    integrand_cutoff = False
    # If True, use cutoff in terms of the integrand appearing in Galactic Centre photon constraints, with the mass function evolved to the present day.
    integrand_cutoff_present = False

    for i in range(len(Deltas[0:2])):
        
        # Find the most accurate constraint:
        if log_normal:
            mf_string = "LN_Delta={:.1f}_dm={:.0f}".format(Deltas[i], -np.log10(min(dm_values)))
        elif SLN_bool:
            mf_string = "SLN_Delta={:.1f}_dm={:.0f}".format(Deltas[i], -np.log10(min(dm_values)))
        elif CC3_bool:
            mf_string = "CC3_Delta={:.1f}_dm={:.0f}".format(Deltas[i], -np.log10(min(dm_values)))
        
        if MF_cutoff:
            mf_string += "_MF_c={:.0f}".format(-np.log10(min(cutoffs)))
        elif integrand_cutoff:
            mf_string += "_integrand_c={:.0f}".format(-np.log10(min(cutoffs)))
        elif integrand_cutoff_present:
            mf_string += "_integrand2_c={:.0f}".format(-np.log10(min(cutoffs)))

        # Constraints calculated using Isatis.
        constraints_names, f_PBH_Isatis_benchmark = load_results_Isatis(mf_string, modified=True, test_mass_range=False)    
        f_PBH_benchmark_envelope = envelope(f_PBH_Isatis_benchmark)
        
        fig, ax = plt.subplots(figsize=(8, 8))

        for cutoff in cutoffs:
                                                 
            for delta_log_m in dm_values:
                
                if log_normal:
                    mf_string = "LN_Delta={:.1f}_dm={:.0f}".format(Deltas[i], -np.log10(delta_log_m))
                elif SLN_bool:
                    mf_string = "SLN_Delta={:.1f}_dm={:.0f}".format(Deltas[i], -np.log10(delta_log_m))
                elif CC3_bool:
                    mf_string = "CC3_Delta={:.1f}_dm={:.0f}".format(Deltas[i], -np.log10(delta_log_m))
                    
                # Indicates which range of masses are being used (for convergence tests).
                if MF_cutoff:
                    mf_string += "_MF_c={:.0f}".format(-np.log10(cutoff))
                elif integrand_cutoff:
                    mf_string += "_integrand_c={:.0f}".format(-np.log10(cutoff))
                elif integrand_cutoff_present:
                    mf_string += "_integrand2_c={:.0f}".format(-np.log10(cutoff))
    
                # Constraints calculated using Isatis.
                constraints_names, f_PBH_Isatis = load_results_Isatis(mf_string, modified=True, test_mass_range=False)  
                # Envelope of constraints.
                f_PBH_envelope = envelope(f_PBH_Isatis)
                # Fractional difference from benchmark constraint.
                frac_diff = abs(f_PBH_envelope/f_PBH_benchmark_envelope - 1)
                
                ax.plot(mc_values, frac_diff, marker="x", label="Cutoff={:.0e}, Spacing $\delta \log_{10} m$={:.0e}".format(cutoff, delta_log_m))
                
                if plot_constraint:
                    fig1, ax1 = plt.subplots(figsize=(8, 8))
            
                    for j in range(len(constraints_names)):
                        ax1.plot(mc_values, f_PBH_Isatis[j], label=constraints_names[j], color=colors_evap[j])
                    
                    ax1.plot(mc_values, f_PBH_envelope, color="k")
                    ax1.set_xlim(1e14, 1e18)
                    ax1.set_ylim(10**(-10), 1)
                    ax1.set_xlabel("$M_c~[\mathrm{g}]$")
                    ax1.set_ylabel("$f_\mathrm{PBH}$")
                    ax1.set_xscale("log")
                    ax1.set_yscale("log")
                    ax1.legend(fontsize="small")
                    plt.tight_layout()

            