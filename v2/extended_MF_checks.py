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
        PBH mass function..
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

    m_pbh_values = np.logspace(11, 21, 1000)
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
# Include test of the envelope() function

if "__main__" == __name__:
    
    m_pbh_values = np.logspace(11, 21, 1000)
    constraints_names, f_PBH_Isatis = load_results_Isatis(modified=True)
    colors_evap = ["tab:orange", "tab:green", "tab:red", "tab:blue"]
    constraints_names_short = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for i in range(len(constraints_names)):
        ax.plot(m_pbh_values, f_PBH_Isatis[i], label=constraints_names[i], color=colors_evap[i])
    
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
        