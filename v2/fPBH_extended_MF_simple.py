#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 19:19:55 2023

@author: ppxmg2
"""
# Calculate the extended MF constraints f_PBH for simplified forms of the delta-function MF constraint f_max.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from preliminaries import constraint_Carr, LN, SLN, CC3, m_max_SLN

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

def power_law_mf(m, m_min, m_max, gamma):
    """
    Power-law mass function.

    Parameters
    ----------
    m : Array-like
        PBH mass.
    m_min : Float
        Minimum mass for which the power-law mass function is defined.
    m_max : Float
        Maximum mass for which the power-law mass function is defined.
    gamma : Float
        Exponent (gamma=-1/2 for PBHs formed during radiation-dominated epoch).

    Returns
    -------
    Array-like
        Value of the power-law mass function.

    """
    return np.power(m, gamma-1) * gamma / (np.power(m_max, gamma) - np.power(m_min, gamma))


if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)

    # Peak PBH masses
    mp_values = np.logspace(14, 17, 30)
    
    # Characteristic masses for the SLN MF
    mc_SLN = np.logspace(14, 19, 50)

    # Functional form of f_max
    PL_exp = -3
    m_pbh_values = np.logspace(11, 20, 1000)
    f_max = (1 / max(mp_values**PL_exp)) * m_pbh_values**PL_exp
        
    for i in range(len(Deltas)):
        
        fig, ax = plt.subplots(figsize=(7,7))
        ax.plot(m_pbh_values, f_max, color="tab:grey", label="Delta function", linewidth=2)
        
        # Log-normal MF constraint
        params_LN = [sigmas_LN[i]]
        mc_values_LN = mp_values * np.exp(sigmas_LN[i]**2)
        f_PBH_values_LN = constraint_Carr(mc_values_LN, m_delta=m_pbh_values, f_max=f_max, psi_initial=LN, params=params_LN, evolved=False)
    
        # SLN MF
        params_SLN = [sigmas_SLN[i], alphas_SLN[i]]
        mp_values_SLN = [m_max_SLN(m_c, sigmas_SLN[i], alphas_SLN[i]) for m_c in mc_SLN]
        f_PBH_values_SLN = constraint_Carr(mc_SLN, m_delta=m_pbh_values, f_max=f_max, psi_initial=SLN, params=params_SLN, evolved=False)
        
        # CC3 MF
        params_CC3 = [alphas_CC3[i], betas[i]]
        f_PBH_values_CC3 = constraint_Carr(mp_values, m_delta=m_pbh_values, f_max=f_max, psi_initial=CC3, params=params_CC3, evolved=False)
        
        ax.plot(mp_values, f_PBH_values_LN, color="r", linestyle="dashed", label="LN")
        ax.plot(mp_values_SLN, f_PBH_values_SLN, color="b", linestyle="dashed", label="SLN")
        ax.plot(mp_values, f_PBH_values_CC3, color="g", linestyle="dashed", label="CC3")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(min(mp_values), max(mp_values))
        ax.set_ylim(min(f_PBH_values_CC3), max(f_PBH_values_CC3))
        ax.set_xlabel("$m_p~[\mathrm{g}]$")
        ax.set_ylabel("$f_\mathrm{PBH}$")
        ax.legend(fontsize="small")
        fig.suptitle("$\Delta={:.1f}$".format(Deltas[i]) + ", $f_\mathrm{max}(m) " + "\propto m^{:.0f}$".format(PL_exp))
        fig.tight_layout()
        
#%%          
    m_min_values = np.logspace(14, 17, 30)
    m_max = 1e25
    gamma = -1/2
    
    # Functional form of f_max
    PL_exp = 1
    m_pbh_values = np.logspace(11, 20, 1000)
    f_max = (1 / max(m_pbh_values**PL_exp)) * m_pbh_values**PL_exp
    
    params_PL = [m_max, gamma]
    f_PBH_values_PL = constraint_Carr(m_min_values, m_pbh_values, f_max, power_law_mf, params_PL, evolved=False)
    
    # Power-law mass function
    fig, ax = plt.subplots(figsize=(7,7))
    ax.plot(m_pbh_values, f_max, color="tab:grey", label="Delta function", linewidth=2)

    ax.plot(mp_values, f_PBH_values_PL, color="tab:blue", linestyle="dashed", label="PL")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(min(m_min_values), max(m_min_values))
    ax.set_ylim(min(f_PBH_values_PL), max(f_PBH_values_PL))
    ax.set_xlabel("$m_\mathrm{min}~[\mathrm{g}]$")
    ax.set_ylabel("$f_\mathrm{PBH}$")
    ax.legend(fontsize="small")
    fig.suptitle("$m_\mathrm{max}=" + "{:.1e}".format(m_max) + "~[\mathrm{g}]$" + ", $f_\mathrm{max}(m) " + "\propto m^{:.0f}$".format(PL_exp))
    fig.tight_layout()
