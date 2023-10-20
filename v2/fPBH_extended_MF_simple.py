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


if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)

    # Peak PBH masses
    mp_values = np.logspace(15, 17, 30)
    
    # Characteristic masses for the SLN MF
    mc_SLN = np.logspace(15, 19, 50)

    # Functional form of f_max
    PL_exp = 4
    m_pbh_values = np.logspace(11, 20, 1000)
    f_max = (1 / max(mp_values**PL_exp)) * m_pbh_values**PL_exp
        
    for i in range(len(Deltas)):
        
        fig, ax = plt.subplots(figsize=(7,7))
        ax.plot(m_pbh_values, f_max, color="tab:grey", label="Delta function", linewidth=2)
 
        fig1, ax1 = plt.subplots(figsize=(7,7))
 
        # Log-normal MF constraint
        params_LN = [sigmas_LN[i]]
        mc_values_LN = mp_values * np.exp(sigmas_LN[i]**2)
        f_PBH_values_LN = constraint_Carr(mc_values_LN, m_delta=m_pbh_values, f_max=f_max, psi_initial=LN, params=params_LN, evolved=False)
        frac_diff_LN = frac_diff(y1=f_PBH_values_LN, y2=f_max, x1=mp_values, x2=m_pbh_values)
        
        # SLN MF
        params_SLN = [sigmas_SLN[i], alphas_SLN[i]]
        mp_values_SLN = [m_max_SLN(m_c, sigmas_SLN[i], alphas_SLN[i]) for m_c in mc_SLN]
        f_PBH_values_SLN = constraint_Carr(mc_SLN, m_delta=m_pbh_values, f_max=f_max, psi_initial=SLN, params=params_SLN, evolved=False)
        frac_diff_SLN = frac_diff(y1=f_PBH_values_SLN, y2=f_max, x1=mp_values_SLN, x2=m_pbh_values)
       
        # CC3 MF
        params_CC3 = [alphas_CC3[i], betas[i]]
        f_PBH_values_CC3 = constraint_Carr(mp_values, m_delta=m_pbh_values, f_max=f_max, psi_initial=CC3, params=params_CC3, evolved=False)
        frac_diff_CC3 = frac_diff(y1=f_PBH_values_CC3, y2=f_max, x1=mp_values, x2=m_pbh_values)
       
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
    
        ax1.plot(mp_values, np.abs(frac_diff_LN), color="r", linestyle="dashed", label="LN")
        ax1.plot(mp_values_SLN, np.abs(frac_diff_SLN), color="b", linestyle="dashed", label="SLN")
        ax1.plot(mp_values, np.abs(frac_diff_CC3), color="g", linestyle="dashed", label="CC3")  
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_ylabel(r"$|f_\mathrm{PBH} / f_\mathrm{max}(m_p) - 1|$")
        ax1.set_xlabel("$m_p~[\mathrm{g}]$")
        ax1.legend(fontsize="small")
        fig1.suptitle("$\Delta={:.1f}$".format(Deltas[i]) + ", $f_\mathrm{max}(m) " + "\propto m^{:.0f}$".format(PL_exp))
        fig1.tight_layout()

#%%
from scipy.special import gamma

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)

    for i in range(len(Deltas)):
        
        fig, ax = plt.subplots(figsize=(6,6))
        alpha = alphas_CC3[i]
        beta = betas[i]
        
        root_values = []
        PL_exp_values = np.linspace(-5, 5, 50)
        
        for PL_exp in PL_exp_values:
            root_values.append(np.power(gamma(beta/alpha), PL_exp/beta) * gamma((alpha+1)/beta) - gamma((alpha+1-PL_exp) / beta))
            
        ax.plot(PL_exp_values, root_values)
        ax.set_xlabel("$n$")
        fig.suptitle("$\Delta={:.1f}$".format(Deltas[i]))
        fig.tight_layout()
     
#%%
from preliminaries import PL_MF as power_law_mf_v2

def power_law_mf(m_pbh_values, m_min, m_max, gamma):
    """
    Power-law mass function.

    Parameters
    ----------
    m_pbh_values : Array-like
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
    mf_values = []
    
    for m in m_pbh_values:
        if m < m_min or m > m_max:
            mf_values.append(0)
        else:
            mf_values.append(np.power(m, gamma-1) * gamma / (np.power(m_max, gamma) - np.power(m_min, gamma)))
    
    return np.array(mf_values)


def fPBH_power_law_analytic(m_min, m_max, gamma, n, k):
    """
    Analytic result for f_PBH for a power-law mass function with exponent
    gamma, defined between masses m_min and m_max, for a delta-function MF
    constraint f_max = k * m^n (m = PBH mass).

    Parameters
    ----------
    m_min : Float
        Minimum PBH mass at which the power-law MF is defined.
    m_max : Float
        Maximum PBH mass at which the power-law MF is defined.
    gamma : Float
        Exponent of the power-law MF.
    n : Float
        Exponent (in the mass) of the delta-function MF constraint.
    k : Float
        Normalisation for the delta-function MF constraint.

    Returns
    -------
    Float
        Extended MF constraint for a power-law PBH MF where the delta-function MF constraint is a power law in the PBH mass.

    """
    return k * ((gamma - n) / gamma) * (np.power(m_max, gamma) - np.power(m_min, gamma)) / (np.power(m_max, gamma-n) - np.power(m_min, gamma-n))


if "__main__" == __name__:

    m_min_values = np.logspace(13, 18, 100)
    m_max = 1e25
    gamma = -1
    
    # Functional form of f_max
    PL_exp = -2
    m_pbh_values = np.logspace(11, 20, 1000)
    f_max = (1 / max(m_pbh_values**PL_exp)) * m_pbh_values**PL_exp
    
    params_PL = [m_max, gamma]
    f_PBH_values_PL = constraint_Carr(m_min_values, m_pbh_values, f_max, power_law_mf, params_PL, evolved=False)
    f_PBH_values_PL_v2 = constraint_Carr(m_min_values, m_pbh_values, f_max, power_law_mf_v2, params_PL, evolved=False)
   
    f_PBH_values_PL_analytic = fPBH_power_law_analytic(m_min_values, m_max, gamma, PL_exp, (1 / max(m_pbh_values**PL_exp)))
    
    # Power-law mass function
    fig, ax = plt.subplots(figsize=(7,7))
    ax.plot(m_pbh_values, f_max, color="tab:grey", label="Delta function", linewidth=2)

    ax.plot(m_min_values, f_PBH_values_PL, color="tab:blue", linestyle="dashed", label="PL")
    ax.plot(m_min_values, f_PBH_values_PL_v2, color="tab:blue", marker="x", linestyle="None", label="PL [preliminaries.py]", alpha=0.5)
    ax.plot(m_min_values, f_PBH_values_PL_analytic, color="k", linestyle="dotted", label="PL [analytic]")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(min(m_min_values), max(m_min_values))
    ax.set_ylim(min(f_PBH_values_PL), max(f_PBH_values_PL))
    ax.set_xlabel("$m_\mathrm{min}~[\mathrm{g}]$")
    ax.set_ylabel("$f_\mathrm{PBH}$")
    ax.legend(fontsize="x-small")
    fig.suptitle("$m_\mathrm{max}=" + "{:.1e}".format(m_max) + "~[\mathrm{g}]$" + ", $f_\mathrm{max}(m) " + "\propto m^{:.0f}$".format(PL_exp) + ", $\gamma={:.1f}$".format(gamma), fontsize="small")
    fig.tight_layout()
