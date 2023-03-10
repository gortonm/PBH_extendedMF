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