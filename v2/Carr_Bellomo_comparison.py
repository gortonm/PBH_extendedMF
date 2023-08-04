#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 13:54:26 2023

@author: ppxmg2
"""
# Script to compare results obtained using the methods from Carr et al. (2017) [1705.05567] and Bellomo et al. (2018) [1709.07467]
import numpy as np
from preliminaries import load_data, LN, SLN, CC3, constraint_Carr, load_results_Isatis, envelope
import matplotlib.pyplot as plt
import matplotlib as mpl

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

m_Pl = 2.176e-5    # Planck mass, in grams
t_Pl = 5.391e-44    # Planck time, in seconds
t_0 = 13.8e9 * 365.25 * 86400    # Age of Universe, in seconds
m_star = 7.473420349255e+14    # Formation mass of a PBH with a lifetimt equal to the age of the Universe, in grams.


#%%

def findroot(f, a, b, args, tolerance=1e-5, n_max=10000):
    n = 1
    while n <= n_max:
        #c = (a + b) / 2
        c = 10**((np.log10(a) + np.log10(b)) / 2)
        #if f(c, *args) == 0 or (b - a) / 2 < tolerance:
        if f(c, *args) == 0 or (np.log10(b) - np.log10(a)) / 2 < tolerance:
            print(n)
            return c
            break
        n += 1
        
        # set new interval
        if np.sign(f(c, *args)) == np.sign(f(a, *args)):
            a = c
        else:
            b = c
    print("Method failed")


#%% Tests of the method from Bellomo et al. (2018) [1709.07467].

def g_test_CMB(m, alpha):
    """
    Function encoding mass-dependence of CMB constraints
    """
    return np.power(m, 2+alpha)


def m_eq_CMB(mu, alpha, sigma):
    """
    Equivalent mass for CMB constraints
    """
    return mu * np.exp( (2+alpha) * sigma**2  / 2)
    
    
def g_integral_CMB(mu, sigma, m_min = 1e10, m_max = 1e20, n_steps=10000):
    """
    Integral used when calculating the equivalent mass (RHS of Eq. 2.5 or 2.6 of Bellomo et al. (2018))
    """
    
    m_pbh_values = np.logspace(np.log10(m_min), np.log10(m_max), n_steps)
    
    return np.trapz(LN(m_pbh_values, mu, sigma) * g_test_CMB(m_pbh_values, alpha), m_pbh_values)


def meq_finder_CMB(m_eq, mu, sigma, alpha, m_min = 1e10, m_max = 1e20, n_steps=10000):
    """
    Function to solve for the equivalent mass.
    """
    return g_test_CMB(m_eq, alpha) - g_integral_CMB(mu, sigma, m_min, m_max, n_steps)


if "__main__" == __name__:

    m_min = 1e10
    m_max = 1e20
    
    m_c = 1e15
    sigma = 0.5
    alpha = -4
    
    # Find M_eq using the exact result
    print("Exact M_eq = {:.8e} g".format(m_eq_CMB(m_c, alpha, sigma)))
    
    # Estimate M_eq numerically
    #print("Numeric M_eq = {:.8e} g".format(fsolve(meq_finder_CMB, m_c, args=(m_c, sigma, alpha))[0]))
    print("Numeric M_eq = {:.8e} g".format(findroot(meq_finder_CMB, m_min, m_max, args=(m_c, sigma, alpha))))

#%% Constraints from COMPTEL, INTEGRAL, EGRET and Fermi-LAT. Approximate results obtained by using f_max as the constraint from each instrument, rather than the minimum over each energy bin.
# Obtained using the method from Bellomo et al. (2018) [1709.07467].

use_LN = False
use_CC3 = True

def m_eq_func(m, f_max_allpositive, m_delta_allpositive, m_c, *args):
    if use_LN:
        return np.interp(m, m_delta_allpositive, 1/f_max_allpositive) - np.trapz(LN(m_delta_allpositive, m_c, sigma) / f_max_allpositive, m_delta_allpositive)
    elif use_CC3:
        return np.interp(m, m_delta_allpositive, 1/f_max_allpositive) - np.trapz(CC3(m_delta_allpositive, m_c, alpha, beta) / f_max_allpositive, m_delta_allpositive)

if "__main__" == __name__:
    fig, ax = plt.subplots(figsize=(8,8))
    
    if use_LN:
        sigma = 1.84859
        
    elif use_CC3:
        alpha = 13.9
        beta = 0.0206
                
    m_delta_values_loaded = np.logspace(11, 21, 1000)
    constraints_names, f_max_Isatis = load_results_Isatis(modified=True)
    colors_evap = ["tab:orange", "tab:green", "tab:red", "tab:blue"]
    mc_values = np.logspace(14, 20, 120)
    
    all_fmax = True

    for i in range(len(constraints_names)):
        
        f_max_i = np.array(f_max_Isatis[i])
        
        if all_fmax:
            f_max_allpositive = f_max_i[f_max_i > 0.]
            m_delta_allpositive = m_delta_values_loaded[f_max_i > 0.]
        
        else:
            f_max_allpositive_intermed = f_max_i[f_max_i > 0.]
            m_delta_allpositive_intermed = m_delta_values_loaded[f_max_i > 0.]
            
            f_max_allpositive = f_max_allpositive_intermed[f_max_allpositive_intermed <= 1.]
            m_delta_allpositive = m_delta_allpositive_intermed[f_max_allpositive_intermed <= 1.]
                      
        f_PBH_Bellomo = []
                
        for m_c in mc_values:
            if use_LN:
                m_eq_estimate = findroot(m_eq_func, min(m_delta_allpositive), max(m_delta_allpositive), args=(f_max_allpositive, m_delta_allpositive, m_c, sigma))
            elif use_CC3:
                m_eq_estimate = findroot(m_eq_func, min(m_delta_allpositive), max(m_delta_allpositive), args=(f_max_allpositive, m_delta_allpositive, m_c, alpha, beta))

            f_PBH_Bellomo.append(np.interp(m_eq_estimate, m_delta_allpositive, f_max_allpositive))
           
        ax.plot(mc_values, f_PBH_Bellomo, color=colors_evap[i])
        
        if use_LN:
            f_PBH_Carr = constraint_Carr(mc_values, m_delta_allpositive, f_max_allpositive, LN, params=[sigma], evolved=False)
        elif use_CC3:
            f_PBH_Carr = constraint_Carr(mc_values, m_delta_allpositive, f_max_allpositive, CC3, params=[alpha, beta], evolved=False)
            
        ax.plot(mc_values, f_PBH_Carr, color=colors_evap[i], linestyle="None", marker="x")
    
    ax.plot(0,0, color="k", label="Bellomo et al. (2018) method")
    ax.plot(0,0, color="k", label="Carr et al. (2017) method", marker="x", linestyle="None")

    ax.set_xlabel("$m_c~[\mathrm{g}]$")                
    ax.set_ylabel("$f_\mathrm{PBH}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    if all_fmax:
        if use_LN:
            ax.set_title("No power law extrapolation \n Unevolved LN ($\Delta=5$)", fontsize="small")
        elif use_CC3:
            ax.set_title("No power law extrapolation \n Unevolved CC3 ($\Delta=5$)", fontsize="small")
    else:
        if use_LN:
            ax.set_title("Power law extrapolation \n Unevolved LN ($\Delta=5$)" + ", mass range only where $f_\mathrm{max} < 1$", fontsize="small")
        elif use_CC3:
            ax.set_title("Power law extrapolation \n Unevolved CC3 ($\Delta=5$)" + ", mass range only where $f_\mathrm{max} < 1$", fontsize="small")
    ax.legend(fontsize="small")
    fig.tight_layout()
    
    if use_LN:
        ax.set_xlim(1e14, 1e19)
        ax.set_ylim(1e-12, 1)
        
    elif use_CC3:
        ax.set_xlim(1e14, 1e18)
        ax.set_ylim(1e-10, 1)
        
    fig.tight_layout()
    

#%% Constraints from COMPTEL, INTEGRAL, EGRET and Fermi-LAT. Approximate results obtained by using f_max as the constraint from each instrument, rather than the minimum over each energy bin.
# Obtained using the method from Bellomo et al. (2018) [1709.07467].
# Extrapolate using a power-law in g(M) to M < 1e11g

use_LN = False
use_CC3 = True

def m_eq_func(m, f_max_i_input, m_delta_input, m_c, sigma):
    if use_LN:
        return np.interp(m, m_delta_input, 1/f_max_i_input) - np.trapz(LN(m_delta_input, m_c, sigma) / f_max_i_input, m_delta_input)
    elif use_CC3:
        return np.interp(m, m_delta_input, 1/f_max_i_input) - np.trapz(CC3(m_delta_input, m_c, alpha, beta) / f_max_i_input, m_delta_input)       

if "__main__" == __name__:
    fig, ax = plt.subplots(figsize=(8,8))
    
    if use_LN:
        sigma = 1.84859
        
    elif use_CC3:
        alpha = 13.9
        beta = 0.0206
                
    m_delta_values_loaded = np.logspace(11, 21, 1000)
    constraints_names, f_max_Isatis = load_results_Isatis(modified=True)
    colors_evap = ["tab:orange", "tab:green", "tab:red", "tab:blue"]
    mc_values = np.logspace(14, 20, 120)
    
    # Power-law exponent to use
    exponent_PL_lower = 2.0
    m_delta_extrapolated = np.logspace(11, 13, 21)
    
    all_fmax = True

    for i in range(len(constraints_names)):
        
        f_max_i = np.array(f_max_Isatis[i])
        
        f_max_allpositive_loaded = []
        m_delta_allpositive_loaded = []

        if all_fmax:
            f_max_allpositive_loaded = f_max_i[f_max_i > 0.]
            m_delta_allpositive_loaded = m_delta_values_loaded[f_max_i > 0.]
        else:
            f_max_allpositive_intermed = f_max_i[f_max_i > 0.]
            m_delta_allpositive_intermed = m_delta_values_loaded[f_max_i > 0.]
            
            f_max_allpositive_loaded = f_max_allpositive_intermed[f_max_allpositive_intermed <= 1.]
            m_delta_allpositive_loaded = m_delta_allpositive_intermed[f_max_allpositive_intermed <= 1.]
                    
        # Truncate arrays of loaded masses and f_max values to only include the mass range M > 1e13g
        m_delta_allpositive_loaded_truncated = m_delta_allpositive_loaded[m_delta_allpositive_loaded > 1e13]
        f_max_allpositive_loaded_truncated = f_max_allpositive_loaded[m_delta_allpositive_loaded > 1e13]
        
        # Estimate f_max using a power-law below 1e13g
        f_max_extrapolated = f_max_allpositive_loaded_truncated[0] * np.power(m_delta_extrapolated/1e13, exponent_PL_lower)
        
        # Combine arrays for full f_max and PBH masses
        f_max_i_input = np.concatenate((f_max_extrapolated, f_max_allpositive_loaded_truncated))
        m_delta_input = np.concatenate((m_delta_extrapolated, m_delta_allpositive_loaded_truncated))
        
        f_PBH_Bellomo = []
                
        for m_c in mc_values:
            if use_LN:
                m_eq_estimate = findroot(m_eq_func, min(m_delta_input), max(m_delta_input), args=(f_max_i_input, m_delta_input, m_c, sigma))
            elif use_CC3:
                m_eq_estimate = findroot(m_eq_func, min(m_delta_input), max(m_delta_input), args=(f_max_i_input, m_delta_input, m_c, alpha, beta))
               
            f_PBH_Bellomo.append(np.interp(m_eq_estimate, m_delta_input, f_max_i_input))
           
        ax.plot(mc_values, f_PBH_Bellomo, color=colors_evap[i])
        
        if use_LN:
            f_PBH_Carr = constraint_Carr(mc_values, m_delta_input, f_max_i_input, LN, params=[sigma], evolved=False)
        elif use_CC3:
            f_PBH_Carr = constraint_Carr(mc_values, m_delta_input, f_max_i_input, LN, params=[alpha, beta], evolved=False)
            
        ax.plot(mc_values, f_PBH_Carr, color=colors_evap[i], linestyle="None", marker="x")
    
    ax.plot(0,0, color="k", label="Bellomo et al. (2018) method")
    ax.plot(0,0, color="k", label="Carr et al. (2017) method", marker="x", linestyle="None")

    ax.set_xlabel("$m_c~[\mathrm{g}]$")                
    ax.set_ylabel("$f_\mathrm{PBH}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    if all_fmax:
        if use_LN:
            ax.set_title("No power law extrapolation \n Unevolved LN ($\Delta=5$)", fontsize="small")
        elif use_CC3:
            ax.set_title("No power law extrapolation \n Unevolved CC3 ($\Delta=5$)", fontsize="small")
    else:
        if use_LN:
            ax.set_title("Power law extrapolation \n Unevolved LN ($\Delta=5$)" + ", mass range only where $f_\mathrm{max} < 1$", fontsize="small")
        elif use_CC3:
            ax.set_title("Power law extrapolation \n Unevolved CC3 ($\Delta=5$)" + ", mass range only where $f_\mathrm{max} < 1$", fontsize="small")
    ax.legend(fontsize="small")
    fig.tight_layout()
    
    if use_LN:
        ax.set_xlim(1e14, 1e19)
        ax.set_ylim(1e-12, 1)
        
    elif use_CC3:
        ax.set_xlim(1e14, 1e18)
        ax.set_ylim(1e-10, 1)
        
    fig.tight_layout()