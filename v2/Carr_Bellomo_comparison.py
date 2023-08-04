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


#%% Constraints from COMPTEL, INTEGRAL, EGRET and Fermi-LAT. Approximate results obtained by using f_max as the constraint from each instrument, rather than the minimum over each energy bin.
# Obtained using the method from Carr et al. (2017) [1705.05567].

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
        
    # Boolean determines whether to use the full mass range where f_max is known or only the range where |f_max| < 1
    all_fmax = True
        
    m_delta_values_loaded = np.logspace(11, 21, 1000)
    constraints_names, f_max_Isatis = load_results_Isatis(modified=True)
    colors_evap = ["tab:orange", "tab:green", "tab:red", "tab:blue"]
    constraints_names_short = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
    
    mc_values = np.logspace(14, 20, 120)
    
    t = t_0
    if all_fmax:
        data_folder = "./Data-tests/Carr_Bellomo_comparison/full_m_range"
    else:
        data_folder = "./Data-tests/Carr_Bellomo_comparison/truncated_m_range"
        
    # Power-law exponent to uses
    exponent_PL_lower = 2.0
    m_delta_extrapolated = np.logspace(11, 13, 21)
    data_folder += "/PL_exp_{:.0f}/".format(exponent_PL_lower)

    sigma = 1.84859
    evolved = False
        
    for i in range(len(constraints_names)):
        f_max_allpositive = []
        m_delta_loaded_allpositive = []

        if all_fmax:
            for k, f_max in enumerate(f_max_Isatis[i]):
                if f_max > 0:
                    f_max_allpositive.append(f_max)
                    m_delta_loaded_allpositive.append(m_delta_values_loaded[k])
        else:
            for k, f_max in enumerate(f_max_Isatis[i]):
                if 0 < f_max <= 1:
                    f_max_allpositive.append(f_max)
                    m_delta_loaded_allpositive.append(m_delta_values_loaded[k])
                    
        m_max = max(m_delta_loaded_allpositive)
        print("m_max = {:.8e} g".format(m_max))
        
        # Extrapolate f_max at masses below 1e13g using a power-law
        f_max_loaded_truncated = np.array(f_max_allpositive)[np.array(m_delta_loaded_allpositive) > 1e13]
        m_delta_loaded_truncated = np.array(m_delta_loaded_allpositive)[np.array(m_delta_loaded_allpositive) > 1e13]
        f_max_extrapolated = f_max_loaded_truncated[0] * np.power(m_delta_extrapolated / 1e13, exponent_PL_lower)
        f_max_i = np.concatenate((f_max_extrapolated, f_max_loaded_truncated))
        m_delta_values = np.concatenate((m_delta_extrapolated, m_delta_loaded_truncated))
                            
        f_PBH_i_LN = constraint_Carr(mc_values, m_delta_values, f_max_i, LN, [sigma], evolved, t)      
        data_filename_LN = data_folder + "LN_GC_%s" % constraints_names_short[i] + "_Carrs_sigma={:.1f}_approx.txt".format(sigma)
        np.savetxt(data_filename_LN, [mc_values, f_PBH_i_LN], delimiter="\t")


#%% Tests of the method from Bellomo et al. (2018) [1709.07467].
from scipy.optimize import fsolve


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
    print("Numeric M_eq = {:.8e} g".format(fsolve(meq_finder_CMB, m_c, args=(m_c, sigma, alpha))[0]))


#%% Constraints from COMPTEL, INTEGRAL, EGRET and Fermi-LAT. Approximate results obtained by using f_max as the constraint from each instrument, rather than the minimum over each energy bin.
# Obtained using the method from Bellomo et al. (2018) [1709.07467].

from scipy.optimize import fsolve

def m_eq_func(m, f_max_allpositive, m_delta_allpositive, m_c, sigma):
    return np.interp(m, m_delta_allpositive, 1/f_max_allpositive) - np.trapz(LN(m_delta_allpositive, m_c, sigma) / f_max_allpositive, m_delta_allpositive)
    #return 10**np.interp(np.log10(m), np.log10(m_delta_allpositive), np.log10(1/f_max_allpositive)) - np.trapz(LN(m_delta_allpositive, m_c, sigma) / f_max_allpositive, m_delta_allpositive)

if "__main__" == __name__:
    fig, ax = plt.subplots(figsize=(8,8))
    sigma = 1.84859
                
    m_delta_values_loaded = np.logspace(11, 21, 1000)
    constraints_names, f_max_Isatis = load_results_Isatis(modified=True)
    colors_evap = ["tab:orange", "tab:green", "tab:red", "tab:blue"]
    mc_values = np.logspace(14, 20, 120)
    
    all_fmax = False

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
            m_eq_estimate = fsolve(m_eq_func, m_c/1000, args=(f_max_allpositive, m_delta_allpositive, m_c, sigma))[0]
            f_PBH_Bellomo.append(np.interp(m_eq_estimate, m_delta_allpositive, f_max_allpositive))
            #f_PBH_Bellomo.append(10**np.interp(np.log10(m_eq_estimate), np.log10(m_delta_allpositive), np.log10(f_max_allpositive)))
           
        ax.plot(mc_values, f_PBH_Bellomo, color=colors_evap[i])
        
        f_PBH_Carr = constraint_Carr(mc_values, m_delta_allpositive, f_max_allpositive, LN, params=[sigma], evolved=False)      
        ax.plot(mc_values, f_PBH_Carr, color=colors_evap[i], linestyle="None", marker="x")
    
    ax.plot(0,0, color="k", label="Bellomo et al. (2018) method")
    ax.plot(0,0, color="k", label="Carr et al. (2017) method", marker="x", linestyle="None")

    ax.set_xlabel("$m_c~[\mathrm{g}]$")                
    ax.set_ylabel("$f_\mathrm{PBH}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    if all_fmax:
        ax.set_title("No power law extrapolation \n Unevolved LN ($\sigma = {:.1f}$)".format(sigma), fontsize="small")
    else:
        ax.set_title("No power law extrapolation \n Unevolved LN ($\sigma = {:.1f}$)".format(sigma) + ", mass range only where $f_\mathrm{max} < 1$", fontsize="small")
    ax.legend(fontsize="small")
    fig.tight_layout()
    ax.set_xlim(1e14, 1e19)
    ax.set_ylim(1e-12, 1)
    fig.tight_layout()
    

#%% Constraints from COMPTEL, INTEGRAL, EGRET and Fermi-LAT. Approximate results obtained by using f_max as the constraint from each instrument, rather than the minimum over each energy bin.
# Obtained using the method from Bellomo et al. (2018) [1709.07467].
# Extrapolate using a power-law in g(M) to M < 1e11g

from scipy.optimize import fsolve


def m_eq_func(m, f_max_i_input, m_delta_input, m_c, sigma):
    return np.interp(m, m_delta_input, 1/f_max_i_input) - np.trapz(LN(m_delta_input, m_c, sigma) / f_max_i_input, m_delta_input)

if "__main__" == __name__:
    fig, ax = plt.subplots(figsize=(8,8))
    sigma = 1.84859
                
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
            m_eq_estimate = fsolve(m_eq_func, m_c/1000, args=(f_max_i_input, m_delta_input, m_c, sigma))
            f_PBH_Bellomo.append(np.interp(m_eq_estimate, m_delta_input, f_max_i_input))
           
        ax.plot(mc_values, f_PBH_Bellomo, color=colors_evap[i])
        
        f_PBH_Carr = constraint_Carr(mc_values, m_delta_input, f_max_i_input, LN, params=[sigma], evolved=False)      
        ax.plot(mc_values, f_PBH_Carr, color=colors_evap[i], linestyle="None", marker="x")
    
    ax.plot(0,0, color="k", label="Bellomo et al. (2018) method")
    ax.plot(0,0, color="k", label="Carr et al. (2017) method", marker="x", linestyle="None")

    ax.set_xlabel("$m_c~[\mathrm{g}]$")                
    ax.set_ylabel("$f_\mathrm{PBH}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    if all_fmax:
        ax.set_title("Power law extrapolation \n Unevolved LN ($\sigma = {:.1f}$)".format(sigma), fontsize="small")
    else:
        ax.set_title("Power law extrapolation \n Unevolved LN ($\sigma = {:.1f}$)".format(sigma) + ", mass range only where $f_\mathrm{max} < 1$", fontsize="small")
    ax.legend(fontsize="small")
    fig.tight_layout()
    ax.set_xlim(1e14, 1e19)
    ax.set_ylim(1e-12, 1)
    fig.tight_layout()