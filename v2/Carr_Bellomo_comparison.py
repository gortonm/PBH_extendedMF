#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 13:54:26 2023

@author: ppxmg2
"""
# Script to compare results obtained using the methods from Carr et al. (2017) [1705.05567] and Bellomo et al. (2018) [1709.07467]
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
            #print(n)
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

def integral_over_g(m_delta_allpositive, m_c, params, t=t_0):
    # Find PBH masses at time t
    m_init_values_input = np.sort(np.concatenate((np.logspace(np.log10(min(m_delta_allpositive)), np.log10(m_star), 1000), np.arange(m_star, m_star*(1+1e-11), 5e2), np.arange(m_star*(1+1e-11), m_star*(1+1e-6), 1e7), np.logspace(np.log10(m_star*(1+1e-4)), np.log10(max(m_delta_allpositive))+4, 1000))))
    m_values_input = mass_evolved(m_init_values_input, t)
    
    if use_LN:
        psi_initial_values = LN(m_init_values_input, m_c, *params)
        
    elif use_CC3:
        psi_initial_values = CC3(m_init_values_input, m_c, *params)
        
    # Calculate evolved MF
    psi_evolved_values = psi_evolved_normalised(psi_initial_values, m_values_input, m_init_values_input)
    
    # Interpolate the evolved mass function at the masses that the delta-function mass function constraints are evaluated at
    m_values_input_nozeros = m_values_input[psi_evolved_values > 0]
    psi_evolved_values_nozeros = psi_evolved_values[psi_evolved_values > 0]
    psi_evolved_interp = 10**np.interp(np.log10(m_delta_allpositive), np.log10(m_values_input_nozeros), np.log10(psi_evolved_values_nozeros), left=-100, right=-100)
    
    integrand = psi_evolved_interp / f_max_allpositive
    return np.trapz(np.nan_to_num(integrand), m_delta_allpositive)

def m_eq_func(m, f_max_allpositive, m_delta_allpositive, m_c, evolved=True, t=t_0):
    
    if evolved:
        return np.interp(m, m_delta_allpositive, 1/f_max_allpositive) - integral
    
    else:
        if use_LN:
            return np.interp(m, m_delta_allpositive, 1/f_max_allpositive) - np.trapz(LN(m_delta_allpositive, m_c, sigma) / f_max_allpositive, m_delta_allpositive)
        elif use_CC3:
            return np.interp(m, m_delta_allpositive, 1/f_max_allpositive) - np.trapz(CC3(m_delta_allpositive, m_c, alpha, beta) / f_max_allpositive, m_delta_allpositive)

if "__main__" == __name__:
    fig, ax = plt.subplots(figsize=(8,8))
    
    if use_LN:
        sigma = 1.84859
        params = [sigma]
        
    elif use_CC3:
        alpha = 13.9
        beta = 0.0206
        params = [alpha, beta]
                
    m_delta_values_loaded = np.logspace(11, 21, 1000)
    constraints_names, f_max_Isatis = load_results_Isatis(modified=True)
    colors_evap = ["tab:orange", "tab:green", "tab:red", "tab:blue"]
    mc_values = np.logspace(14, 20, 120)
    
    all_fmax = False
    evolved = True
    
    if evolved:
        title_part = "Evolved"
    else:
        title_part = "Unevolved"

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
            
            if evolved:
                integral = integral_over_g(m_delta_allpositive, m_c, params)
            
            if use_LN:
                m_eq_estimate = findroot(m_eq_func, min(m_delta_allpositive), max(m_delta_allpositive), args=(f_max_allpositive, m_delta_allpositive, m_c, sigma))
            elif use_CC3:
                m_eq_estimate = findroot(m_eq_func, min(m_delta_allpositive), max(m_delta_allpositive), args=(f_max_allpositive, m_delta_allpositive, m_c, alpha, beta))

            f_PBH_Bellomo.append(np.interp(m_eq_estimate, m_delta_allpositive, f_max_allpositive))
           
        ax.plot(mc_values, f_PBH_Bellomo, color=colors_evap[i])
        
        if use_LN:
            f_PBH_Carr = constraint_Carr(mc_values, m_delta_allpositive, f_max_allpositive, LN, params, evolved)
        elif use_CC3:
            f_PBH_Carr = constraint_Carr(mc_values, m_delta_allpositive, f_max_allpositive, CC3, params, evolved)
            
        ax.plot(mc_values, f_PBH_Carr, color=colors_evap[i], linestyle="None", marker="x")
    
    ax.plot(0,0, color="k", label="Bellomo et al. (2018) method")
    ax.plot(0,0, color="k", label="Carr et al. (2017) method", marker="x", linestyle="None")

    ax.set_ylabel("$f_\mathrm{PBH}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    if all_fmax:
        if use_LN:
            ax.set_title("No power law extrapolation \n " + title_part + " LN ($\Delta=5$)", fontsize="small")
        elif use_CC3:
            ax.set_title("No power law extrapolation \n " + title_part + " CC3 ($\Delta=5$)", fontsize="small")
    else:
        if use_LN:
            ax.set_title("Power law extrapolation \n " + title_part + "  LN ($\Delta=5$)" + ", mass range only where $f_\mathrm{max} < 1$", fontsize="small")
        elif use_CC3:
            ax.set_title("Power law extrapolation \n " + title_part + "  CC3 ($\Delta=5$)" + ", mass range only where $f_\mathrm{max} < 1$", fontsize="small")
    ax.legend(fontsize="small")
    fig.tight_layout()
    
    if use_LN:
        ax.set_xlim(1e14, 1e19)
        ax.set_ylim(1e-12, 1)
        ax.set_xlabel("$m_c~[\mathrm{g}]$")                
        
    elif use_CC3:
        ax.set_xlim(1e14, 1e18)
        ax.set_ylim(1e-10, 1)
        ax.set_xlabel("$m_p~[\mathrm{g}]$")                
        
    fig.tight_layout()
    

#%% Constraints from COMPTEL, INTEGRAL, EGRET and Fermi-LAT. Approximate results obtained by using f_max as the constraint from each instrument, rather than the minimum over each energy bin.
# Obtained using the method from Bellomo et al. (2018) [1709.07467].
# Extrapolate using a power-law in g(M) to M < 1e13g

use_LN = False
use_CC3 = True

def integral_over_g(m_delta_input, m_c, params, t=t_0):
    # Find PBH masses at time t
    m_init_values_input = np.sort(np.concatenate((np.logspace(np.log10(min(m_delta_input)), np.log10(m_star), 1000), np.arange(m_star, m_star*(1+1e-11), 5e2), np.arange(m_star*(1+1e-11), m_star*(1+1e-6), 1e7), np.logspace(np.log10(m_star*(1+1e-4)), np.log10(max(m_delta_input))+4, 1000))))
    m_values_input = mass_evolved(m_init_values_input, t)
    
    if use_LN:
        psi_initial_values = LN(m_init_values_input, m_c, *params)
        
    elif use_CC3:
        psi_initial_values = CC3(m_init_values_input, m_c, *params)
        
    # Calculate evolved MF
    psi_evolved_values = psi_evolved_normalised(psi_initial_values, m_values_input, m_init_values_input)
    
    # Interpolate the evolved mass function at the masses that the delta-function mass function constraints are evaluated at
    m_values_input_nozeros = m_values_input[psi_evolved_values > 0]
    psi_evolved_values_nozeros = psi_evolved_values[psi_evolved_values > 0]
    psi_evolved_interp = 10**np.interp(np.log10(m_delta_input), np.log10(m_values_input_nozeros), np.log10(psi_evolved_values_nozeros), left=-100, right=-100)
    
    integrand = psi_evolved_interp / f_max_i_input
    return np.trapz(np.nan_to_num(integrand), m_delta_input)
    

def m_eq_func(m, f_max_i_input, m_delta_input, m_c, params):
    
    if evolved:        
        return np.interp(m, m_delta_input, 1/f_max_i_input) - integral

    else:
        if use_LN:
            return np.interp(m, m_delta_input, 1/f_max_i_input) - np.trapz(LN(m_delta_input, m_c, *params) / f_max_i_input, m_delta_input)
        elif use_CC3:
            return np.interp(m, m_delta_input, 1/f_max_i_input) - np.trapz(CC3(m_delta_input, m_c, *params) / f_max_i_input, m_delta_input)       
    
if "__main__" == __name__:
    fig, ax = plt.subplots(figsize=(8,8))
    
    if use_LN:
        sigma = 1.84859
        params = [sigma]
        
    elif use_CC3:
        alpha = 13.9
        beta = 0.0206
        params = [alpha, beta]
                
    m_delta_values_loaded = np.logspace(11, 21, 1000)
    constraints_names, f_max_Isatis = load_results_Isatis(modified=True)
    colors_evap = ["tab:orange", "tab:green", "tab:red", "tab:blue"]
    mc_values = np.logspace(14, 20, 120)
    
    # Power-law exponent to use
    exponent_PL_lower = 2.0
    m_delta_extrapolated = np.logspace(9, 13, 410)
    #m_delta_extrapolated = np.logspace(11, 13, 41)
   
    evolved = True
    all_fmax = False
    
    if evolved:
        title_init = "Power law extrapolation \n Evolved"
    else:
        title_init = "Power law extrapolation \n Unevolved"
    
    constraints_names_short = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]

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
            
            if evolved:
                integral = integral_over_g(m_delta_input, m_c, params)
            
            if use_LN:
                m_eq_estimate = findroot(m_eq_func, min(m_delta_input), max(m_delta_input), args=(f_max_i_input, m_delta_input, m_c, params))
            elif use_CC3:
                m_eq_estimate = findroot(m_eq_func, min(m_delta_input), max(m_delta_input), args=(f_max_i_input, m_delta_input, m_c, params))
               
            f_PBH_Bellomo.append(np.interp(m_eq_estimate, m_delta_input, f_max_i_input))
           
        ax.plot(mc_values, f_PBH_Bellomo, color=colors_evap[i])
        
        if use_LN:
            f_PBH_Carr = constraint_Carr(mc_values, m_delta_input, f_max_i_input, LN, params, evolved)
            
            # Load and plot constraints calculated in GC_photon_constraints.py
            data_filename_LN = "./Data-tests/unevolved/PL_exp_{:.0f}".format(exponent_PL_lower) + "/LN_GC_%s" % constraints_names_short[i]  + "_Carr_Delta=5.0_approx_unevolved.txt"
            mc_LN_unevolved, f_PBH_LN_unevolved = np.genfromtxt(data_filename_LN, delimiter="\t")
            ax.plot(mc_LN_unevolved, f_PBH_LN_unevolved, color="k", alpha=0.3, marker="+")
            
        elif use_CC3:
            f_PBH_Carr = constraint_Carr(mc_values, m_delta_input, f_max_i_input, CC3, params, evolved)
            
        ax.plot(mc_values, f_PBH_Carr, color=colors_evap[i], linestyle="None", marker="x")
    
    ax.plot(0,0, color="k", label="Bellomo et al. (2018) method")
    ax.plot(0,0, color="k", label="Carr et al. (2017) method", marker="x", linestyle="None")
    ax.plot(0,0, color="k", alpha=0.5, label="Carr et al. (2017) method \n From GC_constraints_Carr.py", marker="+")

    ax.set_ylabel("$f_\mathrm{PBH}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    if all_fmax:
        if use_LN:
            ax.set_title(title_init + " LN ($\Delta=5$)", fontsize="small")
        
        elif use_CC3:
            
            # Load and plot constraints calculated in GC_photon_constraints.py
            data_filename_LN = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower) + "/CC3_GC_%s" % constraints_names_short[i]  + "_Carr_Delta=5.0_approx.txt"
            mc_LN_evolved, f_PBH_LN_evolved = np.genfromtxt(data_filename_LN, delimiter="\t")
            ax.plot(mc_LN_evolved, f_PBH_LN_evolved, color="k", alpha=0.3, marker="+")
            
            ax.set_title(title_init + " CC3 ($\Delta=5$)", fontsize="small")
    else:
        if use_LN:
            ax.set_title(title_init + " LN ($\Delta=5$), mass range only where $f_\mathrm{max} < 1$", fontsize="small")
        elif use_CC3:
            ax.set_title(title_init + " CC3 ($\Delta=5$)" + ", mass range only where $f_\mathrm{max} < 1$", fontsize="small")
    ax.legend(fontsize="small")
    fig.tight_layout()
    
    if use_LN:
        ax.set_xlim(1e14, 1e19)
        ax.set_ylim(1e-12, 1)
        ax.set_xlabel("$m_c~[\mathrm{g}]$")                
        
    elif use_CC3:
        ax.set_xlim(1e14, 1e18)
        ax.set_ylim(1e-10, 1)
        ax.set_xlabel("$m_p~[\mathrm{g}]$")                
        
    fig.tight_layout()


#%% Constraints from Korwar & Profumo (2023).
# Obtained using the method from Bellomo et al. (2018) [1709.07467].
# Extrapolate using a power-law in g(M) to M < 1e15g

use_LN = False
use_SLN = False
use_CC3 = True

def integral_over_g(m_delta_input, m_c, params, t=t_0):
    # Find PBH masses at time t
    m_init_values_input = np.sort(np.concatenate((np.logspace(np.log10(min(m_delta_input)), np.log10(m_star), 1000), np.arange(m_star, m_star*(1+1e-11), 5e2), np.arange(m_star*(1+1e-11), m_star*(1+1e-6), 1e7), np.logspace(np.log10(m_star*(1+1e-4)), np.log10(max(m_delta_input))+4, 1000))))
    m_values_input = mass_evolved(m_init_values_input, t)
    
    if use_LN:
        psi_initial_values = LN(m_init_values_input, m_c, *params)
        
    elif use_CC3:
        psi_initial_values = CC3(m_init_values_input, m_c, *params)
        
    # Calculate evolved MF
    psi_evolved_values = psi_evolved_normalised(psi_initial_values, m_values_input, m_init_values_input)
    
    # Interpolate the evolved mass function at the masses that the delta-function mass function constraints are evaluated at
    m_values_input_nozeros = m_values_input[psi_evolved_values > 0]
    psi_evolved_values_nozeros = psi_evolved_values[psi_evolved_values > 0]
    psi_evolved_interp = 10**np.interp(np.log10(m_delta_input), np.log10(m_values_input_nozeros), np.log10(psi_evolved_values_nozeros), left=-100, right=-100)
    
    integrand = psi_evolved_interp / f_max_input
    return np.trapz(np.nan_to_num(integrand), m_delta_input)
    

def m_eq_func(m, f_max_input, m_delta_input, m_c, params):
    return np.interp(m, m_delta_input, 1/f_max_input) - integral

    
if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    i = 0
    
    if use_LN:
        params = [sigmas_LN[i]]
        mc_values = np.logspace(16, 18, 10) * np.exp(sigmas_LN[i]**2)  # convert range of peak masses to values in terms of the characteristic mass of a log-normal, m_c
        
    elif use_CC3:
        params = [alphas_CC3[i], betas[i]]
        mc_values = np.logspace(16, 18, 10)
    
    # Load delta function MF constraints calculated using Isatis, to use the method from 1705.05567.
    m_delta_values, f_max = load_data("2302.04408/2302.04408_MW_diffuse_SPI.csv")
    
    # Power-law exponent to use between 1e15g and 1e16g.
    exponent_PL_upper = 2.0
    # Power-law exponent to use between 1e11g and 1e15g.
    exponent_PL_lower = 2.0
    
    m_delta_extrapolated_upper = np.logspace(15, 16, 11)
    m_delta_extrapolated_lower = np.logspace(11, 15, 41)
    
    f_max_extrapolated_upper = min(f_max) * np.power(m_delta_extrapolated_upper / min(m_delta_values), exponent_PL_upper)
    f_max_extrapolated_lower = min(f_max_extrapolated_upper) * np.power(m_delta_extrapolated_lower / min(m_delta_extrapolated_upper), exponent_PL_lower)

    f_max_input = np.concatenate((f_max_extrapolated_lower, f_max_extrapolated_upper, f_max))
    m_delta_input = np.concatenate((m_delta_extrapolated_lower, m_delta_extrapolated_upper, m_delta_values))
       
    title_init = "Power law extrapolation \n Evolved"    
    
    # Calculate results using method from Bellomo et al. (2018)
    f_PBH_Bellomo = []
    m_eq_values = []
            
    for m_c in mc_values:
        integral = integral_over_g(m_delta_input, m_c, params)
        m_eq_estimate = findroot(m_eq_func, min(m_delta_input), max(m_delta_input), args=(f_max_input, m_delta_input, m_c, params))
        m_eq_values.append(m_eq_estimate)
        f_PBH_Bellomo.append(np.interp(m_eq_estimate, m_delta_input, f_max_input))
 
    fig, ax = plt.subplots(figsize=(6, 6))
        
    # Calculate results using method from Bellomo et al. (2018)
    
    ax.set_ylabel("$f_\mathrm{PBH}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    if use_LN:
        ax.plot(mc_values*np.exp(-sigmas_LN[i]**2), f_PBH_Bellomo, color="tab:blue", linestyle="None", marker="x", label="Bellomo et al. (2018) method")
        
        # Load and plot constraints calculated in GC_photon_constraints.py
        data_filename_LN = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower) + "/LN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[i], exponent_PL_lower)
        mc_LN_evolved, f_PBH_LN_evolved = np.genfromtxt(data_filename_LN, delimiter="\t")
        ax.plot(mc_LN_evolved*np.exp(-sigmas_LN[i]**2), f_PBH_LN_evolved, color=(0.5294, 0.3546, 0.7020), label="Carr et al. (2017) method \n From GC_constraints_Carr.py")
        ax.set_title(title_init + " LN ($\Delta={:.1f}$)".format(Deltas[i]), fontsize="small")
        ax.set_xlabel("$m_c~[\mathrm{g}]$")                
       
    
    elif use_CC3:
        ax.plot(mc_values, f_PBH_Bellomo, color="tab:blue", linestyle="None", marker="x", label="Bellomo et al. (2018) method")

        # Load and plot constraints calculated in GC_photon_constraints.py
        data_filename_CC3 = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower) + "/CC3_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[i], exponent_PL_lower)
        mp_CC3_evolved, f_PBH_CC3_evolved = np.genfromtxt(data_filename_CC3, delimiter="\t")
        ax.plot(mp_CC3_evolved, f_PBH_CC3_evolved, color=(0.5294, 0.3546, 0.7020), label="Carr et al. (2017) method \n From GC_constraints_Carr.py")
        ax.set_title(title_init + " CC3 ($\Delta={:.1f}$)".format(Deltas[i]), fontsize="small")
        ax.set_xlabel("$m_p~[\mathrm{g}]$")                
    
    print("M_eq = {:.3e} g".format(m_eq_values[0]))
        
    ax.legend(fontsize="small")
    ax.set_xlim(1e16, 1e18)
    ax.set_ylim(1e-5, 1)
    fig.tight_layout()