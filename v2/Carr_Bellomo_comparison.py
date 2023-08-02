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

"""
#%% Constraints from COMPTEL, INTEGRAL, EGRET and Fermi-LAT. Approximate results obtained by using f_max as the constraint from each instrument, rather than the minimum over each energy bin.
# Obtained using the method from Carr et al. (2017) [1705.05567].

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    # Boolean determines whether to use evolved mass function.
    evolved = True 
    
    # Boolean determines whether to use the full mass range where f_max is known or only the range where |f_max| < 1
    all_fmax = False
        
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
        
    # Power-law exponent to use
    exponent_PL_lower = 2.0
    m_delta_extrapolated = np.logspace(11, 13, 21)
    data_folder += "/PL_exp_{:.0f}/".format(exponent_PL_lower)

    for j in range(len(Deltas)):
        params_LN = [sigmas_LN[j]]
        params_SLN = [sigmas_SLN[j], alphas_SLN[j]]
        params_CC3 = [alphas_CC3[j], betas[j]]
        
        for i in range(len(constraints_names)):
            
            # Only include PBH masses and values of f_max where |f_max| < 1
            f_max_truncated = []
            m_delta_loaded_truncated = []
    
            for k, f_max in enumerate(f_max_Isatis[i]):
                if abs(f_max) < 1:
                    f_max_truncated.append(f_max)
                    m_delta_loaded_truncated.append(m_delta_values_loaded[k])
            
            # Extrapolate f_max at masses below 1e13g using a power-law
            f_max_loaded_truncated = np.array(f_max_truncated)[m_delta_values_loaded > 1e13]
            f_max_extrapolated = f_max_loaded_truncated[0] * np.power(m_delta_extrapolated / 1e13, exponent_PL_lower)
            f_max_i = np.concatenate((f_max_extrapolated, f_max_loaded_truncated))
            m_delta_values = np.concatenate((m_delta_extrapolated, m_delta_values_loaded[m_delta_values_loaded > 1e13]))
                        
            f_PBH_i_LN = constraint_Carr(mc_values, m_delta_values, f_max_i, LN, params_LN, evolved, t)
            f_PBH_i_SLN = constraint_Carr(mc_values, m_delta_values, f_max_i, SLN, params_SLN, evolved, t)
            f_PBH_i_CC3 = constraint_Carr(mc_values, m_delta_values, f_max_i, CC3, params_CC3, evolved, t)
      
            data_filename_LN = data_folder + "/LN_GC_%s" % constraints_names_short[i] + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[j])
            data_filename_SLN = data_folder + "/SLN_GC_%s" % constraints_names_short[i]  + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[j])
            data_filename_CC3 = data_folder + "/CC3_GC_%s" % constraints_names_short[i]  + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[j])
    
            np.savetxt(data_filename_LN, [mc_values, f_PBH_i_LN], delimiter="\t")
            np.savetxt(data_filename_SLN, [mc_values, f_PBH_i_SLN], delimiter="\t")
            np.savetxt(data_filename_CC3, [mc_values, f_PBH_i_CC3], delimiter="\t")
"""
#%% Tests of the method from Bellomo et al. (2018) [1709.07467].
from scipy.optimize import fsolve


def g_test_CMB(m, alpha):
    return np.power(m, 2+alpha)


def m_eq_CMB(mu, alpha, sigma):
    return mu * np.exp( (2+alpha) * sigma**2  / 2)
    
    
def g_integral(mu, sigma, m_min = 1e10, m_max = 1e20, n_steps=10000):
    m_pbh_values = np.logspace(np.log10(m_min), np.log10(m_max), n_steps)
    
    return np.trapz(LN(m_pbh_values, mu, sigma) * g_test_CMB(m_pbh_values, alpha), m_pbh_values)


def meq_finder(m_eq, mu, sigma, alpha, m_min = 1e10, m_max = 1e20, n_steps=10000):
    return g_test_CMB(m_eq, alpha) - g_integral(mu, sigma, m_min, m_max, n_steps)


if "__main__" == __name__:

    m_min = 1e10
    m_max = 1e20
    
    m_c = 1e15
    sigma = 0.5
    alpha = -4
    
    # Find M_eq using the exact result
    print("Exact M_eq = {:.8e} g".format(m_eq_CMB(m_c, alpha, sigma)))
    
    # Estimate M_eq numerically
    print("Numeric M_eq = {:.8e} g".format(fsolve(meq_finder, m_c, args=(m_c, sigma, alpha))[0]))


#%% Constraints from COMPTEL, INTEGRAL, EGRET and Fermi-LAT. Approximate results obtained by using f_max as the constraint from each instrument, rather than the minimum over each energy bin.
# Obtained using the method from Bellomo et al. (2018) [1709.07467].

def g_GC_photons(m_pbh_values, instrument_index):
    g_values = []
    f_max_values = f_max_Isatis[instrument_index]
    
    f_max_truncated_lower = f_max_values[m_delta_values_loaded < 1e13]
    m_delta_truncated_lower = m_delta_values_loaded[m_delta_values_loaded < 1e13]
    
    for m in m_pbh_values:
        if m > 1e13:
            g_values.append(np.interp(m, f_max_truncated_lower, f_max_truncated_lower))
        
        else:
            g_values.append(f_max_truncated_lower * (m / min(m_delta_truncated_lower))**2)
    
    return g_values


if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    # Boolean determines whether to useFalse evolved mass function.
    evolved = True
        
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
        
    # Power-law exponent to use
    exponent_PL_lower = 2.0
    m_delta_extrapolated = np.logspace(11, 13, 21)
    data_folder += "/PL_exp_{:.0f}/".format(exponent_PL_lower)

    for j in range(len(Deltas)):
        params_LN = [sigmas_LN[j]]
        params_SLN = [sigmas_SLN[j], alphas_SLN[j]]
        params_CC3 = [alphas_CC3[j], betas[j]]
        
        for i in range(len(constraints_names)):
            
            # Only include PBH masses and values of f_max where |f_max| < 1
            f_max_truncated = []
            m_delta_loaded_truncated = []
    
            for k, f_max in enumerate(f_max_Isatis[i]):
                if abs(f_max) < 1:
                    f_max_truncated.append(f_max)
                    m_delta_loaded_truncated.append(m_delta_values_loaded[k])
            
            # Extrapolate f_max at masses below 1e13g using a power-law
            f_max_loaded_truncated = np.array(f_max_truncated)[m_delta_values_loaded > 1e13]
            f_max_extrapolated = f_max_loaded_truncated[0] * np.power(m_delta_extrapolated / 1e13, exponent_PL_lower)
            f_max_i = np.concatenate((f_max_extrapolated, f_max_loaded_truncated))
            m_delta_values = np.concatenate((m_delta_extrapolated, m_delta_values_loaded[m_delta_values_loaded > 1e13]))
                        
            f_PBH_i_LN = constraint_Carr(mc_values, m_delta_values, f_max_i, LN, params_LN, evolved, t)
            f_PBH_i_SLN = constraint_Carr(mc_values, m_delta_values, f_max_i, SLN, params_SLN, evolved, t)
            f_PBH_i_CC3 = constraint_Carr(mc_values, m_delta_values, f_max_i, CC3, params_CC3, evolved, t)
      
            data_filename_LN = data_folder + "/LN_GC_%s" % constraints_names_short[i] + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[j])
            data_filename_SLN = data_folder + "/SLN_GC_%s" % constraints_names_short[i]  + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[j])
            data_filename_CC3 = data_folder + "/CC3_GC_%s" % constraints_names_short[i]  + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[j])
    
            np.savetxt(data_filename_LN, [mc_values, f_PBH_i_LN], delimiter="\t")
            np.savetxt(data_filename_SLN, [mc_values, f_PBH_i_SLN], delimiter="\t")
            np.savetxt(data_filename_CC3, [mc_values, f_PBH_i_CC3], delimiter="\t")
