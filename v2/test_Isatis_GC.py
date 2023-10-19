#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 10:42:31 2023

@author: ppxmg2
"""
import matplotlib.pyplot as plt
import numpy as np
from preliminaries import load_results_Isatis, envelope, LN, CC3, constraint_Carr, mass_evolved, psi_evolved_normalised

m_Pl = 2.176e-5    # Planck mass, in grams
t_Pl = 5.391e-44    # Planck time, in seconds
t_0 = 13.8e9 * 365.25 * 86400    # Age of Universe, in seconds
m_star = 7.473420349255e+14    # Formation mass of a PBH with a lifetimt equal to the age of the Universe, in grams.


def constraint_Carr_Isatis(mc_values, m_delta, f_max, psi_initial, params, evolved=True, t=t_0, n_steps=1000):
    """
    Calculate constraint on f_PBH for an extended mass function, using the method from 1705.05567.
    
    Parameters
    ----------
    mc_values : Array-like
    	Characteristic PBH masses (m_c for a (skew-)lognormal, m_p for CC3).
    m_delta : Array-like
    	Masses at which constraints for a delta-function PBH mass function are evaluated.
    f_max : Array-like
    	Constraints obtained for a monochromatic mass function.
    psi_initial : Function
    	Initial PBH mass function (in terms of the mass density).
    params : Array-like
    	Parameters of the PBH mass function.
    evolved : Boolean
    	If True, calculate constraints using the evolved PBH mass function.
    t : Float
    	Time (after PBH formation) at which to evaluate PBH masses.
        
    Returns
    -------
    f_pbh : Array-like
        Constraints on f_PBH.
    
    """
    # If delta-function mass function constraints are only calculated for PBH masses greater than 1e18g, ignore the effect of evaporation
    if min(m_delta) > 1e18:
        evolved = False
    
    if evolved:
        # Find PBH masses at time t
        m_init_values_input = np.sort(np.concatenate((np.logspace(np.log10(min(m_delta)), np.log10(m_star), n_steps), np.arange(m_star, m_star*(1+1e-11), 5e2), np.arange(m_star*(1+1e-11), m_star*(1+1e-6), 1e7), np.logspace(np.log10(m_star*(1+1e-4)), np.log10(max(m_delta))+4, n_steps))))
        m_values_input = mass_evolved(m_init_values_input, t)
        
    f_pbh = []
    
    for m_c in mc_values:
        
        # step in d log m
        dlogm = (np.log10(max(m_delta)) - np.log10(min(m_delta))) / (len(m_delta) - 1)
        # step in d m
        dm = (10**(dlogm/2) - 10**(-dlogm/2)) * m_delta
    
        if evolved:
            # Find evolved mass function at time t
            psi_initial_values = psi_initial(m_init_values_input, m_c, *params)
            psi_evolved_values = psi_evolved_normalised(psi_initial_values, m_values_input, m_init_values_input)
           
            # Interpolate the evolved mass function at the masses that the delta-function mass function constraints are evaluated at
            m_values_input_nozeros = m_values_input[psi_evolved_values > 0]
            psi_evolved_values_nozeros = psi_evolved_values[psi_evolved_values > 0]
            psi_evolved_interp = 10**np.interp(np.log10(m_delta), np.log10(m_values_input_nozeros), np.log10(psi_evolved_values_nozeros), left=-100, right=-100)
            
            integrand = psi_evolved_interp / f_max
            
        else:
            integrand = psi_initial(m_delta, m_c, *params) / f_max
            
        integral = np.sum(integrand * dm)
           
        if integral == 0 or np.isnan(integral):
            f_pbh.append(10)
        else:
            f_pbh.append(1/integral)
            
    return f_pbh


Isatis_results = load_results_Isatis(wide=True)[1]
colors_evap = ["tab:orange", "tab:green", "tab:red", "tab:blue"]
constraints_names_short = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]

m_pbh_values = np.logspace(11, 22, 1000)

fig, ax = plt.subplots(figsize=(6,6))
fig1, ax1 = plt.subplots(figsize=(6,6))
fig2, ax2 = plt.subplots(figsize=(6,6))

# Extended MF constraints

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

j=2

sigma = sigmas_LN[j]
mp_values = np.logspace(14, 18, 100)
mc_values = mp_values * np.exp(sigma**2)

# Extended MF constraints calculated before June 2023
mc_values_old = np.logspace(14, 19, 100)
fname_base = "CC_D={:.1f}_dm{:.0f}_".format(Deltas[j], -np.log10(delta_log_m)) + energies_string + "_c{:.0f}".format(-np.log10(cutoff))
constraints_names, f_PBHs_GC_old = load_results_Isatis(mf_string=fname_base, modified=True)

# Folder that new results are stored in.
data_folder = "./Data-tests/unevolved"

for i in range(len(Isatis_results)):
    
    # Newer evolved MF constraints calculated with the aid of isatis_reproduction.py
    data_filename = data_folder + "/CC3_GC_%s" % (constraints_names_short[i]) + "_Carr_Delta={:.1f}_unevolved.txt".format(Deltas[j])
    mc_new, f_PBH_new = np.genfromtxt(data_filename)
    
    f_max_threshold = np.infty
    
    Isatis_results_i = np.array(Isatis_results[i])
    print("\n%s" % constraints_names_short[i])
    print("len(Isatis results)")
    print(len(Isatis_results_i))
    print("len(Isatis results > 0)")
    print(len(Isatis_results_i[Isatis_results_i > 0.]))
    Isatis_results_i_truncated = Isatis_results_i[Isatis_results_i > 0.]
    m_pbh_allpos_Isatis = m_pbh_values[Isatis_results_i > 0.]
 
    print("len(Isatis results < {:.2e})".format(f_max_threshold))
    print(len(Isatis_results_i[Isatis_results_i < f_max_threshold]))   
 
    # Load data from isatis_reproduction.py
    f_max_Isatis_reproduction_all = np.genfromtxt("./Data/fPBH_GC_full_all_bins_%s_monochromatic_wide.txt" % constraints_names_short[i], unpack=True)
    f_max_Isatis_reproduction = envelope(np.genfromtxt("./Data/fPBH_GC_full_all_bins_%s_monochromatic_wide.txt" % constraints_names_short[i], unpack=True))
    print("len(Isatis results reproduction)")
    print(len(f_max_Isatis_reproduction))
    print("len(Isatis results reproduction < {:.2e})".format(f_max_threshold))
    print(len(f_max_Isatis_reproduction[f_max_Isatis_reproduction < f_max_threshold]))
    f_max_Isatis_reproduction_truncated = f_max_Isatis_reproduction[Isatis_results_i > 0.] 

    #ax.plot(m_pbh_allpos_Isatis, frac_diff(Isatis_results_i_truncated, f_max_Isatis_reproduction_truncated, m_pbh_allpos_Isatis, m_pbh_allpos_Isatis), label=constraints_names_short[i], color=colors_evap[i])
    ax.plot(m_pbh_allpos_Isatis, Isatis_results_i_truncated / f_max_Isatis_reproduction_truncated -1 , label=constraints_names_short[i], color=colors_evap[i])
    ax.set_xlabel("$m~[\mathrm{g}]$")
    ax.set_ylabel("$\Delta f_\mathrm{PBH} / f_\mathrm{PBH}$")
    ax.set_xscale("log")
    ax.set_xlim(min(m_pbh_allpos_Isatis), max(m_pbh_allpos_Isatis))
    ax.legend(fontsize="xx-small")
    fig.tight_layout()
    
    ax1.plot(m_pbh_allpos_Isatis, Isatis_results_i_truncated, color=colors_evap[i], label=constraints_names_short[i])
    ax1.plot(m_pbh_allpos_Isatis, f_max_Isatis_reproduction_truncated, color=colors_evap[i], marker="x", linestyle="None")
    ax1.set_xlabel("$m~[\mathrm{g}]$")
    ax1.set_ylabel("$f_\mathrm{max}$")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.legend(fontsize="xx-small")
    ax1.set_xlim(min(m_pbh_allpos_Isatis), max(m_pbh_allpos_Isatis))
    ax1.set_ylim(1e-10, 1e10)
    fig1.tight_layout()
    
    # Extended MF constraints
    f_PBH_allbins_LN = []
    f_PBH_allbins_CC3 = []
    

    for k in range(len(f_max_Isatis_reproduction_all)):
                        
        # Set values of f_max larger than some threshold to 1e100 from the f_max values calculated using Isatis
        f_max_allpositive = []

        for f_max in f_max_Isatis_reproduction_all[k]:
            if f_max == -1 or f_max > f_max_threshold:
                f_max_allpositive.append(np.infty)
            else:
                f_max_allpositive.append(f_max)
        f_PBH_allbins_LN.append(constraint_Carr(mc_values, m_pbh_values, f_max_allpositive, LN, [sigma], evolved=False))
        f_PBH_allbins_CC3.append(constraint_Carr(mc_values, m_pbh_values, f_max_allpositive, CC3, [alphas_CC3[j], betas[j]], evolved=False))
                       
    f_PBH_i_LN = envelope(f_PBH_allbins_LN)
    f_PBH_i_CC3 = envelope(f_PBH_allbins_CC3)

    if i == 0:
        ax2.plot(mc_values_old, f_PBHs_GC_old[i], color=colors_evap[i], label="Original Isatis")
        ax2.plot(mc_new, f_PBH_new, color=colors_evap[i], alpha=0.5, label="New (Isatis reproduction)")
        ax2.plot(mc_values, f_PBH_i_CC3, color=colors_evap[i], alpha=0.5, marker="x", label="New (Isatis reproduction), recalculated", linestyle="None")
    else:
        ax2.plot(mc_values_old, f_PBHs_GC_old[i], color=colors_evap[i])
        ax2.plot(mc_new, f_PBH_new, color=colors_evap[i], alpha=0.5)
        ax2.plot(mc_values, f_PBH_i_CC3, color=colors_evap[i], alpha=0.5, marker="x", linestyle="None")

ax2.legend(fontsize="xx-small")
ax2.set_xlabel("$m_c~[\mathrm{g}]$")
ax2.set_ylabel("$f_\mathrm{PBH}$")
ax2.set_title("$\Delta={:.1f}$".format(Deltas[j]) + ", CC3")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlim(1e14, 1e18)
ax2.set_ylim(1e-10, 1)
fig2.tight_layout()
