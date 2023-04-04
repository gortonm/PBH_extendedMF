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
            
    return np.array(f_pbh)


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
            if constraints[j][i] <= 0:
                constraints_values.append(10)
            else:
                constraints_values.append(abs(constraints[j][i]))
        
        tightest[i] = min(constraints_values)
    
    return tightest
        

def load_results_Isatis(mf_string="mono_E500", modified=True, test_mass_range=False):
    """
    Read in constraints on f_PBH, obtained using Isatis, with a monochromatic PBH mass function.
    Parameters
    ----------
    mf_string : String, optional
        The mass function to load constraints for. Acceptable inputs are "mono" (monochromatic), "LN" (log-normal), "SLN" (skew-lognormal) and "CC3" (critical collapse 3), plus the value of the power spectrum width Delta. 
    modified : Boolean, optional
        If True, use data from the modified version of Isatis. The modified version corrects a typo in the original version on line 1697 in Isatis.c which means that the highest-energy bin in the observational data set is not included. Otherwise, use the version of Isatis containing the typo. The default is True.
    test_mass_range : Boolean, optional
        If True, use data obtained using the same method as for the constraints from 1705.05567.
    Returns
    -------
    constraints_names : Array-like
        Name of instrument and arxiv reference for constraint on PBHs.
    f_PBH_Isatis : Array-like
        Constraint on the fraction of dark matter in PBHs, calculated using Isatis.
    """
    # Choose path to Isatis.
    if modified:
        Isatis_path = "../../Downloads/version_finale/scripts/Isatis/"
    else:
        mf_string = "GC_mono"
        Isatis_path = "../../Downloads/version_finale_unmodified/scripts/Isatis/"
    
    if test_mass_range:
        mf_string += "_test_range"
    
    # Load Isatis constraints data.
    constraints_file = np.genfromtxt("%sresults_photons_%s.txt"%(Isatis_path, mf_string), dtype = "str", unpack=True)[1:]
    
    constraints_names = []
    f_PBH_Isatis = []
    
    # Create array of constraints for which the constraints are physical
    # (i.e. the constraints are non-zero and positive).
    for i in range(len(constraints_file)):

        constraint = [float(constraints_file[i][j]) for j in range(1, len(constraints_file[i]))]
            
        if not(all(np.array(constraint)<=0)):
    
            f_PBH_Isatis.append(constraint)
            
            # Create labels
            # Based upon code appearing in plotting.py within Isatis.
            temp = constraints_file[i][0].split("_")
            temp2 = ""
            for i in range(len(temp)-1):
                temp2 = "".join([temp2,temp[i],'\,\,'])
            temp2 = "".join([temp2,'\,\,[arXiv:',temp[-1],']'])
    
            constraints_names.append(temp2)
            
    return constraints_names, f_PBH_Isatis


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


#%% Plot results for a monochromatic mass function, obtained using isatis_reproduction.py,
# and compare to the results shown in Fig. 3 of 2201.01265.
# Includes highest-energy bin.
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
# and compare to the results obtained using the method from 1705.05567, from
# the envelope of constraints from each energy bin.
# Using the modified version of Isatis.
# Both forms of the constraint calculated using the same range and number of PBH masses.
if "__main__" == __name__:
    
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

    mc_values = np.logspace(14, 19, 100)
    colors_evap = ["tab:orange", "tab:green", "tab:red", "tab:blue"]
    
    m_mono_values = np.logspace(11, 21, 1000)
    
    for j in range(len(sigmas_LN[:-1])):
        
        # Filename of constraints obtained using Isatis.
        fname_base = "LN_D={:.1f}_dm{:.0f}_".format(Deltas[i], -np.log10(delta_log_m)) + energies_string + "_c{:.0f}".format(-np.log10(cutoff))
        
        # Constraints calculated using Isatis, using a PBH mass range logarithmically spaced between 1e11 and 1e21 grams.
        constraints_names, f_PBH_Isatis = load_results_Isatis(mf_string="LN_D={:.1f}".format(Deltas[j]), modified=True, test_mass_range=True)    
        
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
        ax.set_title("$\Delta={:.1f}$".format(Deltas[j]))
        plt.tight_layout()


#%% Plot results for a log-normal mass function, obtained using Isatis,
# and compare to the results obtained using the method from 1705.05567, using
# the constraint from each energy bin separately.
# Using the modified version of Isatis.
# Both forms of the constraint calculated using the same range and number of PBH masses.
if "__main__" == __name__:
    
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

    mc_values = np.logspace(14, 19, 100)
    colors_evap = ["tab:orange", "tab:green", "tab:red", "tab:blue"]
    constraints_names_short = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
    
    m_mono_values = np.logspace(11, 21, 1000)
    
    for j in range(len(sigmas_LN[:-1])):
        
        # Filename of constraints obtained using Isatis.
        fname_base = "LN_D={:.1f}_dm{:.0f}_".format(Deltas[i], -np.log10(delta_log_m)) + energies_string + "_c{:.0f}".format(-np.log10(cutoff))
        
        if j==5:
        
            # Constraints calculated using Isatis, using a PBH mass range logarithmically spaced between 1e11 and 1e21 grams.
            constraints_names, f_PBH_Isatis = load_results_Isatis(mf_string="LN_D={:.1f}".format(Deltas[j]), modified=True, test_mass_range=True)    
            
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
            ax.set_title("$\Delta={:.1f}$".format(Deltas[j]))
            plt.tight_layout()
            

#%% Plot results for a log-normal mass function, obtained using Isatis,
# and compare to the results obtained using the method from 1705.05567, using
# the constraint from each energy bin separately.
# For the monochromatic MF constraints, only include the range where f_PBH > 1.
# and m_c > 1e16g.
# Using the modified version of Isatis.
# Both forms of the constraint calculated using the same range and number of PBH masses.
if "__main__" == __name__:
    
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

    mc_values = np.logspace(14, 19, 100)
    colors_evap = ["tab:orange", "tab:green", "tab:red", "tab:blue"]
    constraints_names_short = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
    
    m_mono_values_init = np.logspace(11, 21, 1000)
    
    for j in range(len(sigmas_LN)):
        
        # Filename of constraints obtained using Isatis.
        fname_base = "LN_D={:.1f}_dm{:.0f}_".format(Deltas[i], -np.log10(delta_log_m)) + energies_string + "_c{:.0f}".format(-np.log10(cutoff))
        
        # Constraints calculated using Isatis, using a PBH mass range logarithmically spaced between 1e11 and 1e21 grams.
        constraints_names, f_PBH_Isatis = load_results_Isatis(mf_string="LN_D={:.1f}".format(Deltas[j]), modified=True, test_mass_range=True)    
        
        # Load monochromatic MF constraints calculated using Isatis, to use the method from 1705.05567.
        # Using each energy bin per instrument individually for the monochromatic MF constraint, then obtaining the tightest constraint from each instrument using envelope().
        constraints_names, f_max = load_results_Isatis(modified=True)
        
        # Minimum and maximum monochromatic MF masses to include constraints from 1705.05567.
        m_mono_min = 1e14
        m_mono_max = 2e17
        m_mono_values_truncated = m_mono_values_init[m_mono_values_init > m_mono_min]
        m_mono_values = m_mono_values_truncated[m_mono_values_truncated < m_mono_max]
       
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
                
                constraint_energy_bin_truncated = constraint_energy_bin[m_mono_values_init > m_mono_min]
                f_max_values = constraint_energy_bin_truncated[m_mono_values_truncated < m_mono_max]
                
                # Calculate constraint on f_PBH from each bin
                f_PBH_k = constraint_Carr(mc_values, m_mono=m_mono_values, f_max=f_max_values, mf=LN, params=params_LN)
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
        ax.set_title("$\Delta={:.1f}$".format(Deltas[j]))
        plt.tight_layout()
        
#%% Plot results for a skew-lognormal mass function, obtained using Isatis,
# and compare to the results obtained using the method from 1705.05567, using
# the constraint from each energy bin separately.
# For the monochromatic MF constraints, only include the range where f_PBH > 1.
# and m_c > 1e16g.
# Using the modified version of Isatis.
# Both forms of the constraint calculated using the same range and number of PBH masses.
if "__main__" == __name__:
    
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

    mp_CC3_values = np.logspace(14, 19, 100)
    mc_SLN_values = np.logspace(14, 19, 100)
    
    colors_evap = ["tab:orange", "tab:green", "tab:red", "tab:blue"]
    constraints_names_short = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
    
    m_mono_values_init = np.logspace(11, 21, 1000)
        
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    for j in range(len(Deltas)):
        
        mp_SLN_values = [m_max_SLN(m_c, sigma=sigmas_SLN[j], alpha=alphas_SLN[j], log_m_factor=3, n_steps=1000) for m_c in mc_SLN_values]
        
        # Filename of constraints obtained using Isatis.
        fname_base_SLN = "SL_D={:.1f}_dm{:.0f}_".format(Deltas[j], -np.log10(delta_log_m)) + energies_string + "_c{:.0f}".format(-np.log10(cutoff))
        fname_base_CC3 = "CC_D={:.1f}_dm{:.0f}_".format(Deltas[j], -np.log10(delta_log_m)) + energies_string + "_c{:.0f}".format(-np.log10(cutoff))
        
        # Constraints calculated using Isatis, using a PBH mass range logarithmically spaced between 1e11 and 1e21 grams.
        constraints_names, f_PBH_Isatis_SLN = load_results_Isatis(mf_string=fname_base_SLN, modified=True, test_mass_range=False)    
        constraints_names, f_PBH_Isatis_CC3 = load_results_Isatis(mf_string=fname_base_CC3, modified=True, test_mass_range=False)    
        
        # Load monochromatic MF constraints calculated using Isatis, to use the method from 1705.05567.
        # Using each energy bin per instrument individually for the monochromatic MF constraint, then obtaining the tightest constraint from each instrument using envelope().
        constraints_names, f_max = load_results_Isatis(modified=True)
        
        # Minimum and maximum monochromatic MF masses to include constraints from 1705.05567.
        m_mono_min = 5e14
        m_mono_max = 2e17
        m_mono_values_truncated = m_mono_values_init[m_mono_values_init > m_mono_min]
        m_mono_values = m_mono_values_truncated[m_mono_values_truncated < m_mono_max]
       
        params_SLN = [sigmas_SLN[j], alphas_SLN[j]]
        params_CC3 = [alphas_CC3[j], betas[j]]
        
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        
        for i in range(len(constraints_names)):
            ax1.plot(mp_SLN_values, f_PBH_Isatis_SLN[i], label=constraints_names[i], color=colors_evap[i])
            ax2.plot(mp_CC3_values, f_PBH_Isatis_CC3[i], label=constraints_names[i], color=colors_evap[i])
            
            # Calculate constraint using method from 1705.05567.
            
            # Constraints data for each energy bin of each instrument.
            constraints_mono_file = np.transpose(np.genfromtxt("./Data/fPBH_GC_full_all_bins_%s_monochromatic.txt"%(constraints_names_short[i])))
            # Constraints for an extended MF, from each instrument.
            energy_bin_constraints_SLN = []
            energy_bin_constraints_CC3 = []
            
            for k in range(len(constraints_mono_file)):
                
                # Constraint from a particular energy bin
                constraint_energy_bin = constraints_mono_file[k]
                
                constraint_energy_bin_truncated = constraint_energy_bin[m_mono_values_init > m_mono_min]
                f_max_values = constraint_energy_bin_truncated[m_mono_values_truncated < m_mono_max]
                
                # Calculate constraint on f_PBH from each bin
                f_PBH_k = constraint_Carr(mc_values, m_mono=m_mono_values, f_max=f_max_values, mf=SLN, params=params_SLN)
                energy_bin_constraints_SLN.append(f_PBH_k)
                
                f_PBH_k = constraint_Carr(mc_values, m_mono=m_mono_values, f_max=f_max_values, mf=CC3, params=params_CC3)
                energy_bin_constraints_CC3.append(f_PBH_k)

            # Calculate constraint using method from 1705.05567, and plot.
            f_PBH_Carr_SLN = envelope(energy_bin_constraints_SLN)
            f_PBH_Carr_CC3 = envelope(energy_bin_constraints_CC3)

            ax1.plot(mp_SLN_values, f_PBH_Carr_SLN, marker="x", linestyle="None", color=colors_evap[i])
            ax2.plot(mp_CC3_values, f_PBH_Carr_CC3, marker="x", linestyle="None", color=colors_evap[i])
        
        for ax in ax1, ax2:
            ax.set_xlim(1e14, 1e18)
            ax.set_ylim(10**(-10), 1)
            ax.set_ylabel("$f_\mathrm{PBH}$")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend(fontsize="small")
        ax1.set_xlabel("$m_p~[\mathrm{g}]$")
        ax2.set_xlabel("$m_p~[\mathrm{g}]$")
        ax1.set_title("SLN, $\Delta={:.1f}$".format(Deltas[j]))
        ax2.set_title("CC3, $\Delta={:.1f}$".format(Deltas[j]))
        fig1.set_tight_layout(True)
        fig2.set_tight_layout(True)
        fig1.savefig("./Tests/Figures/SLN_mmin={:.0e}g_mmax={:.0e}g_Delta={:.1f}.png".format(m_mono_min, m_mono_max, Deltas[j]))
        fig2.savefig("./Tests/Figures/CC3_mmin={:.0e}g_mmax={:.0e}g_Delta={:.1f}.png".format(m_mono_min, m_mono_max, Deltas[j]))

#%% Plot results for a skew-lognormal mass function, obtained using Isatis,
# and compare to the results obtained using the method from 1705.05567, using
# the constraint from each energy bin separately.
# For the monochromatic MF constraints, set constraint to the constraint at 
# at m = 5e14g when m < 5e14g.
# Using the modified version of Isatis.
# Both forms of the constraint calculated using the same range and number of PBH masses.
if "__main__" == __name__:
    
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

    mp_CC3_values = np.logspace(14, 19, 100)
    mc_SLN_values = np.logspace(14, 19, 100)
    
    colors_evap = ["tab:orange", "tab:green", "tab:red", "tab:blue"]
    constraints_names_short = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
    
    m_star = 5e14    # mass below which to set constraint to large values
    m_mono_values = np.logspace(11, 21, 1000)
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    for j in range(len(Deltas)):
        
        if j >= 5:
            
            mp_SLN_values = [m_max_SLN(m_c, sigma=sigmas_SLN[j], alpha=alphas_SLN[j], log_m_factor=3, n_steps=1000) for m_c in mc_SLN_values]
        
            # Filename of constraints obtained using Isatis.
            fname_base_SLN = "SL_D={:.1f}_dm{:.0f}_".format(Deltas[j], -np.log10(delta_log_m)) + energies_string + "_c{:.0f}".format(-np.log10(cutoff))
            fname_base_CC3 = "CC_D={:.1f}_dm{:.0f}_".format(Deltas[j], -np.log10(delta_log_m)) + energies_string + "_c{:.0f}".format(-np.log10(cutoff))
            
            # Constraints calculated using Isatis, using a PBH mass range logarithmically spaced between 1e11 and 1e21 grams.
            constraints_names, f_PBH_Isatis_SLN = load_results_Isatis(mf_string=fname_base_SLN, modified=True, test_mass_range=False)    
            constraints_names, f_PBH_Isatis_CC3 = load_results_Isatis(mf_string=fname_base_CC3, modified=True, test_mass_range=False)    
            
            # Load monochromatic MF constraints calculated using Isatis, to use the method from 1705.05567.
            # Using each energy bin per instrument individually for the monochromatic MF constraint, then obtaining the tightest constraint from each instrument using envelope().
            constraints_names, f_max_values = load_results_Isatis(modified=True)
            
            params_SLN = [sigmas_SLN[j], alphas_SLN[j]]
            params_CC3 = [alphas_CC3[j], betas[j]]
            
            fig1, ax1 = plt.subplots(figsize=(8, 8))
            fig2, ax2 = plt.subplots(figsize=(8, 8))
            
            for i in range(len(constraints_names)):
                ax1.plot(mp_SLN_values, f_PBH_Isatis_SLN[i], label=constraints_names[i], color=colors_evap[i])
                ax2.plot(mp_CC3_values, f_PBH_Isatis_CC3[i], label=constraints_names[i], color=colors_evap[i])
                
                # Calculate constraint using method from 1705.05567.
                
                # Constraints data for each energy bin of each instrument.
                constraints_mono_file = np.transpose(np.genfromtxt("./Data/fPBH_GC_full_all_bins_%s_monochromatic.txt"%(constraints_names_short[i])))
                # Constraints for an extended MF, from each instrument.
                energy_bin_constraints_SLN = []
                energy_bin_constraints_CC3 = []
                
                for k in range(len(constraints_mono_file)):
                    
                    # Constraint from a particular energy bin
                    constraint_energy_bin = constraints_mono_file[k]
                    
                    min_constraint = constraint_energy_bin[np.argmax(m_mono_values > m_star)]
                    constraint_energy_bin_modified = []
                    
                    for l, f_max in enumerate(constraint_energy_bin):
                        if m_mono_values[l] < m_star:
                            constraint_energy_bin_modified.append(min_constraint)
                        else:
                            constraint_energy_bin_modified.append(constraint_energy_bin[l])
                    
                    
                    print(constraint_energy_bin_modified)
                    
                    # Calculate constraint on f_PBH from each bin
                    f_PBH_k = constraint_Carr(mc_values, m_mono=m_mono_values, f_max=constraint_energy_bin_modified, mf=SLN, params=params_SLN)
                    energy_bin_constraints_SLN.append(f_PBH_k)
                    
                    f_PBH_k = constraint_Carr(mc_values, m_mono=m_mono_values, f_max=constraint_energy_bin_modified, mf=CC3, params=params_CC3)
                    energy_bin_constraints_CC3.append(f_PBH_k)
    
                # Calculate constraint using method from 1705.05567, and plot.
                f_PBH_Carr_SLN = envelope(energy_bin_constraints_SLN)
                f_PBH_Carr_CC3 = envelope(energy_bin_constraints_CC3)
    
                ax1.plot(mp_SLN_values, f_PBH_Carr_SLN, marker="x", linestyle="None", color=colors_evap[i])
                ax2.plot(mp_CC3_values, f_PBH_Carr_CC3, marker="x", linestyle="None", color=colors_evap[i])
            
            for ax in ax1, ax2:
                ax.set_xlim(1e16, 1e18)
                ax.set_ylim(10**(-7), 1)
                ax.set_ylabel("$f_\mathrm{PBH}$")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.legend(fontsize="small")
            ax1.set_xlabel("$m_p~[\mathrm{g}]$")
            ax2.set_xlabel("$m_p~[\mathrm{g}]$")
            ax1.set_title("SLN, $\Delta={:.1f}$".format(Deltas[j]))
            ax2.set_title("CC3, $\Delta={:.1f}$".format(Deltas[j]))
            fig1.set_tight_layout(True)
            fig2.set_tight_layout(True)
            fig1.savefig("./Tests/Figures/SLN_MF_cutoff_mstar={:.1e}g_Delta={:.1f}.png".format(m_star, Deltas[j]))
            fig2.savefig("./Tests/Figures/CC3_MF_cutoff_mstar={:.1e}g_Delta={:.1f}.png".format(m_star, Deltas[j]))

#%% Plot results for a skew-lognormal mass function, obtained using Isatis,
# and compare to the results obtained using the method from 1705.05567, using
# the constraint from each energy bin separately.
# For the monochromatic MF constraints, set constraint to the constraint to a 
# very small / large value m < 5e14g.
# Using the modified version of Isatis.
# Both forms of the constraint calculated using the same range and number of PBH masses.
if "__main__" == __name__:
    
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

    mp_CC3_values = np.logspace(14, 19, 100)
    mc_SLN_values = np.logspace(14, 19, 100)
    
    colors_evap = ["tab:orange", "tab:green", "tab:red", "tab:blue"]
    constraints_names_short = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
    
    m_star = 5e14    # mass below which to set constraint to large values
    min_constraint = 1e100   # constraint to set f_max for m < m_star
    m_mono_values = np.logspace(11, 21, 1000)
        
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    for j in range(len(Deltas)):
        
        if j >= 5:
            
            mp_SLN_values = [m_max_SLN(m_c, sigma=sigmas_SLN[j], alpha=alphas_SLN[j], log_m_factor=3, n_steps=1000) for m_c in mc_SLN_values]
        
            # Filename of constraints obtained using Isatis.
            fname_base_SLN = "SL_D={:.1f}_dm{:.0f}_".format(Deltas[j], -np.log10(delta_log_m)) + energies_string + "_c{:.0f}".format(-np.log10(cutoff))
            fname_base_CC3 = "CC_D={:.1f}_dm{:.0f}_".format(Deltas[j], -np.log10(delta_log_m)) + energies_string + "_c{:.0f}".format(-np.log10(cutoff))
            
            # Constraints calculated using Isatis, using a PBH mass range logarithmically spaced between 1e11 and 1e21 grams.
            constraints_names, f_PBH_Isatis_SLN = load_results_Isatis(mf_string=fname_base_SLN, modified=True, test_mass_range=False)    
            constraints_names, f_PBH_Isatis_CC3 = load_results_Isatis(mf_string=fname_base_CC3, modified=True, test_mass_range=False)    
            
            # Load monochromatic MF constraints calculated using Isatis, to use the method from 1705.05567.
            # Using each energy bin per instrument individually for the monochromatic MF constraint, then obtaining the tightest constraint from each instrument using envelope().
            constraints_names, f_max_values = load_results_Isatis(modified=True)
            
            params_SLN = [sigmas_SLN[j], alphas_SLN[j]]
            params_CC3 = [alphas_CC3[j], betas[j]]
            
            fig1, ax1 = plt.subplots(figsize=(8, 8))
            fig2, ax2 = plt.subplots(figsize=(8, 8))
            
            for i in range(len(constraints_names)):
                ax1.plot(mp_SLN_values, f_PBH_Isatis_SLN[i], label=constraints_names[i], color=colors_evap[i])
                ax2.plot(mp_CC3_values, f_PBH_Isatis_CC3[i], label=constraints_names[i], color=colors_evap[i])
                
                # Calculate constraint using method from 1705.05567.
                
                # Constraints data for each energy bin of each instrument.
                constraints_mono_file = np.transpose(np.genfromtxt("./Data/fPBH_GC_full_all_bins_%s_monochromatic.txt"%(constraints_names_short[i])))
                # Constraints for an extended MF, from each instrument.
                energy_bin_constraints_SLN = []
                energy_bin_constraints_CC3 = []
                
                for k in range(len(constraints_mono_file)):
                    
                    # Constraint from a particular energy bin
                    constraint_energy_bin = constraints_mono_file[k]
                    constraint_energy_bin_modified = []
                    
                    for l, f_max in enumerate(constraint_energy_bin):
                        if m_mono_values[l] < m_star:
                            constraint_energy_bin_modified.append(min_constraint)
                        else:
                            constraint_energy_bin_modified.append(constraint_energy_bin[l])
                                                                    
                    # Calculate constraint on f_PBH from each bin
                    f_PBH_k = constraint_Carr(mc_values, m_mono=m_mono_values, f_max=constraint_energy_bin_modified, mf=SLN, params=params_SLN)
                    energy_bin_constraints_SLN.append(f_PBH_k)
                    
                    f_PBH_k = constraint_Carr(mc_values, m_mono=m_mono_values, f_max=constraint_energy_bin_modified, mf=CC3, params=params_CC3)
                    energy_bin_constraints_CC3.append(f_PBH_k)
    
                # Calculate constraint using method from 1705.05567, and plot.
                f_PBH_Carr_SLN = envelope(energy_bin_constraints_SLN)
                f_PBH_Carr_CC3 = envelope(energy_bin_constraints_CC3)
    
                ax1.plot(mp_SLN_values, f_PBH_Carr_SLN, marker="x", linestyle="None", color=colors_evap[i])
                ax2.plot(mp_CC3_values, f_PBH_Carr_CC3, marker="x", linestyle="None", color=colors_evap[i])
            
            for ax in ax1, ax2:
                ax.set_xlim(1e16, 1e18)
                ax.set_ylim(10**(-7), 1)
                ax.set_ylabel("$f_\mathrm{PBH}$")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.legend(fontsize="small")
            ax1.set_xlabel("$m_p~[\mathrm{g}]$")
            ax2.set_xlabel("$m_p~[\mathrm{g}]$")
            ax1.set_title("SLN, $\Delta={:.1f}$".format(Deltas[j]))
            ax2.set_title("CC3, $\Delta={:.1f}$".format(Deltas[j]))
            fig1.set_tight_layout(True)
            fig2.set_tight_layout(True)
            fig1.savefig("./Tests/Figures/SLN_MF_f={:.1e}_mstar={:.1e}g_Delta={:.1f}.png".format(min_constraint, m_star, Deltas[j]))
            fig2.savefig("./Tests/Figures/CC3_MF_f={:.1e}_mstar={:.1e}g_Delta={:.1f}.png".format(min_constraint, m_star, Deltas[j]))

