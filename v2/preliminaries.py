#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 15:25:33 2023
@author: ppxmg2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import erf, loggamma

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

#%% Create .txt file with fitting function parameters.
# Results for lognormal from Table II of 2008.03289.
# Results for SLN and CC3 from Table II of 2009.03204.

if "__main__" == __name__:
    Deltas = [0., 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    sigmas_LN = np.array([0.373429, 0.37575, 0.39514, 0.430192, 0.557274, 0.87795, 1.84859])
    ln_mc_SLN = [4.13, 4.13, 4.15, 4.21, 4.40, 4.88, 5.41]
    mp_SLN = [40.9, 40.9, 40.9, 40.8, 40.8, 40.6, 32.9]
    sigmas_SLN = [0.55, 0.55, 0.57, 0.60, 0.71, 0.97, 2.77]
    alphas_SLN = [-2.27, -2.24, -2.07, -1.82, -1.31, -0.66, 1.39]
    mp_CC3 = [40.8, 40.8, 40.7, 40.7, 40.8, 40.6, 35.1]
    alphas_CC3 = [3.06, 3.09, 3.34, 3.82, 5.76, 18.9, 13.9]
    betas = [2.12, 2.08, 1.72, 1.27, 0.51, 0.0669, 0.0206]
    
    file_header = "Delta \t sigma (LN) \t ln_mc (SLN) \t m_p (SLN) \t sigma (SLN) \t alpha (SLN) \t alpha (CC3) \t beta (CC3)"
    params = [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas]
    
    np.savetxt("MF_params.txt", np.column_stack(params), delimiter="\t\t ", header=file_header, fmt="%s")
    
    # Check file loads correctly
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    
#%% Functions

def load_data(filename, directory="./Extracted_files/"):
    """
    Load data from a file located in the folder './Extracted_files/'.
    Parameters
    ----------
    filename : String
        Name of file to load data from.
    directory : String
        Directory in which the file is located. The default is "./../Extracted_files/".
    Returns
    -------
    Array-like.
        Contents of file.
    """
    return np.genfromtxt(directory+filename, delimiter=',', unpack=True)


def LN(m, m_c, sigma):
    """
    Log-normal mass function (with characteristic mass m_c and standard deviation sigma), evaluated at m.
    Parameters
    ----------
    m : Array-like
        PBH mass.
    m_c : Float
        Characteristic PBH mass.
    sigma : Float
        Standard deviation of the log-normal mass function.
    Returns
    -------
    Array-like
        Value of the log-normal mass function.
    """
    return np.exp(-np.log(m/m_c)**2 / (2*sigma**2)) / (np.sqrt(2*np.pi) * sigma * m)


def SLN(m, m_c, sigma, alpha):
    """
    Skew-lognormal mass function (defined in 2009.03204 Eq. 8), with characteristic mass m_c and parameters sigma and alpha, evaluated at m.
    Parameters
    ----------
    m : Array-like
        PBH mass.
    m_c : Float
        Characteristic PBH mass.
    sigma : Float
        Parameter relates to width of skew-lognormal mass function.
    alpha : Float
        Parameter controls the skewness (alpha=0 reduces to a lognormal).
    Returns
    -------
    Array-like
        Value of the skew-lognormal mass function.
    """
    return np.exp(-np.log(m/m_c)**2 / (2*sigma**2)) * (1 + erf( alpha * np.log(m/m_c) / (np.sqrt(2) * sigma))) / (np.sqrt(2*np.pi) * sigma * m)


def CC3(m, m_p, alpha, beta):
    """
    Critical collapse 3 mass function (defined in 2009.03204 Eq. 9), with peak mass m_p and parameters alpha and beta, evaluated at m.
    Parameters
    ----------
    m : Array-like
        PBH mass.
    m_p : Float
        Peak mass.
    alpha : Float
        Controls location and shape of the low mass tail.
    beta : Float
        Controls location and shape of the hig mass tail.
    Returns
    -------
    Array-like
        Value of the critical collapse 3 mass function.
    """
    m_f = m_p * np.power(beta/alpha, 1/beta)
    log_psi = np.log(beta/m_f) - loggamma((alpha+1) / beta) + (alpha * np.log(m/m_f)) - np.power(m/m_f, beta)
    return np.exp(log_psi)


def m_peak_LN(m_c, sigma):
    """
    Calculate the mass at which the log-normal mass function is maximised.
    Parameters
    ----------
    m_c : Float
        Characteristic PBH mass.
    sigma : Float
        Standard deviation of the log-normal mass function.
    Returns
    -------
    Float
        Peak mass of the log-normal mass function.
    """
    return m_c * np.exp(-sigma**2)   


def m_max_SLN(m_c, sigma, alpha, log_m_factor=5, n_steps=100000):
    """
    Estimate the mass at which the skew-lognormal mass function is maximised.
    Parameters
    ----------
    m_c : Float
        Characteristic PBH mass.
    sigma : Float
        Parameter relates to width of skew-lognormal mass function.
    alpha : Float
        Parameter controls the skewness (alpha=0 reduces to a lognormal).
    log_m_factor : Float, optional
        Number of multiples of sigma (in log-space) of masses around m_c to consider when estimating the maximum. The default is 5.
    n_steps : Integer, optional
        Number of masses to use for estimating the peak mass of the skew-lognormal mass function. The default is 100000.
    Returns
    -------
    Float
        Estimate for the peak mass of the skew-lognormal mass function..
    """
    log_m_min = np.log10(m_c) - log_m_factor*sigma
    log_m_max = np.log10(m_c) + log_m_factor*sigma

    m_pbh_values = np.logspace(log_m_min, log_m_max, n_steps)

    # Calculate mass function at each PBH mass.
    psi_values = SLN(m_pbh_values, m_c, sigma, alpha)
    
    return m_pbh_values[np.argmax(psi_values)]


def alpha_eff_extracted(M_init):
    """
    Result for alpha_eff, extracted from Fig. 1 of Mosbech & Picker (2022).

    Parameters
    ----------
    M_init_values : Float
        PBH formation mass, in grams.

    Returns
    -------
    Float.
        Value of alpha_eff.

    """
    M_init_extracted, alpha_eff_extracted = load_data("2203.05743/2203.05743_Fig1.csv")
    # Value assigned at large masses equals that of the fitting function at M >~ 1e18g in Eq. 10 of Mosbech & Picker (2022), in turn from Page (1976) [see before Eq. 27].
    alpha_eff = np.interp(M_init, M_init_extracted, alpha_eff_extracted, left=max(alpha_eff_extracted), right=2.011e-4)
    return alpha_eff


def mass_evolved(M_init_values, t):
    """
    Find the PBH mass at time t, evolved from initial masses M_init_values.

    Parameters
    ----------
    M_init_values : Array-like
        Initial PBH masses.
    t : Float
        Time (after PBH formation) at which to evaluate PBH masses.

    Returns
    -------
    Array-like
        PBH mass at time t.

    """
    # Find the PBH mass at time t, evolved from initial masses M_init_values
    M_values = []
    
    for M_init in M_init_values:
        if M_init**3 - 3 * alpha_eff_extracted(M_init) * m_Pl**3 * (t / t_Pl) <= 0:
            M_values.append(0)
        else:
            # By default, alpha_eff_mixed() takes array-like quantities as arguments.
            # Choose the 'zeroth' entry to append a scalar to the list M_values.
            M_values.append(np.power(M_init**3 - 3 * alpha_eff_extracted(M_init) * m_Pl**3 * (t / t_Pl), 1/3))
    
    return np.array(M_values)


def psi_evolved(psi_formation, M_values, M_init_values):
    """
    PBH mass function (in terms of the mass density) at time t, evolved form 
    the initial MF phi_formation using Eq. 11 of Mosbech & Picker (2022).   

    Parameters
    ----------
    psi_formation : Array-like
        Initial PBH mass distribution (in mass density).
    M_values : Array-like
        PBH masses at time t.
    M_init_values : Array-like
        Initial PBH masses.

    Returns
    -------
    Array-like
        Evolved values of the PBH mass density distribution function (not normalised to unity).

    """
    return psi_formation * (M_values / M_init_values)**3


def psi_evolved_normalised(psi_formation, M_values, M_init_values):
    """
    PBH mass function (in terms of the mass density) at time t, evolved form 
    the initial MF phi_formation using Eq. 11 of Mosbech & Picker (2022),
    normalised to one

    Parameters
    ----------
    psi_formation : Array-like
        Initial PBH mass distribution (in mass density).
    M_values : Array-like
        PBH masses at time t.
    M_init_values : Array-like
        Initial PBH masses.

    Returns
    -------
    Array-like
        Evolved values of the PBH mass density distribution function (normalised to unity).

    """
    return psi_evolved(psi_formation, M_values, M_init_values) / np.trapz(psi_evolved(psi_formation, M_values, M_init_values), M_values)


def constraint_Carr(mc_values, m_delta, f_max, psi_initial, params, evolved=True, t=t_0):
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
        m_init_values_input = np.sort(np.concatenate((np.logspace(np.log10(min(m_delta)), np.log10(m_star), 1000), np.arange(m_star, m_star*(1+1e-11), 5e2), np.arange(m_star*(1+1e-11), m_star*(1+1e-6), 1e7), np.logspace(np.log10(m_star*(1+1e-4)), np.log10(max(m_delta))+4, 1000))))
        m_values_input = mass_evolved(m_init_values_input, t)
        
    f_pbh = []
    
    for m_c in mc_values:
    
        if evolved:
            # Find evolved mass function at time t
            psi_initial_values = psi_initial(m_init_values_input, m_c, *params)
            psi_evolved_values = psi_evolved_normalised(psi_initial_values, m_values_input, m_init_values_input)
           
            # Interpolate the evolved mass function at the masses that the delta-function mass function constraints are evaluated at
            m_values_input_nozeros = m_values_input[psi_evolved_values > 0]
            psi_evolved_values_nozeros = psi_evolved_values[psi_evolved_values > 0]
            psi_evolved_interp = 10**np.interp(np.log10(m_delta), np.log10(m_values_input_nozeros), np.log10(psi_evolved_values_nozeros), left=-100, right=-100)
            
            integrand = psi_evolved_interp / f_max
            integral = np.trapz(np.nan_to_num(integrand), m_delta)
            
        else:
            integral = np.trapz(psi_initial(m_delta, m_c, *params) / f_max, m_delta)
            
        if integral == 0 or np.isnan(integral):
            f_pbh.append(10)
        else:
            f_pbh.append(1/integral)
            
    return f_pbh


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
                constraints_values.append(1e100)
            else:
                constraints_values.append(abs(constraints[j][i]))

        tightest[i] = min(constraints_values)

    return tightest


def load_results_Isatis(mf_string="mono_E500", modified=True, test_mass_range=False, wide=False):
    """
    Read in constraints on f_PBH, obtained using Isatis, with a monochromatic PBH mass function.
    Parameters
    ----------
    mf_string : String, optional
        The mass function to load constraints for. Acceptable inputs are "mono" (monochromatic), "LN" (log-normal), "SLN" (skew-lognormal) and "CC3" (critical collapse 3), plus the value of the power spectrum width Delta. 
    modified : Boolean, optional
        If True, use data from the modified version of Isatis. The modified version corrects a typo in the original version on line 1697 in Isatis.c which means that the highest-energy bin in the observational data set is not included. Otherwise, use the version of Isatis containing the typo. The default is True.
    test_mass_range : Boolean, optional
        If True, use data obtained using the same PBH mass range for all Delta (1000 BHs evenly spaced in log space between 1e11-1e21g, if wide=False).
    wide : Boolean, optional
        If True, use the 'wider' PBH mass range for all Delta (1000 BHs evenly spaced in log space between 1e11-1e22g).
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

    if wide:
        mf_string += "_wide"

    # Load Isatis constraints data.
    constraints_file = np.genfromtxt("%sresults_photons_%s.txt" % (
        Isatis_path, mf_string), dtype="str", unpack=True)[1:]

    constraints_names = []
    f_PBH_Isatis = []

    # Create array of constraints for which the constraints are physical
    # (i.e. the constraints are non-zero and positive).
    for i in range(len(constraints_file)):

        constraint = [float(constraints_file[i][j])
                      for j in range(1, len(constraints_file[i]))]

        if not(all(np.array(constraint) <= 0)):

            f_PBH_Isatis.append(constraint)

            # Create labels
            # Based upon code appearing in plotting.py within Isatis.
            temp = constraints_file[i][0].split("_")
            temp2 = ""
            for i in range(len(temp)-1):
                temp2 = "".join([temp2, temp[i], '\,\,'])
            temp2 = "".join([temp2, '\,\,[arXiv:', temp[-1], ']'])

            constraints_names.append(temp2)

    return constraints_names, f_PBH_Isatis


#%% Compare SLN and CC3 MF to Fig. 5 of 2009.03204.

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)

    # Indices in array of Delta values to reproduce Fig. 5 of 2009.03204.
    # Corresponds to Delta = 0.1, 5.0.
    Delta_indices_Fig3 = [1, 6]
    
    # Number of masses to plot
    n_masses = 1000
    
    # Approximate range of masses to include to reproduce Fig. 5 of 2009.03204.
    # Corresponds to Delta = 0.1, 5.0. 
    m_pbh_Fig3 = np.array([np.logspace(1, np.log10(80), n_masses), np.logspace(0, np.log10(2000), n_masses)])
    
    for i, Delta_index in enumerate(Delta_indices_Fig3):
        
        print(ln_mc_SLN[Delta_index])
        print(sigmas_SLN[Delta_index])
        print(alphas_SLN[Delta_index])
        print(mp_SLN[Delta_index])
        print(mp_CC3[Delta_index])
        print(alphas_CC3[Delta_index])
        print(betas[Delta_index])
        
        # Load data from 2009.03204 (provided by Andrew Gow)
        log_m_numeric, psi_numeric = np.genfromtxt("./Data/psiData/psiData_Lognormal_D-{:.1f}.txt".format(Deltas[Delta_index]), unpack=True, skip_header=1)
        # Load data from 2009.03204 (Fig. 5)
        m_loaded_SLN, psi_scaled_SLN = load_data("Delta_{:.1f}_SLN.csv".format(Deltas[Delta_index]), directory="./Extracted_files/2009.03204/")
        m_loaded_CC3, psi_scaled_CC3 = load_data("Delta_{:.1f}_CC3.csv".format(Deltas[Delta_index]), directory="./Extracted_files/2009.03204/")
        
        # Calculate the LN, SLN and CC3 mass functions.
        m_pbh_values = m_pbh_Fig3[i]
        if Delta_index == 1:
            psi_SLN = SLN(m_pbh_values, np.exp(ln_mc_SLN[Delta_index]), sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index])
        else:
            psi_SLN = SLN(m_pbh_values, np.exp(ln_mc_SLN[Delta_index]), sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index])            
        psi_CC3 = CC3(m_pbh_values, mp_CC3[Delta_index], alpha=alphas_CC3[Delta_index], beta=betas[Delta_index])
        
        m_max_SLN_val = m_max_SLN(np.exp(ln_mc_SLN[Delta_index]), sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index], n_steps=10000)
        print("M_max (SLN) = {:.1f} M_\odot".format(m_max_SLN_val))
        
        if Delta_index == 1:
            psi_SLN_max = max(SLN(m_pbh_values, np.exp(ln_mc_SLN[Delta_index]), sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index]))
            print(psi_SLN[10])
            print(psi_SLN_max)

        else:
            psi_SLN_max = max(SLN(m_pbh_values, np.exp(ln_mc_SLN[Delta_index]), sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index]))

        psi_CC3_max = max(CC3(m_pbh_values, mp_CC3[Delta_index], alpha=alphas_CC3[Delta_index], beta=betas[Delta_index]))
        
        
        # Plot the mass function.
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.plot(np.exp(log_m_numeric), psi_numeric / max(psi_numeric), color="k", label="Numeric", linewidth=2)
        ax.plot(m_loaded_SLN, psi_scaled_SLN, color="b", linestyle="None", marker="x")
        ax.plot(m_loaded_CC3, psi_scaled_CC3, color="tab:green", linestyle="None", marker="x")
        ax.plot(m_pbh_values, psi_SLN / psi_SLN_max, color="b", label="SLN", linewidth=2)
        
        if Delta_index == 6:
            ax.plot(m_pbh_values, 0.98 * psi_CC3 / psi_CC3_max, color="tab:green", label="CC3", linewidth=2)
        else:
            ax.plot(m_pbh_values, psi_CC3 / psi_CC3_max, color="tab:green", label="CC3", linewidth=2)

        ax.plot(0, 0, color="grey", linestyle="None", marker="x", label="Extracted (Fig. 5 2009.03204)")
        ax.set_xlabel("$M_\mathrm{PBH}~[M_\odot]$")
        ax.set_ylabel("$\psi(M_\mathrm{PBH}) / \psi_\mathrm{max}$")
        ax.set_xlim(min(m_pbh_values), max(m_pbh_values))
        ax.set_ylim(0.1, 1.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize="small", title="$\Delta={:.1f}$".format(Deltas[Delta_index]))
        plt.tight_layout()
        plt.show()

#%% Compare peak mass of the skew lognormal with different mass ranges and numbers of masses tested, to the values from Table II of 2009.03204.

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    # Set required fractional precision
    precision = 1e-2
    print("Fractional precision =", precision)
    
    m_c = 1e15

    # Number of steps to use when estimating the peak mass.
    n_steps_range = 10**np.arange(2, 6.1, 1)
    # Number of sigmas to use for mass range when estimating the peak mass.   
    n_sigmas_range = np.arange(1, 10.1, 1)
    
    # Minimum number of steps to use when estimating the peak mass.
    n_steps_min = max(n_steps_range) * np.ones(len(Deltas))
    # Minimum number of sigmas to use for mass range when estimating the peak mass.
    n_sigma_min = max(n_sigmas_range) * np.ones(len(Deltas))
    
    # Cycle through range of number of masses to use in the estimate
    for i in range(len(Deltas)):
        
        alpha = alphas_SLN[i]
        sigma = sigmas_SLN[i]

        sigma_range_min = max(n_sigmas_range) * np.ones(len(Deltas))
        n_steps_range_min = max(n_steps_range) * np.ones(len(Deltas))
        
        # Calculate the most precise estimate for the peak mass
        mp_best_estimate = m_max_SLN(m_c, sigma, alpha, log_m_factor=max(n_sigmas_range), n_steps=int(max(n_steps_range)))
        print("Best estimate = {:.4e}".format(mp_best_estimate))
        
        stop_loop = False
        
        for n_steps in n_steps_range:
            
            for n_sigma in n_sigmas_range:
                
                # Estimated peak mass of the SLN mass function.
                m_max_SLN_est = m_max_SLN(m_c, sigma, alpha, log_m_factor=n_sigma, n_steps=int(n_steps))
                print("Estimate = {:.4e}".format(m_max_SLN_est))
                frac_diff = abs((m_max_SLN_est - mp_best_estimate) / mp_best_estimate)
                
                if frac_diff < precision:
                    
                    n_steps_min[i] = n_steps
                    n_sigma_min[i] = n_sigma
                    
                    # Break loop, to give the minimum number of steps required and minimum range for a given precision of the M_p calculation.
                    stop_loop = True
                    break
                
                if stop_loop:
                    break
                    
            if stop_loop:
                break

        print("Delta = {:.1f}".format(Deltas[i]))
        print("n_steps_min = {:.2e}".format(n_steps_min[i]))
        print("n_sigmas_min = {:.0f}".format(n_sigma_min[i]))
        
#%% Check m_max_SLN(), by comparing results to Table II of 2009.03204, accounting for the uncertainty in parameters due to the limited precision given.

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    # Peak mass of the skew lognormal mass function, from Table II of 2009.03204.
    mp_SL = [40.9, 40.9, 40.9, 40.8, 40.8, 40.6, 32.9]
        
    # Cycle through range of number of masses to use in the estimate
    for i in range(len(Deltas)):
        print("\nDelta = {:.1f}".format(Deltas[i]))
        
        # Account for uncertainty due to the limited precision of values given in Table II of 2009.03204.
        for ln_mc in [ln_mc_SLN[i]-0.005,  ln_mc_SLN[i], ln_mc_SLN[i]+0.005]:
            
            for alpha in [alphas_SLN[i]-0.005, alphas_SLN[i], alphas_SLN[i]+0.005]:
                
                for sigma in [sigmas_SLN[i]-0.005, sigmas_SLN[i], sigmas_SLN[i]+0.005]:
                    
                    # Estimated peak mass of the SLN mass function.
                    m_max_SLN_est = m_max_SLN(np.exp(ln_mc), sigma, alpha, log_m_factor=n_sigma_min[i], n_steps=int(n_steps_min[i]))
                    
                    # Compare to peak mass given in Table II of 2009.03204
                    if abs(m_max_SLN_est - mp_SL[i]) < 0.05:
                        
                        print("Success")
                        
                        # Calculate and print fractional difference
                        frac_diff = abs((m_max_SLN_est - mp_SL[i]) / mp_SL[i])
                        print("Fractional difference = {:.2e}".format(frac_diff))
                        
#%% Plot the mass function for Delta = 5.0, showing the mass range relevant
# for the Subaru-HSC microlensing constraints.
 
if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    for i in range(len(Deltas)):
        
        if i == 6:
           
            #m_pbh_values = np.logspace(17, 24, 100)
            m_pbh_values = np.logspace(20, 26, 100)
            fig1, ax1 = plt.subplots(figsize=(6, 6))
            fig2, ax2 = plt.subplots(figsize=(6, 6))

            ymin_scaled, ymax_scaled = 1e-4, 5
            
            # Choose factors so that peak masses of the CC3 and SLN MF match
            # closely, at 1e20g (consider this range since constraints plots)
            # indicate the constraints from the SLN and CC3 MFs are quite
            # different at this peak mass.
            #m_c = 3.1e18*np.exp(ln_mc_SLN[i])
            #m_p = 2.9e18*mp_CC3[i]
            
            m_c = 5.6e20*np.exp(ln_mc_SLN[i])
            m_p = 5.25e20*mp_CC3[i]

            mp_SLN_est = m_max_SLN(m_c, sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=4, n_steps=1000)
            print("m_p (CC3) = {:.2e}".format(m_p))
            print("m_p (SLN) = {:.2e}".format(mp_SLN_est))
         
            mf_SLN = SLN(m_pbh_values, m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i])
            mf_CC3 = CC3(m_pbh_values, m_p, alpha=alphas_CC3[i], beta=betas[i])

            mf_scaled_SLN = SLN(m_pbh_values, m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i]) / max(SLN(m_pbh_values, m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i]))
            mf_scaled_CC3 = CC3(m_pbh_values, m_p, alpha=alphas_CC3[i], beta=betas[i]) / CC3(m_p, m_p, alpha=alphas_CC3[i], beta=betas[i])

            ymin, ymax = CC3(m_p, m_p, alpha=alphas_CC3[i], beta=betas[i]) * ymin_scaled, CC3(m_p, m_p, alpha=alphas_CC3[i], beta=betas[i]) * ymax_scaled

            ax1.plot(m_pbh_values, mf_scaled_SLN, color="b", label="SLN", linestyle=(0, (5, 7)))
            ax1.plot(m_pbh_values, mf_scaled_CC3, color="g", label="CC3", linestyle="dashed")
            
            ax2.plot(m_pbh_values, mf_SLN, color="b", label="SLN", linestyle=(0, (5, 7)))
            ax2.plot(m_pbh_values, mf_CC3, color="g", label="CC3", linestyle="dashed")
            
            for ax in [ax1, ax2]:
                # Show smallest PBH mass constrained by microlensing.
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.grid()
                ax.legend(fontsize="small")
                #ax.vlines(m_x, ymin=0, ymax=1, color="k", linestyle="dotted")
                ax.set_xlabel("$m~[\mathrm{g}]$")
                ax.set_xlim(min(m_pbh_values), max(m_pbh_values))
                ax.set_title("$\Delta={:.1f},~m_p={:.1e}$".format(Deltas[i], m_p) + "$~\mathrm{g}$", fontsize="small")

            ax1.vlines(9.9e21, ymin_scaled, ymax_scaled, color="k", linestyle="dashed")
            ax1.set_ylabel("$\psi / \psi_\mathrm{max}$")
            ax1.set_ylim(ymin_scaled, ymax_scaled)
            
            ax2.vlines(9.9e21, ymin, ymax, color="k", linestyle="dashed")
            ax2.set_ylabel("$\psi$")
            ax2.set_ylim(ymin, ymax)

            fig1.set_tight_layout(True)
            fig2.set_tight_layout(True)
            
            
#%% Plot the mass function for Delta = 5.0, showing the mass range relevant
# for the Subaru-HSC microlensing constraints.
 
if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    
    if i==5:
                   
        m_pbh_values = np.logspace(17, 24, 100)
        #m_pbh_values = np.logspace(20, 26, 100)
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        fig2, ax2 = plt.subplots(figsize=(6, 6))

        ymin_scaled, ymax_scaled = 1e-4, 5
        
        # Choose factors so that peak masses of the CC3 and SLN MF match
        # closely, at 1e20g (consider this range since constraints plots)
        # indicate the constraints from the SLN and CC3 MFs are quite
        # different at this peak mass.
        m_c = 3.1e18*np.exp(ln_mc_SLN[i])
        m_p = 2.9e18*mp_CC3[i]
        
        #m_c = 5.6e20*np.exp(ln_mc_SLN[i])
        #m_p = 5.25e20*mp_CC3[i]

        mp_SLN_est = m_max_SLN(m_c, sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=4, n_steps=1000)
        print("m_p (CC3) = {:.2e}".format(m_p))
        print("m_p (SLN) = {:.2e}".format(mp_SLN_est))
     
        mf_SLN = SLN(m_pbh_values, m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i])
        mf_CC3 = CC3(m_pbh_values, m_p, alpha=alphas_CC3[i], beta=betas[i])
        
        mf_scaled_SLN = SLN(m_pbh_values, m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i]) / max(SLN(m_pbh_values, m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i]))
        mf_scaled_CC3 = CC3(m_pbh_values, m_p, alpha=alphas_CC3[i], beta=betas[i]) / CC3(m_p, m_p, alpha=alphas_CC3[i], beta=betas[i])
        
        ymin, ymax = CC3(m_p, m_p, alpha=alphas_CC3[i], beta=betas[i]) * ymin_scaled, CC3(m_p, m_p, alpha=alphas_CC3[i], beta=betas[i]) * ymax_scaled

        ax1.plot(m_pbh_values, mf_scaled_SLN, color="b", label="SLN", linestyle=(0, (5, 7)))
        ax1.plot(m_pbh_values, mf_scaled_CC3, color="g", label="CC3", linestyle="dashed")
        #ax1.plot(m_pbh_values, mf_scaled_numeric, color="k", label="Numeric")
        
        ax2.plot(m_pbh_values, mf_SLN, color="b", label="SLN", linestyle=(0, (5, 7)))
        ax2.plot(m_pbh_values, mf_CC3, color="g", label="CC3", linestyle="dashed")
        #ax2.plot(m_pbh_values, mf_numeric_test, color="k", label="CC3")
        
        for ax in [ax1, ax2]:
            # Show smallest PBH mass constrained by microlensing.
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid()
            ax.legend(fontsize="small")
            #ax.vlines(m_x, ymin=0, ymax=1, color="k", linestyle="dotted")
            ax.set_xlabel("$m~[\mathrm{g}]$")
            ax.set_xlim(min(m_pbh_values), max(m_pbh_values))
            ax.set_title("$\Delta={:.1f},~m_p={:.0e}$".format(Deltas[i], m_p) + "$~\mathrm{g}$", fontsize="small")

        ax1.vlines(9.9e21, ymin_scaled, ymax_scaled, color="k", linestyle="dashed")
        ax1.set_ylabel("$\psi / \psi_\mathrm{max}$")
        ax1.set_ylim(ymin_scaled, ymax_scaled)
        
        ax2.vlines(9.9e21, ymin, ymax, color="k", linestyle="dashed")
        ax2.set_ylabel("$\psi$")
        ax2.set_ylim(ymin, ymax)

        fig1.set_tight_layout(True)
        fig2.set_tight_layout(True)


#%% Plot the integrand appearing in Eq. 12 of 1705.05567, for the microlensing constraint from Subaru-HSC
 
if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    
    i = 6        
    # Constraints for monochromatic MF.
    m_pbh_values, f_max_subaru = load_data("./2007.12697/Subaru-HSC_2007.12697_dx=5.csv")
               
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Choose factors so that peak masses of the CC3 and SLN MF match
    # closely, at 1e20g (consider this range since constraints plots)
    # indicate the constraints from the SLN and CC3 MFs are quite
    # different at this peak mass.
    m_c = 3.1e18*np.exp(ln_mc_SLN[i])
    m_p = 2.9e18*mp_CC3[i]
    
    #m_c = 5.6e20*np.exp(ln_mc_SLN[i])
    #m_p = 5.25e20*mp_CC3[i]

    mp_SLN_est = m_max_SLN(m_c, sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=4, n_steps=1000)
    print("m_p (CC3) = {:.2e}".format(m_p))
    print("m_p (SLN) = {:.2e}".format(mp_SLN_est))
 
    mf_SLN = SLN(m_pbh_values, m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i])
    mf_CC3 = CC3(m_pbh_values, m_p, alpha=alphas_CC3[i], beta=betas[i])
    #mf_numeric_test = mf_numeric(m_pbh_values, m_p, Delta=Deltas[i])
    
    ymin, ymax = 1e-26, 2.5e-23
    xmin, xmax = 9e21, 1e25

    ax.plot(m_pbh_values, mf_SLN / f_max_subaru, color="b", label="SLN", linestyle=(0, (5, 7)))
    ax.plot(m_pbh_values, mf_CC3 / f_max_subaru, color="g", label="CC3", linestyle="dashed")
    #ax.plot(m_pbh_values, mf_numeric_test / f_max_subaru, color="k", label="Numeric")
            
    # Show smallest PBH mass constrained by microlensing.
    #ax.set_xscale("log")
    #ax.set_yscale("log")
    ax.grid()
    ax.legend(fontsize="small")
    #ax.vlines(m_x, ymin=0, ymax=1, color="k", linestyle="dotted")
    ax.set_xlabel("$m~[\mathrm{g}]$")
    ax.set_xlim(xmin, xmax)
    ax.set_title("$\Delta={:.1f},~m_p={:.0e}$".format(Deltas[i], m_p) + "$~\mathrm{g}$", fontsize="small")

    ax.set_ylabel("$\psi / f_\mathrm{max}$")
    ax.set_ylim(ymin, ymax)
    
    fig.set_tight_layout(True)
    
        
#%% Proportion of values in the energy range constrained by the instruments shown in Fig. 2 of 2201.01265 above 5 GeV (which Hazma cannot calculate secondary spectra for
    
if "__main__" == __name__:
    E_min, E_max = 1e-6, 105.874
    E_number = 10000000
    energies = np.logspace(np.log10(E_min), np.log10(E_max), E_number)
    print(len(energies[energies < 5]) / len(energies))
    
    
#%% Find characteristic mass for which the minimum mass to include in a calculation is smaller than 5e14g, when emission of photons with E < 5 GeV becomes significant.

if "__main__" == __name__:
    
    m_sig = 5e14  # PBH mass below which emission of photons becomes significant.
    cutoff_values = [1e-4]
    
    for cutoff in cutoff_values:
        print("\nCutoff = {:.0e}".format(cutoff))
        
        scaled_masses_filename = "MF_scaled_mass_ranges_c={:.0f}.txt".format(-np.log10(cutoff))
        [Deltas, m_lower_LN, m_upper_LN, m_lower_SLN, m_upper_SLN, m_lower_CC3, m_upper_CC3] = np.genfromtxt(scaled_masses_filename, delimiter="\t\t ", skip_header=2, unpack=True)
        
        for i in range(len(Deltas)):
            print("\nDelta = {:.1f}".format(Deltas[i]))
            mc_sig_LN = m_sig / m_lower_LN[i]
            mc_sig_SLN = m_sig / m_lower_SLN[i]
            mc_sig_CC3 = m_sig / m_lower_CC3[i]
            print("SLN: mc_sig = {:.1e}g".format(mc_sig_SLN))
            print("CC3: mp_sig = {:.1e}g".format(mc_sig_CC3))
            print("LN: mc_sig = {:.1e}g".format(mc_sig_LN))

    
#%% Find characteristic mass for which the minimum mass to include in a calculation is larger than 1e21g, the maximum mass for which I have calculated using isatis_reproduction.py.

if "__main__" == __name__:
    
    m_sig = 1e21  # PBH mass above which results have not been calculated using isatis_reproduction.py
    cutoff_values = [1e-4]
    
    for cutoff in cutoff_values:
        print("\nCutoff = {:.0e}".format(cutoff))
        
        scaled_masses_filename = "MF_scaled_mass_ranges_c={:.0f}.txt".format(-np.log10(cutoff))
        [Deltas, m_lower_LN, m_upper_LN, m_lower_SLN, m_upper_SLN, m_lower_CC3, m_upper_CC3] = np.genfromtxt(scaled_masses_filename, delimiter="\t\t ", skip_header=2, unpack=True)
        
        for i in range(len(Deltas)):
            print("\nDelta = {:.1f}".format(Deltas[i]))
            mc_sig_SLN = m_sig / m_upper_SLN[i]
            mc_sig_CC3 = m_sig / m_upper_CC3[i]
            print("SLN: mc_sig = {:.1e}g".format(mc_sig_SLN))
            print("CC3: mp_sig = {:.1e}g".format(mc_sig_CC3))
            
            
#%% Plot evolved mass functions
if "__main__" == __name__:
    # Initial PBH mass values includes values close to the initial mass of a PBH with lifetime equal to the age of the Universe,
    # corresponding to evolved masses at t=t_0 as low as a few times 10^11 g.
    #m_pbh_values_formation = np.concatenate((np.arange(m_star, m_star*(1+1e-11), 5e2), np.arange(m_star*(1+1e-11), m_star*(1+1e-6), 1e7), np.logspace(np.log10(m_star*(1+1e-4)), 20, 500)))
    
    # Maximum mass that the Korwar & Profumo (2023) delta-function MF constraint is calculated at
    m_delta_max_KP23 = 3e17
    m_pbh_values_formation = np.concatenate((np.logspace(np.log10(m_star) - 3, np.log10(m_star)), np.arange(m_star, m_star*(1+1e-11), 5e2), np.arange(m_star*(1+1e-11), m_star*(1+1e-6), 1e7), np.logspace(np.log10(m_star*(1+1e-4)), np.log10(m_delta_max_KP23)+4, 1000)))
    m_pbh_values_evolved = mass_evolved(m_pbh_values_formation, t_0)
    m_pbh_values_evolved_t_zero = mass_evolved(m_pbh_values_formation, 0)
    m_c = 1e17
    
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)

    plot_LN = False
    plot_SLN = False
    plot_CC3 = True

    for i in range(len(Deltas)):      

        if plot_LN:
            sigma_LN = sigmas_LN[i]
            psi_initial = LN(m_pbh_values_formation, m_c, sigma_LN)
            
        elif plot_SLN:
            sigma_SLN = sigmas_SLN[i]
            alpha_SLN = alphas_SLN[i]
            psi_initial = SLN(m_pbh_values_formation, m_c, sigma_SLN, alpha_SLN)
            
        elif plot_CC3:
            alpha_CC3 = alphas_CC3[i]
            beta = betas[i]
            psi_initial = CC3(m_pbh_values_formation, m_c, alpha_CC3, beta)           
            
        psi_evolved_values = psi_evolved(psi_initial, m_pbh_values_evolved, m_pbh_values_formation)
        psi_evolved_normalised_values = psi_evolved_normalised(psi_initial, m_pbh_values_evolved, m_pbh_values_formation)
        psi_t_zero = psi_evolved_normalised(psi_initial, m_pbh_values_evolved_t_zero, m_pbh_values_formation)
        psi_t_zero_interp = 10**np.interp(np.log10(m_pbh_values_evolved), np.log10(m_pbh_values_evolved_t_zero), np.log10(psi_t_zero))

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(m_pbh_values_formation, psi_initial, label="$t=0$", color="tab:blue")
        ax.plot(m_pbh_values_evolved, psi_t_zero_interp, label="$t=0$ (normalised, \n evolved MF calculation test)", color="tab:green", linestyle="dashed")
        ax.plot(m_pbh_values_evolved, psi_evolved_normalised_values, linestyle="dashed", label="$t=t_0$ (normalised)", color="tab:orange")
        ax.plot(m_pbh_values_evolved, psi_evolved_values, label="$t=t_0$", linestyle="dotted", color="grey")

        ax.set_xlabel("$M~[\mathrm{g}]$")
        ax.set_ylabel("$\psi(M)~[\mathrm{g}]^{-1}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        
        if plot_LN:
            ax.set_title("LN, $\Delta={:.1f}$, $M_c$={:.1e}g".format(Deltas[i], m_c), fontsize="small")            
        elif plot_SLN:
            ax.set_title("SLN, $\Delta={:.1f}$, $M_c$={:.1e}g".format(Deltas[i], m_c), fontsize="small")
        elif plot_CC3:
            ax.set_title("CC3, $\Delta={:.1f}$, $M_p$={:.1e}g".format(Deltas[i], m_c), fontsize="small")

        ax.legend(fontsize="x-small")
        ax.set_xlim(1e11, max(m_pbh_values_formation))
        fig.tight_layout()
        
        fig, ax = plt.subplots(figsize=(6, 6))        
        psi_initial_interp = 10**np.interp(np.log10(m_pbh_values_evolved), np.log10(m_pbh_values_formation), np.log10(psi_initial))
        ratio_evolved = psi_evolved_values/psi_initial_interp
        ratio_evolved_normalised = psi_evolved_normalised_values/psi_initial_interp
        ratio_t_zero = psi_t_zero_interp/psi_initial_interp
        ax.plot(m_pbh_values_evolved, abs(ratio_t_zero-1), label="$t=0$ (normalised, \n evolved MF calculation test)", color="tab:green")
        ax.plot(m_pbh_values_evolved, abs(ratio_evolved_normalised-1), label="Normalised $\psi_\mathrm{N}$", color="tab:orange")
        ax.plot(m_pbh_values_evolved, abs(ratio_evolved-1), label="Unnormalised $\psi$", color="grey")
        ax.set_xlabel("$M~[\mathrm{g}]$")
        ax.set_ylabel("$|\psi(M, t_0) / \psi(M_i, t_i) - 1|$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        
        if plot_LN:
            ax.set_title("LN, $\Delta={:.1f}$, $M_c$={:.1e}g".format(Deltas[i], m_c), fontsize="small")            
        elif plot_SLN:
            ax.set_title("SLN, $\Delta={:.1f}$, $M_c$={:.1e}g".format(Deltas[i], m_c), fontsize="small")
        elif plot_CC3:
            ax.set_title("CC3, $\Delta={:.1f}$, $M_p$={:.1e}g".format(Deltas[i], m_c), fontsize="small")

        ax.set_xlim(min(m_pbh_values_formation), max(m_pbh_values_evolved))
        ax.set_ylim(5e-6, 1)
        ax.legend(fontsize="x-small")
        fig.tight_layout()

