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


def constraint_Carr(mc_values, m_delta, f_max, psi_initial, params, evolved=True, t=t_0, n_steps=1000):
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
                        
#%% Plot the CC3 mass function, and compare to the limits at small and large (m / m_p) [m_p = peak mass]

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)    
    # Peak mass
    m_p = 1e16
    
    for i in range(len(Deltas[0:])):
        alpha = alphas_CC3[i]
        beta = betas[i]
    
        fig, ax = plt.subplots(figsize=(7, 7))
    
        m_pbh_values = np.logspace(np.log10(m_p) - 1, np.log10(m_p) + 2, 100)
        mf_CC3_exact = CC3(m_pbh_values, m_p, alpha=alphas_CC3[i], beta=betas[i])
        
        m_f = m_p * np.power(beta/alpha, 1/beta)
        
        log_mf_approx_lowmass = np.log(beta/m_f) - loggamma((alpha+1) / beta) + (alpha * np.log(m_pbh_values/m_f))
        mf_CC3_approx_lowmass = np.exp(log_mf_approx_lowmass)
        
        log_mf_approx_highmass = np.log(beta/m_f) - loggamma((alpha+1) / beta)- np.power(m_pbh_values/m_f, beta)
        mf_CC3_approx_highmass = np.exp(log_mf_approx_highmass)
        
        ax.plot(m_pbh_values, mf_CC3_exact, label="Exact")
        ax.plot(m_pbh_values, mf_CC3_approx_lowmass, label=r"$\propto (m/m_p)^\alpha$", linestyle="dotted")  
        ax.plot(m_pbh_values, mf_CC3_approx_highmass, label=r"$\propto \exp\left(-\frac{\alpha}{\beta}\frac{m}{m_p}\right)$", linestyle="dotted")  

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize="small", title="$\Delta={:.1f}$".format(Deltas[i]))
        ax.set_xlabel("$m~[\mathrm{g}]$")
        ax.set_xlim(min(m_pbh_values), max(m_pbh_values)/30)
        ax.set_ylim(mf_CC3_exact[0], 3*max(mf_CC3_exact))
        ax.set_ylabel("$\psi~[\mathrm{g}^{-1}]$")
        fig.tight_layout()
                        
                        
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
            
            mc_SLN = 5.6e20*np.exp(ln_mc_SLN[i])
            m_p = 5.25e20*mp_CC3[i]
            mc_LN = m_p * np.exp(+sigmas_LN[i]**2)
            
            mp_SLN_est = m_max_SLN(mc_SLN, sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=4, n_steps=1000)
            print("m_p (CC3) = {:.2e}".format(m_p))
            print("m_p (SLN) = {:.2e}".format(mp_SLN_est))

            mf_LN = LN(m_pbh_values, mc_LN, sigma=sigmas_LN[i])
            mf_SLN = SLN(m_pbh_values, mc_SLN, sigma=sigmas_SLN[i], alpha=alphas_SLN[i])
            mf_CC3 = CC3(m_pbh_values, m_p, alpha=alphas_CC3[i], beta=betas[i])

            mf_scaled_SLN = SLN(m_pbh_values, mc_SLN, sigma=sigmas_SLN[i], alpha=alphas_SLN[i]) / max(SLN(m_pbh_values, mc_SLN, sigma=sigmas_SLN[i], alpha=alphas_SLN[i]))
            mf_scaled_CC3 = CC3(m_pbh_values, m_p, alpha=alphas_CC3[i], beta=betas[i]) / CC3(m_p, m_p, alpha=alphas_CC3[i], beta=betas[i])

            ymin, ymax = CC3(m_p, m_p, alpha=alphas_CC3[i], beta=betas[i]) * ymin_scaled, CC3(m_p, m_p, alpha=alphas_CC3[i], beta=betas[i]) * ymax_scaled

            ax1.plot(m_pbh_values, mf_scaled_SLN, color="b", label="SLN", linestyle=(0, (5, 7)))
            ax1.plot(m_pbh_values, mf_scaled_CC3, color="g", label="CC3", linestyle="dashed")

            ax2.plot(m_pbh_values, mf_LN, color="r", label="LN", dashes=[6, 2])            
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
# for the Korwar & Profumo (2023) constraints.
 
if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    for i in range(len(Deltas)):
        
        if i == 6:
           
            m_pbh_values_init = np.logspace(14, 21, 100)
            fig, ax = plt.subplots(figsize=(6, 6))
            
            # Choose peak masses corresponding roughly to the maximum mass at which f_PBH < 1 for the CC3 and LN MFs in KP '23
            mc_SLN = 1.02e19
            m_p = 1.5e18
            mc_LN = m_p * np.exp(+sigmas_LN[i]**2)
            
            mp_SLN_est = m_max_SLN(mc_SLN, sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=4, n_steps=1000)
            print("m_p (CC3) = {:.2e}".format(m_p))
            print("m_p (SLN) = {:.2e}".format(mp_SLN_est))

            mf_LN_init = LN(m_pbh_values_init, mc_LN, sigma=sigmas_LN[i])
            mf_SLN_init = SLN(m_pbh_values_init, mc_SLN, sigma=sigmas_SLN[i], alpha=alphas_SLN[i])
            mf_CC3_init = CC3(m_pbh_values_init, m_p, alpha=alphas_CC3[i], beta=betas[i])
            
            m_pbh_values_evolved = mass_evolved(m_pbh_values_init, t_0)
            mf_LN_evolved = psi_evolved_normalised(mf_LN_init, m_pbh_values_evolved, m_pbh_values_init)
            mf_SLN_evolved = psi_evolved_normalised(mf_SLN_init, m_pbh_values_evolved, m_pbh_values_init)
            mf_CC3_evolved = psi_evolved_normalised(mf_CC3_init, m_pbh_values_evolved, m_pbh_values_init)

            ax.plot(m_pbh_values_evolved, mf_LN_evolved, color="r", label="LN", dashes=[6, 2])            
            ax.plot(m_pbh_values_evolved, mf_SLN_evolved, color="b", label="SLN", linestyle=(0, (5, 7)))
            ax.plot(m_pbh_values_evolved, mf_CC3_evolved, color="g", label="CC3", linestyle="dashed")
            
            # Show smallest PBH mass constrained by microlensing.
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend(fontsize="small")
            ax.set_xlabel("$m~[\mathrm{g}]$")
            ax.set_xlim(min(m_pbh_values_evolved), max(m_pbh_values_evolved))
            ax.set_title("$\Delta={:.1f},~m_p={:.1e}$".format(Deltas[i], m_p) + "$~\mathrm{g}$", fontsize="small")
            ax.set_ylabel("$\psi_\mathrm{N}$~\mathrm{[g^{-1}]}$")

            fig.tight_layout()
            
            
#%% Plot the mass function for Delta = 0 and 5, showing the mass range relevant
# for the Galactic Centre photon constraints from Isatis (1e14g)
 
if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    for i in range(len(Deltas)):
        
        if i == 0 or i == 6:
           
            m_pbh_values_init = np.sort(np.concatenate((np.arange(m_star, m_star*(1+1e-11), 5e2), np.arange(m_star*(1+1e-11), m_star*(1+1e-6), 1e7),  np.logspace(15, 21, 100))))
            fig, ax = plt.subplots(figsize=(6, 6))
            
            # Choose peak masses corresponding roughly to the maximum mass at which f_PBH < 1 for the CC3 and LN MFs in KP '23
            mc_SLN = 1.53e14
            m_p = 1e14
            mc_LN = m_p * np.exp(+sigmas_LN[i]**2)
            
            mp_SLN_est = m_max_SLN(mc_SLN, sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=4, n_steps=1000)
            print("m_p (CC3) = {:.2e}".format(m_p))
            print("m_p (SLN) = {:.2e}".format(mp_SLN_est))

            mf_LN_init = LN(m_pbh_values_init, mc_LN, sigma=sigmas_LN[i])
            mf_SLN_init = SLN(m_pbh_values_init, mc_SLN, sigma=sigmas_SLN[i], alpha=alphas_SLN[i])
            mf_CC3_init = CC3(m_pbh_values_init, m_p, alpha=alphas_CC3[i], beta=betas[i])
            
            m_pbh_values_evolved = mass_evolved(m_pbh_values_init, t_0)
            mf_LN_evolved = psi_evolved_normalised(mf_LN_init, m_pbh_values_evolved, m_pbh_values_init)
            mf_SLN_evolved = psi_evolved_normalised(mf_SLN_init, m_pbh_values_evolved, m_pbh_values_init)
            mf_CC3_evolved = psi_evolved_normalised(mf_CC3_init, m_pbh_values_evolved, m_pbh_values_init)

            ax.plot(m_pbh_values_evolved, mf_LN_evolved, color="r", label="LN", dashes=[6, 2])            
            ax.plot(m_pbh_values_evolved, mf_SLN_evolved, color="b", label="SLN", linestyle=(0, (5, 7)))
            ax.plot(m_pbh_values_evolved, mf_CC3_evolved, color="g", label="CC3", linestyle="dashed")
            
            # Show smallest PBH mass constrained by microlensing.
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend(fontsize="small")
            ax.set_xlabel("$m~[\mathrm{g}]$")
            ax.set_xlim(min(m_pbh_values_evolved), max(m_pbh_values_evolved))
            ax.set_ylim(1e-30, 1e-10)
            ax.set_title("$\Delta={:.1f},~m_p={:.1e}$".format(Deltas[i], m_p) + "$~\mathrm{g}$", fontsize="small")
            ax.set_ylabel("$\psi_\mathrm{N}$~\mathrm{[g^{-1}]}$")

            fig.tight_layout()


#%% Plot the mass function for Delta = 0, 2 and 5, showing the mass range relevant
# for the Galactic Centre photon constraints from Isatis (1e16g)
 
if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    for i in range(len(Deltas)):
        
        if i == 0:
           
            m_pbh_values_init = np.sort(np.concatenate((np.logspace(np.log10(m_star)-5, np.log10(m_star), 100), np.arange(m_star, m_star*(1+1e-11), 5e2), np.arange(m_star*(1+1e-11), m_star*(1+1e-6), 1e7),  np.logspace(15, 21, 100))))
            fig, ax = plt.subplots(figsize=(6, 6))
            
            # Choose peak masses corresponding roughly to the maximum mass at which f_PBH < 1 for the CC3 and LN MFs in KP '23
            #mc_SLN = 6.8e16   # for Delta = 5
            #mc_SLN = 3.24e16    # for Delta = 2
            mc_SLN = 1.53e16   # for Delta = 0
            m_p = 1e16
            mc_LN = m_p * np.exp(+sigmas_LN[i]**2)
            
            mp_SLN_est = m_max_SLN(mc_SLN, sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=4, n_steps=1000)
            print("m_p (CC3) = {:.2e}".format(m_p))
            print("m_p (SLN) = {:.2e}".format(mp_SLN_est))

            mf_LN_init = LN(m_pbh_values_init, mc_LN, sigma=sigmas_LN[i])
            mf_SLN_init = SLN(m_pbh_values_init, mc_SLN, sigma=sigmas_SLN[i], alpha=alphas_SLN[i])
            mf_CC3_init = CC3(m_pbh_values_init, m_p, alpha=alphas_CC3[i], beta=betas[i])
            
            m_pbh_values_evolved = mass_evolved(m_pbh_values_init, t_0)
            mf_LN_evolved = psi_evolved_normalised(mf_LN_init, m_pbh_values_evolved, m_pbh_values_init)
            mf_SLN_evolved = psi_evolved_normalised(mf_SLN_init, m_pbh_values_evolved, m_pbh_values_init)
            mf_CC3_evolved = psi_evolved_normalised(mf_CC3_init, m_pbh_values_evolved, m_pbh_values_init)


            ax.plot(m_pbh_values_init, mf_LN_init, color="r", linestyle="dotted")            
            ax.plot(m_pbh_values_init, mf_SLN_init, color="b", linestyle="dotted")
            ax.plot(m_pbh_values_init, mf_CC3_init, color="g", linestyle="dotted")


            ax.plot(m_pbh_values_evolved, mf_LN_evolved, color="r", label="LN", dashes=[6, 2])            
            ax.plot(m_pbh_values_evolved, mf_SLN_evolved, color="b", label="SLN", linestyle=(0, (5, 7)))
            ax.plot(m_pbh_values_evolved, mf_CC3_evolved, color="g", label="CC3", linestyle="dashed")
            
            # Show smallest PBH mass constrained by microlensing.
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend(fontsize="small")
            ax.set_xlabel("$m~[\mathrm{g}]$")
            ax.set_xlim(1e13, 1e18)
            ax.set_ylim(1e-23, 1e-17)
            ax.set_title("$\Delta={:.1f},~m_p={:.1e}$".format(Deltas[i], m_p) + "$~\mathrm{g}$", fontsize="small")
            ax.set_ylabel("$\psi_\mathrm{N}$")

            fig.tight_layout()
            

#%% Plot the mass function for Delta = 0, 2 and 5, showing the mass range relevant
# for the Galactic Centre photon constraints from Isatis (1e16g)
 
if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, has_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    plot_LN = True
    plot_SLN = False
    plot_CC3 = False
    
    if plot_LN:
        colors = ["grey", "lime", "lime", "lime", "lime", "tab:red", "pink"]
        m_eq_values = [1.007e16, 0,0,0,0, 9.991e15, 1.616e16]
    elif plot_SLN:
        colors = ["grey", "lime", "lime", "lime", "lime", "tab:blue", "deepskyblue"]
        m_eq_values = [0,0,0]
    elif plot_CC3:
        colors = ["grey", "lime", "lime", "lime", "lime", "tab:green", "lime"]
        m_eq_values = [8.319e15, 0,0,0,0, 9.765e15, 1.585e16]

    fig0, ax0 = plt.subplots(figsize=(7, 7))
    fig1, ax1 = plt.subplots(figsize=(7, 7))
    
    f_Mi_denominator = []

    for i in [0, 5, 6]:

        m_pbh_values_init = np.sort(np.concatenate((np.logspace(np.log10(m_star)-5, np.log10(m_star), 100), np.arange(m_star, m_star*(1+1e-11), 5e2), np.arange(m_star*(1+1e-11), m_star*(1+1e-6), 1e7),  np.logspace(15, 21, 100))))
                               
        # Choose peak masses corresponding roughly to the maximum mass at which f_PBH < 1 for the CC3 and LN MFs in KP '23
        if i == 6:
            mc_SLN = 6.8e16   # for Delta = 5
        elif i == 5:
            mc_SLN = 3.24e16    # for Delta = 2
        elif i ==0:
            mc_SLN = 1.53e16   # for Delta = 0
            
        m_p = 1e16
        mc_LN = m_p * np.exp(+sigmas_LN[i]**2)
        
        mp_SLN_est = m_max_SLN(mc_SLN, sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=4, n_steps=1000)
        print("m_p (CC3) = {:.2e}".format(m_p))
        print("m_p (SLN) = {:.2e}".format(mp_SLN_est))

        mf_LN_init = LN(m_pbh_values_init, mc_LN, sigma=sigmas_LN[i])
        mf_SLN_init = SLN(m_pbh_values_init, mc_SLN, sigma=sigmas_SLN[i], alpha=alphas_SLN[i])
        mf_CC3_init = CC3(m_pbh_values_init, m_p, alpha=alphas_CC3[i], beta=betas[i])
        
        m_pbh_values_evolved = mass_evolved(m_pbh_values_init, t_0)
        mf_LN_evolved = psi_evolved_normalised(mf_LN_init, m_pbh_values_evolved, m_pbh_values_init)
        mf_SLN_evolved = psi_evolved_normalised(mf_SLN_init, m_pbh_values_evolved, m_pbh_values_init)
        mf_CC3_evolved = psi_evolved_normalised(mf_CC3_init, m_pbh_values_evolved, m_pbh_values_init)
        
        mf_LN_evolved_unnormalised = psi_evolved(mf_LN_init, m_pbh_values_evolved, m_pbh_values_init)
        mf_SLN_evolved_unnormalised = psi_evolved(mf_SLN_init, m_pbh_values_evolved, m_pbh_values_init)
        mf_CC3_evolved_unnormalised = psi_evolved(mf_CC3_init, m_pbh_values_evolved, m_pbh_values_init)

        if plot_LN:
                        
            ax0.plot(m_pbh_values_init, mf_LN_init, color=colors[i], linestyle="dotted")
            ax0.plot(m_pbh_values_evolved, mf_LN_evolved, color=colors[i], label="${:.0f}$".format(Deltas[i]))
            
            ax1.plot(m_pbh_values_init, mf_LN_init/max(mf_LN_init), color=colors[i], linestyle="dotted")
            ax1.plot(m_pbh_values_evolved, mf_LN_evolved_unnormalised/max(mf_LN_evolved_unnormalised), color=colors[i], label="${:.0f}$".format(Deltas[i]))
            
            if i == 5:
                f_Mi_denominator = mf_LN_init
                mf_evolved_Delta2 = mf_LN_evolved_unnormalised
            elif i == 6:
                f_Mi = mf_LN_init / f_Mi_denominator
                #ax1.plot(m_pbh_values_evolved, f_Mi*mf_evolved_Delta2, color="k", linestyle="dashed")
                #ax1.plot(m_pbh_values_init, f_Mi*mf_evolved_Delta2, color="k", linestyle="dashed")

        elif plot_SLN:
                          
            ax0.plot(m_pbh_values_init, mf_SLN_init, color=colors[i], linestyle="dotted")
            ax0.plot(m_pbh_values_evolved, mf_SLN_evolved, color=colors[i], label="${:.0f}$".format(Deltas[i]))
            
            ax1.plot(m_pbh_values_init, mf_SLN_init/max(mf_SLN_init), color=colors[i], linestyle="dotted")
            ax1.plot(m_pbh_values_evolved, mf_SLN_evolved_unnormalised/max(mf_SLN_evolved_unnormalised), color=colors[i], label="${:.0f}$".format(Deltas[i]))
            
            if i == 5:
                f_Mi_denominator = mf_SLN_init
                mf_evolved_Delta2 = mf_SLN_evolved_unnormalised
            elif i == 6:
                f_Mi = mf_SLN_init / f_Mi_denominator
                #ax1.plot(m_pbh_values_evolved, f_Mi*mf_evolved_Delta2, color="k", linestyle="dashed")
                #ax1.plot(m_pbh_values_init, f_Mi*mf_evolved_Delta2, color="k", linestyle="dashed")

        elif plot_CC3:
                      
            ax0.plot(m_pbh_values_init, mf_CC3_init, color=colors[i], linestyle="dotted")
            ax0.plot(m_pbh_values_evolved, mf_CC3_evolved,  color=colors[i], label="${:.0f}$".format(Deltas[i]))  
            
            ax1.plot(m_pbh_values_init, mf_CC3_init/max(mf_CC3_init), color=colors[i], linestyle="dotted")
            ax1.plot(m_pbh_values_evolved, mf_CC3_evolved_unnormalised/max(mf_CC3_evolved_unnormalised), color=colors[i], label="${:.0f}$".format(Deltas[i]))
            
            if i == 5:
                f_Mi_denominator = mf_CC3_init
                mf_evolved_Delta2 = mf_CC3_evolved_unnormalised
            elif i == 6:
                f_Mi = mf_CC3_init / f_Mi_denominator
                #ax1.plot(m_pbh_values_evolved, f_Mi*mf_evolved_Delta2, color="k", linestyle="dashed")
                #ax1.plot(m_pbh_values_init, f_Mi*mf_evolved_Delta2, color="k", linestyle="dashed")
            
        # Plot equivalent mass M_eq
        """
        for ax in [ax0, ax1]:
            ax.vlines(m_eq_values[i], 1e-23, 1e-15, linestyle="dashdot", color=colors[i])
        """
    # Show smallest PBH mass constrained by microlensing.
    for ax in [ax0, ax1]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize="small", title="$\Delta$")
        ax.set_xlabel("$m~[\mathrm{g}]$")
        ax.set_xlim(1e13, 1e18)
        
        if plot_CC3:
            ax.set_title("CC3, $m_p = {:.1e}".format(m_p)+"~\mathrm{g}$", fontsize="small")
        elif plot_SLN:
            ax.set_title("SLN, $m_p = {:.1e}".format(m_p)+"~\mathrm{g}$", fontsize="small")
        elif plot_LN:
            ax.set_title("LN, $m_p = {:.1e}".format(m_p)+"~\mathrm{g}$", fontsize="small")
            
        ax.vlines(m_star, 1e-23, 1e-15, color="k", linestyle="dotted")
    
    
    ax0.set_ylim(1e-23, 1e-15)
    ax1.set_ylim(1e-5, 5)

    ax0.set_ylabel("$\psi_\mathrm{N}~[\mathrm{g}]^{-1}$")
    ax1.set_ylabel("$\psi / \psi_\mathrm{max}$")
    
    fig0.tight_layout()
    fig1.tight_layout()
    
    
#%% Plot the mass function for Delta = 0, 2 and 5, showing the mass range relevant
# for the Korwar & Profumo (2023) or Galactic Centre photon constraints.
 
def findroot(f, a, b, args, tolerance=1e-9, n_max=10000):
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

def m_min_finder(m_min, mf_interp, m_pbh_values, integral_total, threshold):
    integral_lower = np.trapz(mf_interp[m_pbh_values<m_min] / f_max[m_pbh_values<m_min], m_pbh_values[m_pbh_values<m_min])
    return integral_lower / integral_total - threshold

def m_max_finder(m_max, mf_interp, m_pbh_values, integral_total, threshold):
    integral_upper = np.trapz(mf_interp[m_pbh_values<m_max] / f_max[m_pbh_values<m_max], m_pbh_values[m_pbh_values<m_max])
    return integral_upper / integral_total - (1-threshold)

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    plot_LN = False
    plot_SLN = False
    plot_CC3 = True
    
    plot_KP23 = True
    plot_GC_Isatis = False
    plot_BC19 = False
    
    # Peak mass (in grams)
    m_p = 2e16
    
    file_string = ""
    
    if plot_KP23:
        file_string = "KP23"

        if plot_LN:
            file_string += "_LN"

            colors = ["k", "lime", "lime", "lime", "lime", "tab:red", "pink"]
            if m_p == 1e15:
                #m_min_values = [5.61e13, 0, 0, 0, 0, 4.65e13, 4.48e13]
                #m_max_values = [1.77e15, 0, 0, 0, 0, 2.58e15, 3.18e15]    
                m_min_values = np.zeros(7)
                m_max_values = np.zeros(7)
            elif m_p == 3e15:
                #m_min_values = [8.72e14, 0, 0, 0, 0, 6.59e13, 4.84e13]
                #m_max_values = [5.09e15, 0, 0, 0, 0, 4.75e15, 3.82e15]
                m_min_values = [2.00e14, 0, 0, 0, 0, 1.26e14, 1.00e14]
                m_max_values = [6.31e15, 0, 0, 0, 0, 1.40e16, 2.98e16]
            elif m_p == 1e16:
                #m_min_values = [3.5e15, 0, 0, 0, 0, 1.45e14, 5e13]
                #m_max_values = [1.7e16, 0, 0, 0, 0, 1.4e16, 5e15]
                m_min_values = [3.98e15, 0, 0, 0, 0, 3.98e14, 1.26e14]
                m_max_values = [2.30e16, 0, 0, 0, 0, 4.23e16, 4.94e16]

            elif m_p == 1e17:
                #m_min_values = [3.2e16, 0, 0, 0, 0, 3.3e15, 6.3e13]
                #m_max_values = [1.2e17, 0, 0, 0, 0, 9e16, 1.2e16]
                m_min_values = np.zeros(7)
                m_max_values = np.zeros(7)

            elif m_p == 2e17:
                #m_min_values = [5.32e16, 0, 0, 0, 0, 6.29e15, 6.7e13]
                #m_max_values = [2.2e17, 0, 0, 0, 0, 1.15e17, 1.7e16]
                m_min_values = [5.18e16, 0, 0, 0, 0, 7.94e15, 2.51e14]
                m_max_values = [2.92e17, 0, 0, 0, 0, 2.18e17, 1.06e17]
                
            else:
                m_min_values = 1e14 * np.ones(7)
                m_max_values = 1e17 * np.ones(7)
              
   
        elif plot_SLN:
            file_string += "_SLN"
            colors = ["grey", "lime", "lime", "lime", "lime", "tab:blue", "deepskyblue"]
            
        elif plot_CC3:
            file_string += "_CC3"

            colors = ["k", "lime", "lime", "lime", "lime", "mediumseagreen", "lime"]
            if m_p == 1e15:
                #m_min_values = [5.3e13, 0, 0, 0, 0, 4.63e13, 2.36e12]
                #m_max_values = [1.645e15, 0, 0, 0, 0, 2.59e15, 9.30e15]                
                m_min_values = np.zeros(7)
                m_max_values = np.zeros(7)
            elif m_p == 3e15:
                #m_min_values = [1.25e14, 0, 0, 0, 0, 6.26e13, 4.83e13]
                #m_max_values = [4.84e15, 0, 0, 0, 0, 4.81e15, 3.85e15]
                m_min_values = np.zeros(7)
                m_max_values = np.zeros(7)
            elif m_p == 1e16:
                #m_min_values = [4e14, 0, 0, 0, 0, 1.1e14, 5e13]
                #m_max_values = [1.6e16, 0, 0, 0, 0, 1.4e16, 5e15]
                m_min_values = [6.31e14, 0, 0, 0, 0, 3.16e14, 1.26e14]
                m_max_values = [1.88e16, 0, 0, 0, 0, 4.04e16, 4.94e16]
            elif m_p == 1e17:
                #m_min_values = [3.6e15, 0, 0, 0, 0, 2.4e15, 6.2e13]
                #m_max_values = [1.15e17, 0, 0, 0, 0, 8.9e16, 1.1e16]
                m_min_values = np.zeros(7)
                m_max_values = np.zeros(7)

            elif m_p == 2e17:
                #m_min_values = [4.78e15, 0, 0, 0, 0, 4.65e15, 6.46e13]
                #m_max_values = [1.46e17, 0, 0, 0, 0, 1.13e17, 1.48e16]
                m_min_values = [7.94e15, 0, 0, 0, 0, 6.31e15, 2.00e14]
                m_max_values = [2.67e17, 0, 0, 0, 0, 2.05e17, 1.02e17]

    elif plot_GC_Isatis:
        
        file_string = "GC"
        
        if plot_LN:
            colors = ["k", "lime", "lime", "lime", "lime", "tab:red", "pink"]
            if m_p == 1e16:
                m_min_values = [2.97e15, 0, 0, 0, 0, 4.24e12, 1.71e12]
                m_max_values = [1.50e16, 0, 0, 0, 0, 2.87e15, 2.285e15]
    
        elif plot_SLN:
            colors = ["grey", "lime", "lime", "lime", "lime", "tab:blue", "deepskyblue"]
            
        elif plot_CC3:
            colors = ["k", "lime", "lime", "lime", "lime", "mediumseagreen", "lime"]
            if m_p == 1e16:
                m_min_values = [1.37e15, 0, 0, 0, 0, 3.7e12, 1.70e12]
                m_max_values = [1.29e16, 0, 0, 0, 0, 2.81e15, 2.28e15]

    elif plot_BC19:
        
        file_string = "BC19"
        
        if plot_LN:
            colors = ["k", "lime", "lime", "lime", "lime", "tab:red", "pink"]
    
        elif plot_SLN:
            colors = ["grey", "lime", "lime", "lime", "lime", "tab:blue", "deepskyblue"]
            
        elif plot_CC3:
            colors = ["k", "lime", "lime", "lime", "lime", "mediumseagreen", "lime"]

    if plot_KP23:
        
        # Korwar & Profumo (2023) 511 keV line constraints
        m_delta_values_loaded, f_max_loaded = load_data("./2302.04408/2302.04408_MW_diffuse_SPI.csv")
        # Power-law exponent to use between 1e15g and 1e16g.
        exponent_PL_upper = 2.0
        # Power-law exponent to use between 1e11g and 1e15g.
        exponent_PL_lower = 2.0
        
        m_delta_extrapolated_upper = np.logspace(15, 16, 11)
        m_delta_extrapolated_lower = np.logspace(11, 15, 41)
        
        f_max_extrapolated_upper = min(f_max_loaded) * np.power(m_delta_extrapolated_upper / min(m_delta_values_loaded), exponent_PL_upper)
        f_max_extrapolated_lower = min(f_max_extrapolated_upper) * np.power(m_delta_extrapolated_lower / min(m_delta_extrapolated_upper), exponent_PL_lower)
    
        m_pbh_values_upper = np.concatenate((m_delta_extrapolated_upper, m_delta_values_loaded))
        f_max_upper = np.concatenate((f_max_extrapolated_upper, f_max_loaded))
        
        f_max = np.concatenate((f_max_extrapolated_lower, f_max_extrapolated_upper, f_max_loaded))
        m_pbh_values = np.concatenate((m_delta_extrapolated_lower, m_delta_extrapolated_upper, m_delta_values_loaded))

    elif plot_GC_Isatis:
        
        # Galactic Centre photon constraints from Isatis
        m_delta_values_loaded = np.logspace(11, 21, 1000)
        constraints_names, f_max_Isatis = load_results_Isatis(modified=True)
        
    elif plot_BC19:
        # Boolean determines which propagation model to load data from
        prop_A = True
        prop_B = not prop_A
        
        with_bkg = True
        
        if prop_A:
            prop_string = "prop_A"
            if with_bkg:
                m_pbh_values, f_max = load_data("1807.03075/1807.03075_prop_A_bkg.csv")
            else:
                m_pbh_values, f_max = load_data("1807.03075/1807.03075_prop_A_nobkg.csv")

        elif prop_B:
            prop_string = "prop_B"
            if with_bkg:
                m_pbh_values, f_max = load_data("1807.03075/1807.03075_prop_B_bkg.csv")
            else:
                m_pbh_values, f_max = load_data("1807.03075/1807.03075_prop_B_nobkg.csv")

    file_string += "_mp={:.0e}g".format(m_p)
    fig0, ax0 = plt.subplots(figsize=(7,7))
    fig1, ax1 = plt.subplots(figsize=(7,7))
    fig2, ax2 = plt.subplots(figsize=(7,7))
    fig3, ax3 = plt.subplots(figsize=(7,7))
    
    for i in [0, 5, 6]:
                
        if plot_GC_Isatis:
            if m_p == 1e16:
                if Deltas[i] == 0:
                    j = 0
                elif Deltas[i] == 2:
                    j = 1
                elif Deltas[i] == 5:
                    j = 1
                
            f_max_loaded = f_max_Isatis[j]   # 0 for COMPTEL, 1 for EGRET, 2 for Fermi-LAT, 3 for INTEGRAL
            m_delta_extrapolated = np.logspace(11, 13, 21)
            
            # Power-law exponent to use between 1e11g and 1e13g.
            exponent_PL_lower = 2.0
                                
            # Extrapolate f_max at masses below 1e13g using a power-law
            f_max_loaded_truncated = np.array(f_max_loaded)[m_delta_values_loaded > 1e13]
            f_max_extrapolated = f_max_loaded_truncated[0] * np.power(m_delta_extrapolated / 1e13, exponent_PL_lower)
            f_max = np.concatenate((f_max_extrapolated, f_max_loaded_truncated))
            m_pbh_values = np.concatenate((m_delta_extrapolated, m_delta_values_loaded[m_delta_values_loaded > 1e13]))

        m_pbh_values_init = np.sort(np.concatenate((np.logspace(np.log10(m_star)-5, np.log10(m_star), 100), np.arange(m_star, m_star*(1+1e-11), 5e2), np.arange(m_star*(1+1e-11), m_star*(1+1e-6), 1e7),  np.logspace(15, 21, 100))))
        
        # Choose mass parameter values for the skew-lognormal corresponding to the peak mass chosen   
        if Deltas[i] == 0:
            mc_SLN = 1.53 * m_p   # for Delta = 0
        elif Deltas[i] == 2:
            mc_SLN = 3.24 * m_p   # for Delta = 2
        elif Deltas[i] == 5:
            mc_SLN = 6.8 * m_p   # for Delta = 5
                        
        mc_LN = m_p * np.exp(+sigmas_LN[i]**2)
        mp_SLN_est = m_max_SLN(mc_SLN, sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=4, n_steps=1000)
        print("m_p (CC3) = {:.2e}".format(m_p))
        print("m_p (SLN) = {:.2e}".format(mp_SLN_est))

        mf_LN_init = LN(m_pbh_values_init, mc_LN, sigma=sigmas_LN[i])
        mf_SLN_init = SLN(m_pbh_values_init, mc_SLN, sigma=sigmas_SLN[i], alpha=alphas_SLN[i])
        mf_CC3_init = CC3(m_pbh_values_init, m_p, alpha=alphas_CC3[i], beta=betas[i])
        
        m_pbh_values_evolved = mass_evolved(m_pbh_values_init, t_0)
        mf_LN_evolved = psi_evolved_normalised(mf_LN_init, m_pbh_values_evolved, m_pbh_values_init)
        mf_SLN_evolved = psi_evolved_normalised(mf_SLN_init, m_pbh_values_evolved, m_pbh_values_init)
        mf_CC3_evolved = psi_evolved_normalised(mf_CC3_init, m_pbh_values_evolved, m_pbh_values_init)
        
        if plot_LN:
            mf_evolved = psi_evolved_normalised(mf_LN_init, m_pbh_values_evolved, m_pbh_values_init)
                        
        elif plot_SLN:
            mf_evolved = psi_evolved_normalised(mf_SLN_init, m_pbh_values_evolved, m_pbh_values_init)
            
        elif plot_CC3:
            mf_evolved = psi_evolved_normalised(mf_CC3_init, m_pbh_values_evolved, m_pbh_values_init)
        
        mf_evolved_interp = 10**np.interp(np.log10(m_pbh_values), np.log10(m_pbh_values_evolved), np.log10(mf_evolved))

        ax0.plot(m_pbh_values_evolved, mf_evolved, color=colors[i], label="${:.0f}$".format(Deltas[i]))  
        ax1.plot(m_pbh_values, 10**np.interp(np.log10(m_pbh_values), np.log10(m_pbh_values_evolved), np.log10(mf_CC3_evolved)) / f_max, color=colors[i], label="${:.0f}$".format(Deltas[i]))
        
        # Plot fractional difference between evolved mass functions, and a log-normal.
        ax3.plot(m_pbh_values_evolved, np.abs(mf_evolved/LN(m_pbh_values_evolved, mc_LN, sigma=sigmas_LN[i]) - 1), color=colors[i])
  
        print("\nEvolved MFs, Delta = {:.0f}".format(Deltas[i]))
        integral_total = np.trapz(mf_evolved_interp / f_max, m_pbh_values)
        print("1 / total integral = {:.2e}".format(1 / integral_total))
        
        threshold = 0.005        
        m_min = findroot(m_min_finder, 1e10, 1e20, args=(mf_evolved_interp, m_pbh_values, integral_total, threshold))
        m_max = findroot(m_max_finder, 1e10, 1e20, args=(mf_evolved_interp, m_pbh_values, integral_total, threshold))

        print("M_min = {:.2e} g".format(m_min))
        print("M_max = {:.2e} g".format(m_max))
        
        integral_lower = np.trapz(mf_evolved_interp[m_pbh_values<m_min] / f_max[m_pbh_values<m_min], m_pbh_values[m_pbh_values<m_min])
        integral_upper = np.trapz(mf_evolved_interp[m_pbh_values<m_max] / f_max[m_pbh_values<m_max], m_pbh_values[m_pbh_values<m_max])

        print("Integral (M < {:.1e}g) / total integral [CC3] = {:.3f}".format(m_min, integral_lower / integral_total))
        print("Integral (M < {:.1e}g) / total integral [CC3] = {:.3f}".format(m_max, integral_upper / integral_total))
        print("1 / total integral [CC3] = {:.2e}".format(1 / integral_total))
       
        if plot_KP23:
            ax2.plot(m_pbh_values, f_max / (min(f_max_loaded) * np.power(m_pbh_values / min(m_delta_values_loaded), exponent_PL_upper)) - 1, color=(0.5294, 0.3546, 0.7020))

        if m_p == 1e15:
            ax0.vlines(m_min, 1e-20, 1e-15, color=colors[i], linestyle="dotted")
            ax0.vlines(m_max, 1e-20, 1e-15, color=colors[i], linestyle="dotted")
            ax1.vlines(m_min, 1e-12, 5e-9, color=colors[i], linestyle="dotted")
            ax1.vlines(m_max, 1e-12, 5e-9, color=colors[i], linestyle="dotted")
            
        elif m_p == 3e15:
            ax0.vlines(m_min, 1e-22, 5e-16, color=colors[i], linestyle="dotted")
            ax0.vlines(m_max, 1e-22, 5e-16, color=colors[i], linestyle="dotted")
            ax1.vlines(m_min, 1e-13, 2e-10, color=colors[i], linestyle="dotted")
            ax1.vlines(m_max, 1e-13, 2e-10, color=colors[i], linestyle="dotted")
    
        elif m_p == 1e16:
            ax0.vlines(m_min, 1e-23, 1e-15, color=colors[i], linestyle="dotted")
            ax0.vlines(m_max, 1e-23, 1e-15, color=colors[i], linestyle="dotted")
            ax1.vlines(m_min, 1e-14, 2e-11, color=colors[i], linestyle="dotted")
            ax1.vlines(m_max, 1e-14, 2e-11, color=colors[i], linestyle="dotted")
   
        elif m_p == 1e17:
            ax0.vlines(m_min, 1e-23, 1e-15, color=colors[i], linestyle="dotted")
            ax0.vlines(m_max, 1e-23, 1e-15, color=colors[i], linestyle="dotted")
            ax1.vlines(m_min, 5e-17, 1e-13, color=colors[i], linestyle="dotted")
            ax1.vlines(m_max, 5e-17, 1e-13, color=colors[i], linestyle="dotted")
 
        elif m_p == 2e17:
            ax0.vlines(m_min, 1e-23, 1e-17, color=colors[i], linestyle="dotted")
            ax0.vlines(m_max, 1e-23, 1e-17, color=colors[i], linestyle="dotted")
            ax1.vlines(m_min, 5e-18, 2e-14, color=colors[i], linestyle="dotted")
            ax1.vlines(m_max, 5e-18, 2e-14, color=colors[i], linestyle="dotted")

        ax3.set_xscale("log")
        ax3.set_yscale("log")
        ax3.legend(fontsize="small")
        ax3.set_xlabel("$m~[\mathrm{g}]$")
        
        if m_p < 2e18:
            ax2.vlines(m_min, 0., 50, color=colors[i], label="{:.0f}".format(Deltas[i]), linestyle="dotted")
            ax2.vlines(m_max, 0., 50, color=colors[i], linestyle="dotted")
            ax3.vlines(m_min, 1e-5, 1e5, color=colors[i], label="{:.0f}".format(Deltas[i]), linestyle="dotted")
            ax3.vlines(m_max, 1e-5, 1e5, color=colors[i], linestyle="dotted")
            #ax3.set_xlim(5e14, 1.1*max(np.array(m_max_values)[np.array(m_min_values)>0.]))

        if plot_CC3:
            ax3.set_title("Initial CC3, $(\Delta={:.0f}, m_p = {:.1e}".format(Deltas[i], m_p)+"~\mathrm{g})$", fontsize="small")
        elif plot_SLN:
            ax3.set_title("Initial SLN, $(\Delta={:.0f}, m_p = {:.1e}".format(Deltas[i], m_p)+"~\mathrm{g})$", fontsize="small")
        elif plot_LN:
            ax3.set_title("Initial LN, $(\Delta={:.0f}, m_p = {:.1e}".format(Deltas[i], m_p)+"~\mathrm{g})$", fontsize="small")
           
    for ax in [ax0, ax1, ax2, ax3]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize="small", title="$\Delta$")
        ax.set_xlabel("$m~[\mathrm{g}]$")
        """
        if m_p < 2e18:
            ax.set_xlim(0.9*min(np.array(m_min_values)[np.array(m_min_values)>0.]), 1.1*max(np.array(m_max_values)[np.array(m_min_values)>0.]))
        """
        if plot_CC3:
            ax.set_title("Initial CC3, $m_p = {:.1e}".format(m_p)+"~\mathrm{g}$", fontsize="small")
        elif plot_SLN:
            ax.set_title("Initial SLN, $m_p = {:.1e}".format(m_p)+"~\mathrm{g}$", fontsize="small")
        elif plot_LN:
            ax.set_title("Initial LN, $m_p = {:.1e}".format(m_p)+"~\mathrm{g}$", fontsize="small")
            
    if m_p == 1e15:
        ax0.set_ylim(1e-20, 1e-15)
        ax1.set_ylim(1e-12, 5e-9)
    elif m_p == 3e15:
        ax0.set_ylim(1e-22, 5e-16)
        ax1.set_ylim(1e-13, 2e-10)    
    elif m_p == 3e15:
        ax0.set_ylim(1e-22, 5e-16)
        ax1.set_ylim(1e-13, 2e-10)    
    elif m_p == 1e16:
        ax0.set_ylim(1e-23, 1e-15)
        ax1.set_ylim(1e-14, 2e-11)    
    elif m_p == 1e17:
        ax0.set_ylim(1e-23, 1e-15)
        ax1.set_ylim(5e-17, 1e-13)    
    elif m_p == 2e17:
        ax0.set_ylim(1e-23, 1e-17)
        ax1.set_ylim(5e-18, 2e-14)
        
    ax2.set_ylim(1e-5, 50)
        
    ax0.set_ylabel("$\psi_\mathrm{N}~[\mathrm{g}^{-1}]$")
    fig0.tight_layout()
    
    ax1.set_ylabel("$\psi_\mathrm{N} / f_\mathrm{max}~[\mathrm{g}^{-1}]$")
    fig1.tight_layout()
    
    ax2.set_ylabel("$f_\mathrm{max} / (km^2) - 1$")
    ax2.set_ylim(1e-6, 2)
    fig2.tight_layout()
    fig2.savefig("./Tests/Figures/EMF_constraints_work/fmax_fracdifff_from_m^2_" + file_string + ".png")
    
    ax3.set_ylim(1e-4, 1e3)
    ax3.set_ylabel("$|\psi_\mathrm{N}(m, t_0) / \psi_\mathrm{LN} - 1|$")
    fig3.tight_layout()
    fig3.savefig("./Tests/Figures/EMF_constraints_work/psi_fracdifff_from_LN_" + file_string + ".png")
    
    ax3.set_xscale("linear")
    fig3.savefig("./Tests/Figures/EMF_constraints_work/psi_fracdifff_from_LN_" + file_string + "_loglin.png")
    

#%% Plot the mass function for Delta = 0, 2 and 5, showing the mass range relevant
# for the Galactic Centre photon constraints from Isatis (1e16g) - ratio plots.
 
if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    plot_LN = True
    plot_SLN = False
    plot_CC3 = False
    
    
    if plot_LN:
        colors = ["grey", "lime", "lime", "lime", "lime", "tab:red", "pink"]
    elif plot_SLN:
        colors = ["grey", "lime", "lime", "lime", "lime", "tab:blue", "deepskyblue"]
    elif plot_CC3:
        colors = ["grey", "lime", "lime", "lime", "lime", "tab:green", "lime"]

    fig0, ax0 = plt.subplots(figsize=(7, 7))
    fig1, ax1 = plt.subplots(figsize=(7, 7))
    
    f_Mi_denominator = []

    m_pbh_values_init = np.sort(np.concatenate((np.logspace(np.log10(m_star)-5, np.log10(m_star), 100), np.arange(m_star, m_star*(1+1e-11), 5e2), np.arange(m_star*(1+1e-11), m_star*(1+1e-6), 1e7),  np.logspace(15, 21, 100))))
                               
    # Choose peak masses corresponding roughly to the maximum mass at which f_PBH < 1 for the CC3 and LN MFs in KP '23
    mc_SLN_D5 = 6.8e16   # for Delta = 5
    mc_SLN_D2 = 3.24e16    # for Delta = 2
        
    m_p = 1e16
    mc_LN_D2 = m_p * np.exp(+sigmas_LN[5]**2)
    mc_LN_D5 = m_p * np.exp(+sigmas_LN[6]**2)
   
    mp_SLN_est_D2 = m_max_SLN(mc_SLN_D2, sigmas_SLN[5], alpha=alphas_SLN[5], log_m_factor=4, n_steps=1000)
    mp_SLN_est_D5 = m_max_SLN(mc_SLN_D5, sigmas_SLN[6], alpha=alphas_SLN[6], log_m_factor=4, n_steps=1000)

    print("m_p (CC3) = {:.2e}".format(m_p))
    print("m_p (SLN, D=2) = {:.2e}".format(mp_SLN_est_D2))
    print("m_p (SLN, D=5) = {:.2e}".format(mp_SLN_est_D5))
    
    m_pbh_values_evolved = mass_evolved(m_pbh_values_init, t_0)

    if plot_LN:
        mf_init_D2 = LN(m_pbh_values_init, mc_LN_D2, sigma=sigmas_LN[5])
        mf_init_D5 = LN(m_pbh_values_init, mc_LN_D5, sigma=sigmas_LN[6])
            
    elif plot_SLN:
        mf_init_D2 = SLN(m_pbh_values_init, mc_SLN_D2, sigma=sigmas_SLN[5], alpha=alphas_SLN[5])
        mf_init_D5 = SLN(m_pbh_values_init, mc_SLN_D5, sigma=sigmas_SLN[6], alpha=alphas_SLN[6])
       
    elif plot_CC3:
        mf_init_D2 = CC3(m_pbh_values_init, m_p, alpha=alphas_CC3[5], beta=betas[5])
        mf_init_D5 = CC3(m_pbh_values_init, m_p, alpha=alphas_CC3[6], beta=betas[6])
 
    ratio_init = mf_init_D5 / mf_init_D2

    mf_evolved_D2 = psi_evolved_normalised(mf_init_D2, m_pbh_values_evolved, m_pbh_values_init)
    mf_evolved_D5 = psi_evolved_normalised(mf_init_D5, m_pbh_values_evolved, m_pbh_values_init)
    
    ratio_evolved = mf_evolved_D5 / mf_evolved_D2
            
    ax0.plot(m_pbh_values_evolved, ratio_evolved, color=colors[-1], label="Evolved \n (plotted against evolved mass)")
    ax0.plot(m_pbh_values_init, ratio_init, color=colors[-2], linestyle="dotted", label="Initial \n (plotted against initial mass)")
    ax0.set_xlabel("$M~[\mathrm{g}]$")
    
    ax1.plot(m_pbh_values_init, ratio_evolved, color=colors[-1], label="Evolved")
    ax1.plot(m_pbh_values_init, ratio_init, color=colors[-2], linestyle="dotted", label="Initial")
    ax1.vlines(m_star, 1e-3, 1e3, color="k", linestyle="dotted")
    ax1.set_xlabel("$M_i~[\mathrm{g}]$")
                
    # Show smallest PBH mass constrained by microlensing.
    for ax in [ax0, ax1]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e13, 1e18)
        ax.set_ylim(1e-3, 1e3)
        if plot_CC3:
            ax.set_title("CC3, $M_p = {:.1e}".format(m_p)+"~\mathrm{g}$", fontsize="small")
        elif plot_SLN:
            ax.set_title("SLN, $M_p = {:.1e}".format(m_p)+"~\mathrm{g}$", fontsize="small")
        elif plot_LN:
            ax.set_title("LN, $M_p = {:.1e}".format(m_p)+"~\mathrm{g}$", fontsize="small")
            
        ax.set_ylabel("$\psi_\mathrm{N}(\Delta=5) / \psi_\mathrm{N}(\Delta=2)$")
        ax.legend(fontsize="x-small")
        ax.grid()
    
    fig0.tight_layout()
    fig1.tight_layout()
    
    fig2, ax2 = plt.subplots(figsize=(8,8))
    #ax2.plot(m_pbh_values_evolved, ratio_init / ratio_evolved)
    ax2.plot(m_pbh_values_init, ratio_init / ratio_evolved)
    ax2.set_xscale("log")
    ax2.set_xlabel("$M~[\mathrm{g}]$")
    ax2.set_ylabel("$[\psi(\Delta=5) / \psi(\Delta=2)]_\mathrm{init} / [\psi(\Delta=5) / \psi(\Delta=2)]$")
    fig2.tight_layout()


#%% Plot the mass function for Delta = 0, 2 and 5, showing the mass range relevant
# for the prospective microlensing constraints (m ~ 1e22g)
 
if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    """
    # for Delta = 0
    i=0
    
    # gives m_p = 1e22g
    mc_SLN = 2.45e20*np.exp(ln_mc_SLN[i])
    m_p = 2.45e20*mp_CC3[i]
    
    # gives m_p = 1e25g
    mc_SLN = 2.45e23*np.exp(ln_mc_SLN[i])
    m_p = 2.45e23*mp_CC3[i]
    """
    
    # for Delta = 2
    """
    i=5
    
    # gives m_p = 1e20g
    mc_SLN = 2.46e18*np.exp(ln_mc_SLN[i])
    m_p = 2.465e18*mp_CC3[i]

    # gives m_p = 1e22g
    #mc_SLN = 2.46e20*np.exp(ln_mc_SLN[i])
    #m_p = 2.465e20*mp_CC3[i]
    
    # gives m_p = 1e25g
    #mc_SLN = 2.46e23*np.exp(ln_mc_SLN[i])
    #m_p = 2.465e23*mp_CC3[i]
    """
    
     
    #for Delta = 5
    i=6
    
    # gives m_p = 1e16g
    mc_SLN = 3.1e14*np.exp(ln_mc_SLN[i])
    m_p = 2.9e14*mp_CC3[i]
 
    # gives m_p = 1e18g
    mc_SLN = 3.1e16*np.exp(ln_mc_SLN[i])
    m_p = 2.9e16*mp_CC3[i]   
    
    #mc_SLN = 3.1e17*np.exp(ln_mc_SLN[i])
    #m_p = 2.9e17*mp_CC3[i]

    #mc_SLN = 3.1e18*np.exp(ln_mc_SLN[i])
    #m_p = 2.9e18*mp_CC3[i]
    
    #mc_SLN = 5.6e23*np.exp(ln_mc_SLN[i])
    #m_p = 5.25e23*mp_CC3[i]
    
           
    m_pbh_values_init = np.sort(np.concatenate((np.arange(m_star, m_star*(1+1e-11), 5e2), np.arange(m_star*(1+1e-11), m_star*(1+1e-6), 1e7),  np.logspace(np.log10(m_p)-3, np.log10(m_p)+3, 100))))
    fig, ax = plt.subplots(figsize=(6, 6))
    mc_LN = m_p * np.exp(+sigmas_LN[i]**2)
    
    mp_SLN_est = m_max_SLN(mc_SLN, sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=4, n_steps=1000)
    print("m_p (CC3) = {:.2e}".format(m_p))
    print("m_p (SLN) = {:.2e}".format(mp_SLN_est))

    mf_LN_init = LN(m_pbh_values_init, mc_LN, sigma=sigmas_LN[i])
    mf_SLN_init = SLN(m_pbh_values_init, mc_SLN, sigma=sigmas_SLN[i], alpha=alphas_SLN[i])
    mf_CC3_init = CC3(m_pbh_values_init, m_p, alpha=alphas_CC3[i], beta=betas[i])
    
    m_pbh_values_evolved = mass_evolved(m_pbh_values_init, t_0)
    mf_LN_evolved = psi_evolved_normalised(mf_LN_init, m_pbh_values_evolved, m_pbh_values_init)
    mf_SLN_evolved = psi_evolved_normalised(mf_SLN_init, m_pbh_values_evolved, m_pbh_values_init)
    mf_CC3_evolved = psi_evolved_normalised(mf_CC3_init, m_pbh_values_evolved, m_pbh_values_init)

    ax.plot(m_pbh_values_evolved, mf_LN_evolved, color="r", label="LN", dashes=[6, 2])            
    ax.plot(m_pbh_values_evolved, mf_SLN_evolved, color="b", label="SLN", linestyle=(0, (5, 7)))
    ax.plot(m_pbh_values_evolved, mf_CC3_evolved, color="g", label="CC3", linestyle="dashed")
    
    # Show smallest PBH mass constrained by microlensing.
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize="small")
    ax.set_xlabel("$m~[\mathrm{g}]$")
    ax.set_xlim(1e13, 1e18)
    ax.set_ylim(1e-23, 1e-17)
    ax.set_title("$\Delta={:.1f},~m_p={:.1e}$".format(Deltas[i], m_p) + "$~\mathrm{g}$", fontsize="small")
    ax.set_ylabel("$\psi_\mathrm{N}$~\mathrm{[g^{-1}]}$")

    fig.tight_layout()


#%% Plot the integrand appearing in Eq. 12 of 1705.05567, for different delta-function MF constraints
 
def extract_GC_Isatis(j, f_max_Isatis, exponent_PL_lower):
    """
    Load delta-function MF constraint on f_PBH from Galactic Centre photons.

    Parameters
    ----------
    j : Integer
        Index for which instrument to load data from (0 for COMPTEL, 1 for EGRET, 2 for Fermi-LAT, 3 for INTEGRAL).
    f_max_Isatis : Array-like
        Array containing constraints on the delta-function MF obtained by Isatis.
    exponent_PL_lower : Float
        Power-law exponent to use between 1e11g and 1e13g.

    Returns
    -------
    f_max : Array-like
        Values of the delta-function mass function constraint.
    m_pbh_values : Array-like
        PBH masses the delta-function mass function constraint is evaluated at, in grams.

    """
    
    m_delta_values_loaded = np.logspace(11, 21, 1000)            
    m_delta_extrapolated = np.logspace(11, 13, 21)
    
    # Set non-physical values of f_max (-1) to 1e100 from the f_max values calculated using Isatis
    f_max_allpositive = []

    for f_max_value in f_max_Isatis[j]:
        if f_max_value == -1:
            f_max_allpositive.append(1e100)
        else:
            f_max_allpositive.append(f_max_value)
            
    # Extrapolate f_max at masses below 1e13g using a power-law
    f_max_loaded_truncated = np.array(f_max_allpositive)[m_delta_values_loaded > 1e13]
    f_max_extrapolated = f_max_loaded_truncated[0] * np.power(m_delta_extrapolated / 1e13, exponent_PL_lower)
    f_max = np.concatenate((f_max_extrapolated, f_max_loaded_truncated))
    
    m_pbh_values = np.concatenate((m_delta_extrapolated, m_delta_values_loaded[m_delta_values_loaded > 1e13]))
    
    return f_max, m_pbh_values


if "__main__" == __name__:
    
    plot_KP23 = False
    plot_GC_Isatis = True
    plot_BC19 = False
    plot_Subaru = False
    plot_Sugiyama19 = False
    Delta = 5
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    # Constraints for monochromatic MF
    
    if plot_Subaru:
        # Subaru-HSC microlensing constraint from Croon et al. (2020) [2007.12697]
        m_pbh_values, f_max = load_data("./2007.12697/Subaru-HSC_2007.12697_dx=5.csv")
    
    
    elif plot_Sugiyama19:
        # Prospective white dwarf microlensing constraint from Sugiyama et al. (2020) [1905.06066]
        m_pbh_values, f_max = load_data("./1905.06066/1905.06066_Fig8_finite+wave.csv")
    
    
    elif plot_KP23:
        # Korwar & Profumo (2023) [2302.04408] constraints
        m_delta_values_loaded, f_max_loaded = load_data("./2302.04408/2302.04408_MW_diffuse_SPI.csv")
        # Power-law exponent to use between 1e15g and 1e16g.
        exponent_PL_upper = 2.0
        # Power-law exponent to use between 1e11g and 1e15g.
        exponent_PL_lower = 2.0
        
        m_delta_extrapolated_upper = np.logspace(15, 16, 11)
        m_delta_extrapolated_lower = np.logspace(11, 15, 41)
        
        f_max_extrapolated_upper = min(f_max_loaded) * np.power(m_delta_extrapolated_upper / min(m_delta_values_loaded), exponent_PL_upper)
        f_max_extrapolated_lower = min(f_max_extrapolated_upper) * np.power(m_delta_extrapolated_lower / min(m_delta_extrapolated_upper), exponent_PL_lower)
    
        m_pbh_values_upper = np.concatenate((m_delta_extrapolated_upper, m_delta_values_loaded))
        f_max_upper = np.concatenate((f_max_extrapolated_upper, f_max_loaded))
        
        f_max = np.concatenate((f_max_extrapolated_lower, f_max_extrapolated_upper, f_max_loaded))
        m_pbh_values = np.concatenate((m_delta_extrapolated_lower, m_delta_extrapolated_upper, m_delta_values_loaded))
        
        #Uncomment to check how the evolved MF integrand would appear if there is no cutoff below 1e15g.
        m_pbh_values_upper = m_pbh_values
        f_max_upper = f_max
        
        
    elif plot_GC_Isatis:
        # Constraints from Galactic Centre photons from Isatis (see Auffinger (2022) [2201.01265])
        
        # Select which instrument places the tightest constraint, for the evolved MF constraints (depends on Delta and the peak mass). Values are for a peak mass m_p = 1e16g
        # 0 for COMPTEL, 1 for EGRET, 2 for Fermi-LAT, 3 for INTEGRAL
        # Note: the defaults here apply to the evolved MF constraints and a peak mass m_p = 1e16g, and do not depend on the fitting function at m_p = 1e16g 
        if Delta == 0:
            j = 0
        elif Delta == 2:
            j = 1
        elif Delta == 5:
            j = 2
        
        constraints_names, f_max_Isatis = load_results_Isatis(modified=True)
        colours_GC_fit = ["tab:orange", "tab:green", "tab:red", "tab:blue"]
        
        f_max, m_pbh_values = extract_GC_Isatis(j, f_max_Isatis, exponent_PL_lower)
        # For the unevolved MF constraints, j=2 when m_p = 1e16g
        f_max_FermiLAT, m_pbh_values_FermiLAT = extract_GC_Isatis(2, f_max_Isatis, exponent_PL_lower)
        
        
    elif plot_BC19:
        # Voyager 1 cosmic ray constraints, from Boudaud & Cirelli (2019) [1807.03075]
        
        # Boolean determines which propagation model to load data from
        prop_A = True
        prop_B = not prop_A
        
        with_bkg = True
        
        if prop_A:
            prop_string = "prop_A"
            if with_bkg:
                m_pbh_values, f_max = load_data("1807.03075/1807.03075_prop_A_bkg.csv")
            else:
                m_pbh_values, f_max = load_data("1807.03075/1807.03075_prop_A_nobkg.csv")

        elif prop_B:
            prop_string = "prop_B"
            if with_bkg:
                m_pbh_values, f_max = load_data("1807.03075/1807.03075_prop_B_bkg.csv")
            else:
                m_pbh_values, f_max = load_data("1807.03075/1807.03075_prop_B_nobkg.csv")
    
    # Peak mass, in grams
    m_p = 1e16
    
    # Choose mass parameter values for the skew-lognormal corresponding to the peak mass chosen   
    if Delta == 0:
        i = 0
        mc_SLN = 1.53 * m_p   # for Delta = 0
    elif Delta == 2:
        i = 5
        mc_SLN = 3.24 * m_p   # for Delta = 2
    elif Delta == 5:
        i = 6
        mc_SLN = 6.8 * m_p   # for Delta = 5
            
    
    mp_SLN = m_max_SLN(mc_SLN, sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=4, n_steps=1000)
    mc_LN = m_p * np.exp(+sigmas_LN[i]**2)
    print("m_p (CC3) = {:.2e}".format(m_p))
    print("m_p (SLN) = {:.2e}".format(mp_SLN))
     
    m_pbh_values_init = np.sort(np.concatenate((np.arange(m_star, m_star*(1+1e-11), 5e2), np.arange(m_star*(1+1e-11), m_star*(1+1e-6), 1e7),  np.logspace(np.log10(m_p)-3, np.log10(m_p)+3, 100))))

    mf_LN_init = LN(m_pbh_values_init, mc_LN, sigma=sigmas_LN[i])
    mf_SLN_init = SLN(m_pbh_values_init, mc_SLN, sigma=sigmas_SLN[i], alpha=alphas_SLN[i])
    mf_CC3_init = CC3(m_pbh_values_init, m_p, alpha=alphas_CC3[i], beta=betas[i])
    
    m_pbh_values_evolved = mass_evolved(m_pbh_values_init, t_0)
    mf_LN_evolved = 10**np.interp(np.log10(m_pbh_values), np.log10(m_pbh_values_evolved), np.log10(psi_evolved_normalised(mf_LN_init, m_pbh_values_evolved, m_pbh_values_init)))
    mf_SLN_evolved = 10**np.interp(np.log10(m_pbh_values), np.log10(m_pbh_values_evolved), np.log10(psi_evolved_normalised(mf_SLN_init, m_pbh_values_evolved, m_pbh_values_init)))
    mf_CC3_evolved = 10**np.interp(np.log10(m_pbh_values), np.log10(m_pbh_values_evolved), np.log10(psi_evolved_normalised(mf_CC3_init, m_pbh_values_evolved, m_pbh_values_init)))

    # Unevolved MF, evaluated at the masses the delta-function MF constraint is evaluated at
    if plot_GC_Isatis and Delta == 5:
        mf_LN_unevolved = LN(m_pbh_values_FermiLAT, mc_LN, sigma=sigmas_LN[i])
        mf_SLN_unevolved = SLN(m_pbh_values_FermiLAT, mc_SLN, sigma=sigmas_SLN[i], alpha=alphas_SLN[i])
        mf_CC3_unevolved = CC3(m_pbh_values_FermiLAT, m_p, alpha=alphas_CC3[i], beta=betas[i])

    else:
        mf_LN_unevolved = LN(m_pbh_values, mc_LN, sigma=sigmas_LN[i])
        mf_SLN_unevolved = SLN(m_pbh_values, mc_SLN, sigma=sigmas_SLN[i], alpha=alphas_SLN[i])
        mf_CC3_unevolved = CC3(m_pbh_values, m_p, alpha=alphas_CC3[i], beta=betas[i])


    fig, ax = plt.subplots(figsize=(6, 6))

    ymin, ymax = 1e-26, 2.5e-23
    xmin, xmax = 9e21, 1e25

    ax.plot(m_pbh_values, mf_SLN_evolved / f_max, color="b", label="SLN", linestyle=(0, (5, 7)))
    ax.plot(m_pbh_values, mf_CC3_evolved / f_max, color="g", label="CC3", linestyle="dashed")
    ax.plot(m_pbh_values, mf_LN_evolved / f_max, color="r", label="LN", dashes=[6, 2])
    
    # Plot the integrand obtained with the unevolved MF
    if plot_KP23:
        ax.plot(m_pbh_values_upper, SLN(m_pbh_values_upper, mc_SLN, sigma=sigmas_SLN[i], alpha=alphas_SLN[i]) / f_max_upper, color="b", linestyle="dotted")
        ax.plot(m_pbh_values_upper, CC3(m_pbh_values_upper, m_p, alpha=alphas_CC3[i], beta=betas[i]) / f_max_upper, color="g", linestyle="dotted")
        ax.plot(m_pbh_values_upper, LN(m_pbh_values_upper, mc_LN, sigma=sigmas_LN[i]) / f_max_upper, color="r", linestyle="dotted")

    if plot_GC_Isatis and Delta == 5:
        ax.plot(m_pbh_values_FermiLAT, mf_SLN_unevolved / f_max_FermiLAT, color="b", linestyle="dotted")
        ax.plot(m_pbh_values_FermiLAT, mf_CC3_unevolved / f_max_FermiLAT, color="g", linestyle="dotted")
        ax.plot(m_pbh_values_FermiLAT, mf_LN_unevolved / f_max_FermiLAT, color="r", linestyle="dotted")

    else:
        ax.plot(m_pbh_values, mf_SLN_unevolved / f_max, color="b", linestyle="dotted")
        ax.plot(m_pbh_values, mf_CC3_unevolved / f_max, color="g", linestyle="dotted")
        ax.plot(m_pbh_values, mf_LN_unevolved / f_max, color="r", linestyle="dotted")
           
    ax.grid()
    ax.legend(fontsize="small")
    ax.set_xlabel("$m~[\mathrm{g}]$")

    if plot_KP23:
        
        if Delta == 0:
            ax.set_xlim(0, 2e16)
            ax.set_ylim(0, 6e-12)
            
        else:
            ax.set_xlim(0, 1e16)
            ax.set_ylim(0, 7e-12)
            
        ax.set_title("KP '23, $\Delta={:.1f},~m_p={:.0e}$".format(Deltas[i], m_p) + "$~\mathrm{g}$", fontsize="x-small")
        psinorm_fit_2 = mf_CC3_evolved[44] * np.power(m_pbh_values/m_pbh_values[44], 2)

    elif plot_GC_Isatis:
        ax.set_xlim(0, 5e15)
        
        if Delta == 0:
            ax.set_xlim(0, 2e16)
           
        if Delta == 2:
            ax.set_ylim(0, 5e-12)

        elif Delta == 5:
            ax.set_ylim(0, 1e-11)
        ax.set_title("GC Isatis constraints, $\Delta={:.1f},~m_p={:.0e}$".format(Deltas[i], m_p) + "$~\mathrm{g}$", fontsize="x-small")

    elif plot_BC19:
        psinorm_fit_2 = mf_CC3_evolved[10] * np.power(m_pbh_values/m_pbh_values[10], 2)

    ax.set_ylabel("$\psi_\mathrm{N} / f_\mathrm{max}~[\mathrm{g}^{-1}]$")
    #ax.set_ylim(ymin, ymax)
    
    fig.set_tight_layout(True)    
    
    # Plot psi_N, f_max and the integrand in the same figure window
    fig, axes = plt.subplots(3, 1, figsize=(5, 14))
    
    psinorm_fit = mf_CC3_evolved[0] * np.power(m_pbh_values/m_pbh_values[0], 3)
    
    for ax in axes.flatten():
        ax.grid('on', linestyle='--')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    axes[0].plot(m_pbh_values, mf_SLN_evolved, color="b", label="SLN", linestyle=(0, (5, 7)))
    axes[0].plot(m_pbh_values, mf_CC3_evolved, color="g", label="CC3", linestyle="dashed")
    axes[0].plot(m_pbh_values, mf_LN_evolved, color="r", label="LN", dashes=[6, 2])
    axes[0].plot(m_pbh_values, mf_SLN_unevolved, color="b", linestyle="dotted", alpha=0.5)
    axes[0].plot(m_pbh_values, mf_CC3_unevolved, color="g", linestyle="dotted", alpha=0.5)
    axes[0].plot(m_pbh_values, mf_LN_unevolved, color="r", linestyle="dotted", alpha=0.5)
    axes[0].plot(m_pbh_values, psinorm_fit, linestyle="dotted", color="k", label="$m^3$ fit")
        
   # if Delta == 5:
   #     axes[0].plot(m_pbh_values, psinorm_fit_2, linestyle="dotted", color="magenta", label="$m^2$ fit")
    
    if m_p > 1e17:
        axes[0].set_ylim(1e-24, 3e-21)
    else:
        axes[0].set_ylim(1e-23, 1e-15)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("$\psi_\mathrm{N}~[\mathrm{g}^{-1}]$")
    
    
    if plot_KP23:
        axes[1].plot(m_pbh_values, 1 / f_max, color=(0.5294, 0.3546, 0.7020))
        if exponent_PL_lower == 3:
            axes[1].text(1.5e15, 1e7,"$\propto m^{-2}$", fontsize="x-small")
            axes[1].text(7e13, 1e11,"$\propto m^{-3}$", fontsize="x-small")
            axes[1].set_ylim(1e3, 1e13)

        elif exponent_PL_lower == 2:
            axes[1].text(2e14, 2e8,"$\propto m^{-2}$", fontsize="x-small")
            axes[1].set_ylim(1e4, 1e12)

        if Delta == 0:
            axes[2].set_ylim(1e-14, 1e-11)

        elif Delta == 2:
            axes[2].set_ylim(1e-13, 1e-11)
            
        elif Delta == 5:
            axes[2].set_ylim(1e-13, 1e-10)

    elif plot_GC_Isatis:
               
        axes[1].plot(m_pbh_values, 1 / f_max, color=colours_GC_fit[j])
        if Delta == 5:
            axes[1].plot(m_pbh_values, 1 / f_max_FermiLAT, color="tab:red", linestyle="dotted", alpha=0.5)

        axes[2].set_ylim(1e-13, 1e-11)
        exponent_fit = 2.8
        fmax_fit = f_max[90] * np.power(m_pbh_values/m_pbh_values[90], 2.8)
        axes[1].plot(m_pbh_values, 1 / fmax_fit, color="k", linestyle="dotted", label="$m^{-2.8}$ fit")
            
        axes[1].legend(fontsize="xx-small")
        axes[1].set_ylim(1, 1e14)
        
        if Delta == 0:
            axes[2].set_ylim(1e-14, 2e-12)        
            
        elif Delta == 5:
            axes[2].set_ylim(1e-14, 1e-8)        
        
    else:
        if with_bkg:
            linestyle = "dashed"
        else:
            linestyle = "solid"
        if prop_A:
            axes[1].plot(m_pbh_values, 1 / f_max, color="b", linestyle=linestyle)
        elif prop_B:
            axes[1].plot(m_pbh_values, 1 / f_max, color="r", linestyle=linestyle)
        
        if Delta == 5:
            PL_fit = (1 / f_max[0]) * np.power(m_pbh_values / m_pbh_values[0], -2)
            axes[1].plot(m_pbh_values, PL_fit, color="k", linestyle="dotted", label="$m^{-2}$ fit")
            axes[1].legend(fontsize="x-small")
            
    if m_p > 1e17:
        if Delta == 5:
            axes[1].set_ylim(1, 1e9)
            axes[2].set_ylim(1e-20, 1e-14)

        elif Delta == 2:
            axes[0].set_ylim(1e-27, 1e-22)
            axes[1].set_ylim(1, 1e9)
            axes[2].set_ylim(1e-25, 1e-20)
        

    axes[1].set_ylabel("$1/f_\mathrm{max}$")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")

    axes[2].plot(m_pbh_values, mf_SLN_evolved / f_max, color="b", label="SLN", linestyle=(0, (5, 7)))
    axes[2].plot(m_pbh_values, mf_CC3_evolved / f_max, color="g", label="CC3", linestyle="dashed")
    axes[2].plot(m_pbh_values, mf_LN_evolved / f_max, color="r", label="LN", dashes=[6, 2])
    
    if plot_GC_Isatis and Delta == 5:
        axes[2].plot(m_pbh_values_FermiLAT, mf_SLN_unevolved / f_max_FermiLAT, color="b", linestyle="dotted")
        axes[2].plot(m_pbh_values_FermiLAT, mf_CC3_unevolved / f_max_FermiLAT, color="g", linestyle="dotted")
        axes[2].plot(m_pbh_values_FermiLAT, mf_LN_unevolved / f_max_FermiLAT, color="r", linestyle="dotted")

    else:
        axes[2].plot(m_pbh_values, mf_SLN_unevolved / f_max, color="b", linestyle="dotted")
        axes[2].plot(m_pbh_values, mf_CC3_unevolved / f_max, color="g", linestyle="dotted")
        axes[2].plot(m_pbh_values, mf_LN_unevolved / f_max, color="r", linestyle="dotted")
    
    axes[2].set_ylabel("$\psi_\mathrm{N} / f_\mathrm{max}~[\mathrm{g}^{-1}]$")
    axes[2].set_xlabel("$m~[\mathrm{g}]$")
    
    if plot_KP23:
        fig.suptitle("KP '23 constraints, $\Delta={:.0f}$".format(Delta))

    elif plot_GC_Isatis:
        fig.suptitle("GC photon constraints, $\Delta={:.0f}$".format(Delta))
        
    elif plot_BC19:
        fig.suptitle("Voyager 1 electron/positron constraints \n $\Delta={:.0f}$".format(Delta), fontsize="small")

    axes[0].legend(fontsize="xx-small")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[2].set_yscale("linear")

    for ax in axes:
        
        ax.set_xlim(1e13, 1e16)
        
        if m_p > 1e18:
            ax.set_xlim(5e14, 1e18)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid('on')
        
        # set x-axis and y-axis ticks
        # see https://stackoverflow.com/questions/30887920/how-to-show-minor-tick-labels-on-log-scale-with-matplotlib
        
        x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
        ax.xaxis.set_major_locator(x_major)
        x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 5)
        ax.xaxis.set_minor_locator(x_minor)
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())


    fig.tight_layout()
    
    # Calculate the contribution to the integral in Eq. 12 of Carr et al. (2017) [1705.05567] (inverse of f_PBH), from different mass ranges.

    # Minimum and maximum masses for which the evolved CC3 MF for Delta = 2 is larger than the evolved Delta = 5 MF, for a peak mass M_p = 1e16g
    #m_min_CC3 = 1.13e15
    
    #m_min = 6.35e14
    m_max = 1e25
    m_min = 1.13e15   # Initial PBH mass that has lost more than 10% of its mass over its lifetime
    #m_max = 7.33e16
    #m_min = 5e14
    #m_max = 2e22
    m_min_CC3 = m_min
    m_max_CC3 = m_max
        
    integral_lower = np.trapz(mf_CC3_evolved[m_pbh_values<m_min_CC3] / f_max[m_pbh_values<m_min_CC3], m_pbh_values[m_pbh_values<m_min_CC3])
    integral_upper = np.trapz(mf_CC3_evolved[m_pbh_values<m_max_CC3] / f_max[m_pbh_values<m_max_CC3], m_pbh_values[m_pbh_values<m_max_CC3])
    integral_total = np.trapz(mf_CC3_evolved / f_max, m_pbh_values)
    print("\n Evolved MFs")
    print("Integral (M < {:.1e}g) / total integral [CC3] = {:.3f}".format(m_min_CC3, integral_lower / integral_total))
    print("Integral ({:.1e}g < M < {:.1e}g) / total integral [CC3] = {:.3f}".format(m_min_CC3, m_max_CC3, (integral_upper - integral_lower) / integral_total))
    print("1 / total integral [CC3] = {:.2e}".format(1 / integral_total))

    integral_lower = np.trapz(mf_SLN_evolved[m_pbh_values<m_min] / f_max[m_pbh_values<m_min], m_pbh_values[m_pbh_values<m_min])
    integral_upper = np.trapz(mf_SLN_evolved[m_pbh_values<m_max] / f_max[m_pbh_values<m_max], m_pbh_values[m_pbh_values<m_max])
    integral_total = np.trapz(mf_SLN_evolved / f_max, m_pbh_values)
    print("\n Evolved MFs")
    print("Integral (M < {:.1e}g) / total integral [SLN] = {:.3f}".format(m_min, integral_lower / integral_total))
    print("Integral ({:.1e}g < M < {:.1e}g) / total integral [SLN] = {:.3f}".format(m_min, m_max, (integral_upper - integral_lower) / integral_total))
    print("1 / total integral [SLN] = {:.2e}".format(1 / integral_total))

    integral_lower = np.trapz(mf_LN_evolved[m_pbh_values<m_min] / f_max[m_pbh_values<m_min], m_pbh_values[m_pbh_values<m_min])
    integral_upper = np.trapz(mf_LN_evolved[m_pbh_values<m_max] / f_max[m_pbh_values<m_max], m_pbh_values[m_pbh_values<m_max])
    integral_total = np.trapz(mf_LN_evolved / f_max, m_pbh_values)
    print("\n Evolved MFs")
    print("Integral (M < {:.1e}g) / total integral [LN] = {:.3f}".format(m_min, integral_lower / integral_total))
    print("Integral ({:.1e}g < M < {:.1e}g) / total integral [LN] = {:.3f}".format(m_min, m_max, (integral_upper - integral_lower) / integral_total))
    print("1 / total integral [LN] = {:.2e}".format(1 / integral_total))
    
    # For initial MFs, restrict the mass range to that loaded from the data itself
    
    if plot_KP23:
        f_max = np.array(f_max_upper)
        m_pbh_values = np.array(m_pbh_values_upper)
    
    if plot_GC_Isatis and Delta == 5:
        f_max = f_max_FermiLAT
        m_pbh_values = m_pbh_values_FermiLAT
    
    m_pbh_values_lower = m_pbh_values[m_pbh_values<m_min]
    m_pbh_values_upper = m_pbh_values[m_pbh_values<m_max]
    integral_lower = np.trapz(CC3(m_pbh_values_lower, m_p, alpha=alphas_CC3[i], beta=betas[i]) / f_max[m_pbh_values<m_min_CC3], m_pbh_values_lower)
    integral_upper = np.trapz(CC3(m_pbh_values_upper, m_p, alpha=alphas_CC3[i], beta=betas[i]) / f_max[m_pbh_values<m_max_CC3], m_pbh_values_upper)
    integral_total = np.trapz(CC3(m_pbh_values, m_p, alpha=alphas_CC3[i], beta=betas[i]) / f_max, m_pbh_values)
    print("\n init MFs")
    print("Integral (M < {:.1e}g) / total integral [CC3] = {:.3f}".format(m_min_CC3, integral_lower / integral_total))
    print("Integral ({:.1e}g < M < {:.1e}g) / total integral [CC3] = {:.3f}".format(m_min_CC3, m_max_CC3, (integral_upper - integral_lower) / integral_total))
    print("1 / total integral [CC3] = {:.2e}".format(1 / integral_total))
    
    m_pbh_values_lower = m_pbh_values[m_pbh_values<m_min]
    m_pbh_values_upper = m_pbh_values[m_pbh_values<m_max]
    integral_lower = np.trapz(SLN(m_pbh_values_lower, mc_SLN, sigma=sigmas_SLN[i], alpha=alphas_SLN[i]) / f_max[m_pbh_values<m_min], m_pbh_values_lower)
    integral_upper = np.trapz(SLN(m_pbh_values_upper, mc_SLN, sigma=sigmas_SLN[i], alpha=alphas_SLN[i]) / f_max[m_pbh_values<m_max], m_pbh_values_upper)
    integral_total = np.trapz(SLN(m_pbh_values, mc_SLN, sigma=sigmas_SLN[i], alpha=alphas_SLN[i]) / f_max, m_pbh_values)
    print("\n init MFs")
    print("Integral (M < {:.1e}g) / total integral [SLN] = {:.3f}".format(m_min, integral_lower / integral_total))
    print("Integral ({:.1e}g < M < {:.1e}g) / total integral [SLN] = {:.3f}".format(m_min, m_max_CC3, (integral_upper - integral_lower) / integral_total))
    print("1 / total integral [SLN] = {:.2e}".format(1 / integral_total))
   
    integral_lower = np.trapz(LN(m_pbh_values_lower, mc_LN, sigma=sigmas_LN[i]) / f_max[m_pbh_values<m_min], m_pbh_values_lower)
    integral_upper = np.trapz(LN(m_pbh_values_upper, mc_LN, sigma=sigmas_LN[i]) / f_max[m_pbh_values<m_max], m_pbh_values_upper)
    integral_total = np.trapz(LN(m_pbh_values, mc_LN, sigma=sigmas_LN[i]) / f_max, m_pbh_values)
    print("\n init MFs")
    print("Integral (M < {:.1e}g) / total integral [LN] = {:.3f}".format(m_min, integral_lower / integral_total))
    print("Integral ({:.1e}g < M < {:.1e}g) / total integral [LN] = {:.3f}".format(m_min, m_max_CC3, (integral_upper - integral_lower) / integral_total))
    print("1 / total integral [LN] = {:.2e}".format(1 / integral_total))
    

    #%% Find mass range that contributes > 10% of the total integral
     
    
def findroot(f, a, b, args, tolerance=1e-9, n_max=10000):
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

def m_min_finder(m_min, m_pbh_values, integral_total, threshold):
    integral_lower = np.trapz(mf[m_pbh_values<m_min] / f_max[m_pbh_values<m_min], m_pbh_values[m_pbh_values<m_min])
    return integral_lower / integral_total - threshold

def m_max_finder(m_max, m_pbh_values, integral_total, threshold):
    integral_upper = np.trapz(mf[m_pbh_values<m_max] / f_max[m_pbh_values<m_max], m_pbh_values[m_pbh_values<m_max])
    return integral_upper / integral_total - (1-threshold)


if "__main__" == __name__:
    
    plot_KP23 = True
    plot_GC_Isatis = False
    plot_BC19 = False
    Delta = 0
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    
    # Constraints for monochromatic MF
    
    # Subaru-HSC microlensing constraint
    #m_pbh_values, f_max = load_data("./2007.12697/Subaru-HSC_2007.12697_dx=5.csv")
    
    # Prospective white dwarf microlensing constraint
    #m_pbh_values, f_max = load_data("./1905.06066/1905.06066_Fig8_finite+wave.csv")
    
    if plot_KP23:
        # Korwar & Profumo (2023) 511 keV line constraints
        m_delta_values_loaded, f_max_loaded = load_data("./2302.04408/2302.04408_MW_diffuse_SPI.csv")
        # Power-law exponent to use between 1e15g and 1e16g.
        exponent_PL_upper = 2.0
        # Power-law exponent to use between 1e11g and 1e15g.
        exponent_PL_lower = 2.0
        
        m_delta_extrapolated_upper = np.logspace(15, 16, 11)
        m_delta_extrapolated_lower = np.logspace(11, 15, 41)
        
        f_max_extrapolated_upper = min(f_max_loaded) * np.power(m_delta_extrapolated_upper / min(m_delta_values_loaded), exponent_PL_upper)
        f_max_extrapolated_lower = min(f_max_extrapolated_upper) * np.power(m_delta_extrapolated_lower / min(m_delta_extrapolated_upper), exponent_PL_lower)
    
        m_pbh_values_upper = np.concatenate((m_delta_extrapolated_upper, m_delta_values_loaded))
        f_max_upper = np.concatenate((f_max_extrapolated_upper, f_max_loaded))
        
        f_max = np.concatenate((f_max_extrapolated_lower, f_max_extrapolated_upper, f_max_loaded))
        m_pbh_values = np.concatenate((m_delta_extrapolated_lower, m_delta_extrapolated_upper, m_delta_values_loaded))
        
        #Uncomment to check how the evolved MF integrand would appear if there is no cutoff below 1e15g.
        m_pbh_values_upper = m_pbh_values
        f_max_upper = f_max
        
    elif plot_GC_Isatis:
        
        if Delta == 0:
            j = 3
        elif Delta == 2:
            j = 1
        elif Delta == 5:
            j = 1
        
        # Galactic Centre photon constraints from Isatis
        m_delta_values_loaded = np.logspace(11, 21, 1000)
        constraints_names, f_max_Isatis = load_results_Isatis(modified=True)
        f_max_loaded = f_max_Isatis[j]   # 0 for COMPTEL, 1 for EGRET, 2 for Fermi-LAT, 3 for INTEGRAL
        f_max_loaded_FermiLAT = f_max_Isatis[2]                              
        
        m_delta_extrapolated = np.logspace(11, 13, 21)
        
        # Power-law exponent to use between 1e11g and 1e13g.
        exponent_PL_lower = 2.0
        
        # Set non-physical values of f_max (-1) to 1e100 from the f_max values calculated using Isatis
        f_max_allpositive = []
        f_max_allpositive_FermiLAT = []


        for f_max_value in f_max_Isatis[j]:
            if f_max_value == -1:
                f_max_allpositive.append(1e100)
            else:
                f_max_allpositive.append(f_max_value)
                
        for f_max_value in f_max_loaded_FermiLAT:
            if f_max_value == -1:
                f_max_allpositive_FermiLAT.append(1e100)
            else:
                f_max_allpositive_FermiLAT.append(f_max_value)
        
        # Extrapolate f_max at masses below 1e13g using a power-law
        f_max_loaded_truncated = np.array(f_max_allpositive)[m_delta_values_loaded > 1e13]
        f_max_extrapolated = f_max_loaded_truncated[0] * np.power(m_delta_extrapolated / 1e13, exponent_PL_lower)
        f_max = np.concatenate((f_max_extrapolated, f_max_loaded_truncated))
        
        # Extrapolate f_max at masses below 1e13g using a power-law
        f_max_FermiLAT_loaded_truncated = np.array(f_max_allpositive_FermiLAT)[m_delta_values_loaded > 1e13]
        f_max_FermiLAT_extrapolated = f_max_FermiLAT_loaded_truncated[0] * np.power(m_delta_extrapolated / 1e13, exponent_PL_lower)
        f_max_FermiLAT = np.concatenate((f_max_FermiLAT_extrapolated, f_max_FermiLAT_loaded_truncated))

        m_pbh_values = np.concatenate((m_delta_extrapolated, m_delta_values_loaded[m_delta_values_loaded > 1e13]))

    elif plot_BC19:
        # Boolean determines which propagation model to load data from
        prop_A = True
        prop_B = not prop_A
        
        with_bkg = True
        
        if prop_A:
            prop_string = "prop_A"
            if with_bkg:
                m_pbh_values, f_max = load_data("1807.03075/1807.03075_prop_A_bkg.csv")
            else:
                m_pbh_values, f_max = load_data("1807.03075/1807.03075_prop_A_nobkg.csv")

        elif prop_B:
            prop_string = "prop_B"
            if with_bkg:
                m_pbh_values, f_max = load_data("1807.03075/1807.03075_prop_B_bkg.csv")
            else:
                m_pbh_values, f_max = load_data("1807.03075/1807.03075_prop_B_nobkg.csv")
        
    # Peak mass, in grams
    m_p = 2e17
    
    # Choose mass parameter values for the skew-lognormal corresponding to the peak mass chosen   
    if Delta == 0:
        i = 0
        mc_SLN = 1.53 * m_p   # for Delta = 0
    elif Delta == 2:
        i = 5
        mc_SLN = 3.24 * m_p   # for Delta = 2
    elif Delta == 5:
        i = 6
        mc_SLN = 6.8 * m_p   # for Delta = 5
            
    mp_SLN = m_max_SLN(mc_SLN, sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=4, n_steps=1000)
    mc_LN = m_p * np.exp(+sigmas_LN[i]**2)
    print("m_p (CC3) = {:.2e}".format(m_p))
    print("m_p (SLN) = {:.2e}".format(mp_SLN))
     
    m_pbh_values_init = np.sort(np.concatenate((np.arange(m_star, m_star*(1+1e-11), 5e2), np.arange(m_star*(1+1e-11), m_star*(1+1e-6), 1e7),  np.logspace(np.log10(m_p)-3, np.log10(m_p)+3, 100))))

    mf_LN_init = LN(m_pbh_values_init, mc_LN, sigma=sigmas_LN[i])
    mf_SLN_init = SLN(m_pbh_values_init, mc_SLN, sigma=sigmas_SLN[i], alpha=alphas_SLN[i])
    mf_CC3_init = CC3(m_pbh_values_init, m_p, alpha=alphas_CC3[i], beta=betas[i])
    
    m_pbh_values_evolved = mass_evolved(m_pbh_values_init, t_0)
    mf_LN_evolved = 10**np.interp(np.log10(m_pbh_values), np.log10(m_pbh_values_evolved), np.log10(psi_evolved_normalised(mf_LN_init, m_pbh_values_evolved, m_pbh_values_init)))
    mf_SLN_evolved = 10**np.interp(np.log10(m_pbh_values), np.log10(m_pbh_values_evolved), np.log10(psi_evolved_normalised(mf_SLN_init, m_pbh_values_evolved, m_pbh_values_init)))
    mf_CC3_evolved = 10**np.interp(np.log10(m_pbh_values), np.log10(m_pbh_values_evolved), np.log10(psi_evolved_normalised(mf_CC3_init, m_pbh_values_evolved, m_pbh_values_init)))
    
    threshold = 0.005
    
    for j in (0,1):

        if j == 0:
            print("\nCC3")
            mf = mf_CC3_evolved
            
        if j == 1:
            print("\nLN")
            mf = mf_LN_evolved
        
                
        integral_total = np.trapz(mf / f_max, m_pbh_values)
        print("Total integral calculated using np.trapz = {:.3e}".format(integral_total))
        
        m_min = findroot(m_min_finder, 1e10, 1e18, args=(m_pbh_values, integral_total, threshold))
        m_max = findroot(m_max_finder, 1e14, 1e20, args=(m_pbh_values, integral_total, threshold))

        print("M_min = {:.2e} g".format(m_min))
        print("M_max = {:.2e} g".format(m_max))
        
        integral_lower = np.trapz(mf[m_pbh_values<m_min] / f_max[m_pbh_values<m_min], m_pbh_values[m_pbh_values<m_min])
        integral_upper = np.trapz(mf[m_pbh_values<m_max] / f_max[m_pbh_values<m_max], m_pbh_values[m_pbh_values<m_max])
        integral_total = np.trapz(mf / f_max, m_pbh_values)
        print("Integral (M < {:.1e}g) / total integral = {:.3f}".format(m_min, integral_lower / integral_total))
        print("Integral (M < {:.1e}g) / total integral = {:.3f}".format(m_max, integral_upper / integral_total))
        print("1 / total integral = {:.2e}".format(1 / integral_total))
    

    #%% Plot the mass functions, for different Delta and peak masses
     
    if "__main__" == __name__:
        
        # Load mass function parameters.
        [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
        
        
        # Constraints for monochromatic MF
        
        # Subaru-HSC microlensing constraint
        #m_pbh_values, f_max = load_data("./2007.12697/Subaru-HSC_2007.12697_dx=5.csv")
        
        # Prospective white dwarf microlensing constraint
        #m_pbh_values, f_max = load_data("./1905.06066/1905.06066_Fig8_finite+wave.csv")
         
        # Korwar & Profumo (2023) 511 keV line constraints
        m_pbh_values, f_max = load_data("./2302.04408/2302.04408_MW_diffuse_SPI.csv")
             
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Choose factors so that peak masses of the CC3 and SLN MF match
        # closely, at 1e20g (consider this range since constraints plots)
        # indicate the constraints from the SLN and CC3 MFs are quite
        # different at this peak mass.
        
        
        # for Delta = 0
        i=0
        
        # gives m_p = 1e16g    
        #mc_SLN = 2.45e14*np.exp(ln_mc_SLN[i])
        #m_p = 2.45e14*mp_CC3[i]
        
        # gives m_p = 1e18g
        mc_SLN = 2.45e16*np.exp(ln_mc_SLN[i])
        m_p = 2.45e16*mp_CC3[i]
        
        # gives m_p = 1e22g
        #mc_SLN = 2.45e20*np.exp(ln_mc_SLN[i])
        #m_p = 2.45e20*mp_CC3[i]
        
        # gives m_p = 1e25g
        #mc_SLN = 2.45e23*np.exp(ln_mc_SLN[i])
        #m_p = 2.45e23*mp_CC3[i]
        
        
        # for Delta = 2
        """
        i=5
        
        # gives m_p = 1e16g
        #mc_SLN = 2.46e14*np.exp(ln_mc_SLN[i])
        #m_p = 2.465e14*mp_CC3[i]

        # gives m_p = 1e18g
        #mc_SLN = 2.46e16*np.exp(ln_mc_SLN[i])
        #m_p = 2.465e16*mp_CC3[i]
        
        # gives m_p = 1e20g
        #mc_SLN = 2.46e18*np.exp(ln_mc_SLN[i])
        #m_p = 2.465e18*mp_CC3[i]

        # gives m_p = 1e22g
        #mc_SLN = 2.46e20*np.exp(ln_mc_SLN[i])
        #m_p = 2.465e20*mp_CC3[i]
        
        # gives m_p = 1e25g
        mc_SLN = 2.46e23*np.exp(ln_mc_SLN[i])
        m_p = 2.465e23*mp_CC3[i]
        """
        
        """ 
        #for Delta = 5
        i=6
        
        mc_SLN = 3.1e17*np.exp(ln_mc_SLN[i])
        m_p = 2.9e17*mp_CC3[i]

        #mc_SLN = 3.1e18*np.exp(ln_mc_SLN[i])
        #m_p = 2.9e18*mp_CC3[i]
        
        #mc_SLN = 5.6e23*np.exp(ln_mc_SLN[i])
        #m_p = 5.25e23*mp_CC3[i]
        """
        
        mc_LN = m_p * np.exp(+sigmas_LN[i]**2)
        mp_SLN_est = m_max_SLN(m_c, sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=4, n_steps=1000)
        print("m_p (CC3) = {:.2e}".format(m_p))
        print("m_p (SLN) = {:.2e}".format(mp_SLN_est))
     
        m_pbh_values_init = np.sort(np.concatenate((np.arange(m_star, m_star*(1+1e-11), 5e2), np.arange(m_star*(1+1e-11), m_star*(1+1e-6), 1e7),  np.logspace(np.log10(m_p)-3, np.log10(m_p)+3, 100))))
    
        mf_LN_init = LN(m_pbh_values_init, mc_LN, sigma=sigmas_LN[i])
        mf_SLN_init = SLN(m_pbh_values_init, mc_SLN, sigma=sigmas_SLN[i], alpha=alphas_SLN[i])
        mf_CC3_init = CC3(m_pbh_values_init, m_p, alpha=alphas_CC3[i], beta=betas[i])
        
        m_pbh_values_evolved = mass_evolved(m_pbh_values_init, t_0)
        mf_LN_evolved = psi_evolved_normalised(mf_LN_init, m_pbh_values_evolved, m_pbh_values_init)
        mf_SLN_evolved = psi_evolved_normalised(mf_SLN_init, m_pbh_values_evolved, m_pbh_values_init)
        mf_CC3_evolved = psi_evolved_normalised(mf_CC3_init, m_pbh_values_evolved, m_pbh_values_init)
        
        ax.plot(m_pbh_values, mf_SLN_evolved, color="b", label="SLN", linestyle=(0, (5, 7)))
        ax.plot(m_pbh_values, mf_CC3_evolved, color="g", label="CC3", linestyle="dashed")
        # Show smallest PBH mass constrained by microlensing.
        #ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid()
        ax.set_xlim(min(m_pbh_values), max(m_pbh_values))
        ax.set_xlabel("$m~[\mathrm{g}]$")
        ax.set_title("$\Delta={:.1f},~m_p={:.0e}$".format(Deltas[i], m_p) + "$~\mathrm{g}$", fontsize="small")

        ax.set_ylabel("$\psi_\mathrm{N}$~\mathrm{[g^{-1}]}$")        
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

#%% Find the integration limits used in the KP '23 extended MF constraints calculations:
if "__main__" == __name__:
    
    from scipy.special import erfinv
        
    n_steps = 1000
    # Load delta function MF constraints calculated using Isatis, to use the method from 1705.05567.
    m_delta_loaded, f_max = load_data("2302.04408/2302.04408_MW_diffuse_SPI.csv")
    m_delta_extrapolated_upper = np.logspace(15, 16, 11)
    m_delta_extrapolated_lower = np.logspace(11, 15, 41)
    m_delta_total = np.concatenate((m_delta_extrapolated_lower, m_delta_extrapolated_upper, m_delta_loaded))
    
    # Find PBH masses at time t
    t = t_0
    m_init_values_input = np.sort(np.concatenate((np.logspace(np.log10(min(m_delta_total)), np.log10(m_star), n_steps), np.arange(m_star, m_star*(1+1e-11), 5e2), np.arange(m_star*(1+1e-11), m_star*(1+1e-6), 1e7), np.logspace(np.log10(m_star*(1+1e-4)), np.log10(max(m_delta_total))+4, n_steps))))
    m_values_input = mass_evolved(m_init_values_input, t)
    
    print("Integration limits:")
    print("Lower limit = {:.2e} g".format(min(m_values_input)))
    print("Lower limit (>0) = {:.2e} g".format(min(m_values_input[m_values_input>0])))
    print("Upper limit = {:.2e} g".format(max(m_values_input)))
    
    # Fractional accuracy to which the result must agree with the calculation with integration limits (0, infty) 
    threshold = 0.99
    # Maximum value of the argument allowed by the error function for the true answer to agree to within the threshold of the answer with integration limits (0, infty)
    max_value = erfinv(threshold)
    
    for i in range(len(Deltas)):
    
        print("\nConstraints on m_p for simplified problem (Delta={:.1f}):".format(Deltas[i]))
        print("m_p <= {:.2e}g".format(min(m_values_input[m_values_input>0]) * np.exp(max_value*np.sqrt(2)*sigmas_LN[i]) * np.exp(sigmas_LN[i]**2)))
        print("m_p >= {:.2e}g".format(max(m_values_input) * np.exp(-max_value*np.sqrt(2)*sigmas_LN[i]) * np.exp(sigmas_LN[i]**2)))


#%% Sanity check: calculate the constraint in the simplified problem using GC_constraint_Carr, and compare to f_max evaluated at that peak mass     
if "__main__" == __name__:    
    m_delta_loaded, f_max_loaded = load_data("2302.04408/2302.04408_MW_diffuse_SPI.csv")
    m_delta_extrapolated_upper = np.logspace(15, 16, 11)
    m_delta_extrapolated_lower = np.logspace(11, 15, 41)
    m_delta_total = np.concatenate((m_delta_extrapolated_lower, m_delta_extrapolated_upper, m_delta_loaded))
    
    # Simplified estimate
    f_max_simplified = min(f_max_loaded) * np.power(m_delta_total / min(m_delta_loaded), 2)
    
    # Full calculation
    exponent_PL_upper = 2
    exponent_PL_lower = 2
    f_max_extrapolated_upper = min(f_max) * np.power(m_delta_extrapolated_upper / min(m_delta_loaded), exponent_PL_upper)
    f_max_extrapolated_lower = min(f_max_extrapolated_upper) * np.power(m_delta_extrapolated_lower / min(m_delta_extrapolated_upper), exponent_PL_lower)
    f_max_total = np.concatenate((f_max_extrapolated_lower, f_max_extrapolated_upper, f_max))

    m_p = 2e17
    f_PBH_simplified = min(f_max_loaded) * np.power(m_p / min(m_delta_loaded), 2)   # Find value of the simplified constraint
    #m_p = min(m_delta_loaded)
    #f_PBH_simplified = min(f_max_loaded)
    
    for i in range(len(Deltas)):
        print("\nDelta={:.1f}:".format(Deltas[i]))
        params = [sigmas_LN[i]]
    
        f_PBH_approx = constraint_Carr([m_p * np.exp(sigmas_LN[i]**2)], m_delta=m_delta_total, f_max=f_max_simplified, psi_initial=LN, params=[sigmas_LN[i]], evolved=True)
        #f_PBH_approx = constraint_Carr([m_p], m_delta=m_delta_total, f_max=f_max_simplified, psi_initial=CC3, params=[alphas_CC3[i], betas[i]], evolved=True)
        print("Fractional difference from simplified problem = {:.3e}".format(f_PBH_approx[0] / f_PBH_simplified - 1))

        f_PBH_full = constraint_Carr([m_p * np.exp(sigmas_LN[i]**2)], m_delta=m_delta_total, f_max=f_max_total, psi_initial=LN, params=[sigmas_LN[i]], evolved=False)
        print("Fractional difference from full problem = {:.3e}".format(f_PBH_full[0] / f_PBH_simplified - 1))
