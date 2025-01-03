#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 15:25:33 2023
@author: ppxmg2
"""

# Script with methods underpinning the rest of the code
# Includes tests and checks

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import erf, loggamma
import os

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
# Results for lognormal from Andrew Gow.
# Results for SLN and CC3 from Table II of 2009.03204.

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
        Directory in which the file is located. The default is 
        "./../Extracted_files/".
    Returns
    -------
    Array-like.
        Contents of file.
    """
    return np.genfromtxt(directory+filename, delimiter=',', unpack=True)


def frac_diff(y1, y2, x1, x2, interp_log = True):
    """
    Find the fractional difference between two arrays (y1, y2), evaluated
    at (x1, x2), of the form (y1/y2 - 1).
    
    In the calculation, interpolation (logarithmic or linear) is used to 
    evaluate the array y2 at x-axis values x1.

    Parameters
    ----------
    y1 : Array-like
        First array.
    y2 : Array-like
        Second array. The length of y2 must be the same as the length of y1.
    x1 : Array-like
        x-axis values that y1 is evaluated at.
    x2 : Array-like
        x-axis values that y2 is evaluated at.
    interp_log : Boolean, optional
        If True, use logarithmic interpolation to evaluate y1 at x2. 
        The default is True.

    Returns
    -------
    Array-like
        Fractional difference between y1 and y2.

    """
    if interp_log:
        return y1 / 10**np.interp(np.log10(x1), np.log10(x2), np.log10(y2)) - 1
    else:
        return np.interp(x1, x2, y2) / y1 - 1


def LN(m, m_c, sigma):
    """
    Log-normal mass function (with characteristic mass m_c and standard 
    deviation sigma), evaluated at m.
    
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
    Skew-lognormal mass function (defined in 2009.03204 Eq. 8), with 
    characteristic mass m_c and parameters sigma and alpha, evaluated at m.
    
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
    Critical collapse 3 mass function (defined in 2009.03204 Eq. 9), with peak 
    mass m_p and parameters alpha and beta, evaluated at m.
    
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


def mf_numeric(m, m_p, Delta, extrap_lower=False, extrap_upper_const=False, normalised=True, n=1, gamma=0.36, custom_mp=True, log_interp=True):
    """
    Estimate the numerical mass function shown in Fig. 5 of 2009.03204 
    evaluated at an arbitrary mass m, with peak mass m_p, using linear 
    interpolation.

    Parameters
    ----------
    m : Array-like
        PBH mass.
    m_p : Float
        Peak mass.
    Delta : Float
        Power spectrum peak width.
    extrap_lower : Boolean, optional
        If True, extrapolate the MF at smaller masses than given in the data
        using the power-law tail m^{1/gamma} from critical collapse. The 
        default is False.
    extrap_upper_const : Boolean, optional
        If True, extrapolate the MF at larger masses than given in the data
        as a constant. The default is False.
    normalised : Boolean
        If True, manually normalise the MF after performing any extrapolations.
        The default is True.
    n : Float, optional
        Number of orders of magnitude in the mass to extrapolate the numeric 
        MF to, above (below) the maximum (minimum) mass for which data is
        availale. The default is 1.
    gamma : Float, optional
        Inverse of the power-law exponent used when extrap_lower = True. 
        The default is 0.36.
    custom_mp : Boolean
        If True, uses the input m_p for the peak mass and rescales the masses
        at which the mass function is evaluated. Otherwise, use the masses
        shown in Fig. 5 of 2009.03204. The default is True.
    log_interp : Boolean, optional
        If True, use logarithmic interpolation to evaluate the MF. If False,
        use linear interpolation. The default is True.
        
    Returns
    -------
    Array-like
        Estimate for the numerical PBH mass function, evaluated at m.

    """
    # Load data from numerical MFs provided by Andrew Gow.
    if Delta < 0.1:
        log_m_data, mf_data_loaded = np.genfromtxt("./Data/psiData/psiData_Delta_k35.txt", unpack=True, skip_header=1)
        # Only include the range of masses for which the numerical MF data has non-zero positive values.
        mf_data = mf_data_loaded[mf_data_loaded >= 0]
        m_data = np.exp(log_m_data[mf_data_loaded >= 0])
    elif Delta < 2:
        log_m_data, mf_data_loaded = np.genfromtxt("./Data/psiData/psiData_Lognormal_D-{:.1f}.txt".format(Delta), unpack=True, skip_header=1)
        # Only include the range of masses for which the numerical MF data has non-zero positive values.
        mf_data = mf_data_loaded[mf_data_loaded >= 0]
        m_data = np.exp(log_m_data[mf_data_loaded >= 0])
    else:
        log_m_data_tabulated, mf_data_tabulated = np.genfromtxt("./Data/psiData/psiData_Lognormal_D-{:.1f}.txt".format(Delta), unpack=True, skip_header=1)
        # For the Delta = 5 case, load the data from Fig. 5, since this covers a wider PBH mass range than that provided by Andrew Gow
        m_data, mf_data_Fig5 = load_data("2009.03204/Delta_{:.1f}_numeric.csv".format(Delta))
        # Rescale the MF data from Fig. 5 (which is scaled by the maximum of the MF) so that its maximum matches the maximum from the data provided by Andrew Gow
        mf_data = mf_data_Fig5 * max(mf_data_tabulated) / max(mf_data_Fig5)
        
    if custom_mp:
        # Find peak mass of the mass function extracted from Fig. 5 of 2009.03204,
        # and scale the masses so that the peak mass corresponds to m_p.
        mp_data = m_data[np.argmax(mf_data)]
        m_data_scaled = m_data * m_p / mp_data
    else:
        m_data_scaled = m_data
        
    # Number of masses at which to extrapolate the numeric MF at for masses above (below) the maximum (minimum) mass for which data on the numeric MF is availale
    n_steps = 100
    # Find the total range of masses at which to evaluate the numeric MF (to calculate the normalisation)
    m_lower = min(m_data_scaled) * np.logspace(-n, 0, n_steps)
    m_upper = max(m_data_scaled) * np.logspace(0, n, n_steps)
    m_total = np.concatenate((m_lower, m_data_scaled, m_upper))
        
    # Extrapolate the numeric MF to masses below and above that where data is available, if required
    if extrap_lower:
        mf_values_lower = mf_data[0] * np.power(m_lower/min(m_data_scaled), 1/gamma)
    else:
        mf_values_lower = np.zeros(len(m_lower))
        
    if extrap_upper_const:
        mf_values_upper = mf_data[-1] * np.ones(len(m_upper))
    else:
        mf_values_upper = np.zeros(len(m_upper))
    
    # Calculate integral of the numeric MF over all masses, after performing extrapolation
    mf_values_total = np.concatenate((mf_values_lower, mf_data, mf_values_upper))
    
    normalisation_factor = 1/np.trapz(mf_values_total[mf_values_total>0], m_total[mf_values_total>0])

    # Interpolate the mass function at the masses required
    if log_interp == False:
        mf_values_interp = 10**np.interp(np.log10(m), np.log10(m_total), np.log10(mf_values_total), -np.infty, -np.infty)
    else:
        mf_values_interp = np.interp(m, m_total, mf_values_total, 0, 0)
      
    if not normalised:
        return mf_values_interp
    else:
        return mf_values_interp * normalisation_factor


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


def m_max_SLN(m_c, sigma, alpha, log_m_factor=4, n_steps=1000):
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
        Number of multiples of sigma (in log-space) of masses around m_c to 
        consider when estimating the maximum. The default is 4.
    n_steps : Integer, optional
        Number of masses to use for estimating the peak mass of the skew-
        lognormal mass function. The default is 1000.
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
        Evolved values of the PBH mass density distribution function (not 
        normalised to unity).

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
        Evolved values of the PBH mass density distribution function 
        (normalised to unity).

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
    	Masses at which constraints for a delta-function PBH mass function are 
        evaluated.
    f_max : Array-like
    	Constraints obtained for a monochromatic mass function.
    psi_initial : Function
    	Initial PBH mass function (in terms of the mass density).
    params : Array-like
    	Parameters of the PBH mass function.
    evolved : Boolean
    	If True, calculate constraints using the evolved PBH mass function. The 
        default is True.
    t : Float
    	If evolved == True, the time (after PBH formation) at which to evaluate 
        PBH mass function. The default is t_0 (the present age of the Universe).
        
    Returns
    -------
    f_pbh : Array-like
        Constraints on f_PBH.
    
    """
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
            integral = np.trapz((psi_initial(m_delta, m_c, *params) / f_max), m_delta)
            
        if integral == 0 or np.isnan(integral):
            f_pbh.append(10)
        else:
            f_pbh.append(1/integral)
            
    return f_pbh


#%% Tests of the method frac_diff:

if "__main__" == __name__:

    # Constant fractional difference
    y1, y2 = 1.5*np.ones(10), np.ones(10)
    x1 = x2 = np.linspace(0, 10, 10)
    print(frac_diff(y1, y2, x1, x2))

    y1, y2 = 1.5*np.ones(10), np.ones(30)
    x1 = np.linspace(0, 10, 10)
    x2 = np.linspace(-10, 20, 30)
    print(frac_diff(y1, y2, x1, x2))

    # x-dependent fractional difference
    x1 = x2 = np.linspace(1, 10, 10)
    y1 = 1 + x1**2
    y2 = np.ones(10)
    print(frac_diff(y1, y2, x1, x2))

#%% Test of the methods SLN and CC3 by comparing to Fig. 5 of 2009.03204.

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
        psi_SLN_max = max(SLN(m_pbh_values, np.exp(ln_mc_SLN[Delta_index]), sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index]))
        psi_CC3_max = max(CC3(m_pbh_values, mp_CC3[Delta_index], alpha=alphas_CC3[Delta_index], beta=betas[Delta_index]))
        
        
        # Plot the mass function.
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.plot(np.exp(log_m_numeric), psi_numeric / max(psi_numeric), color="k", label="Numeric", linewidth=2)
        ax.plot(m_loaded_SLN, psi_scaled_SLN, color="b", linestyle="None", marker="x")
        ax.plot(m_loaded_CC3, psi_scaled_CC3, color="tab:green", linestyle="None", marker="x")
        ax.plot(m_pbh_values, psi_SLN / psi_SLN_max, color="b", label="SLN", linewidth=2)
        
        if Delta_index == 6:
            ax.plot(m_pbh_values, psi_CC3 / psi_CC3_max, color="tab:green", label="CC3", linewidth=2)
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


#%% Test of m_max_SLN. Compare peak mass of the skew lognormal calculated with different mass ranges and numbers of masses tested.

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
        print("\nBest estimate = {:.4e}".format(mp_best_estimate))
        
        stop_loop = False
        
        for n_steps in n_steps_range:
            
            for n_sigma in n_sigmas_range:
                
                # Estimated peak mass of the SLN mass function.
                m_max_SLN_est = m_max_SLN(m_c, sigma, alpha, log_m_factor=n_sigma, n_steps=int(n_steps))
                print("Estimate = {:.4e}".format(m_max_SLN_est))
                frac_diff_val = abs((m_max_SLN_est - mp_best_estimate) / mp_best_estimate)
                
                if frac_diff_val < precision:
                    
                    n_steps_min[i] = n_steps
                    n_sigma_min[i] = n_sigma
                    
                    # Break loop, to give the minimum number of steps required and minimum range for a given precision of the M_p calculation.
                    stop_loop = True
                    break
                
                if stop_loop:
                    break
                    
            if stop_loop:
                break

        print("\nDelta = {:.1f}".format(Deltas[i]))
        print("n_steps_min = {:.2e}".format(n_steps_min[i]))
        print("n_sigmas_min = {:.0f}".format(n_sigma_min[i]))
        
#%% Test of m_max_SLN(). Compare peak mass to Table II of 2009.03204, accounting for the uncertainty in parameters due to the limited precision given.

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    # Peak mass of the skew lognormal mass function, from Table II of 2009.03204.
    mp_SL = [40.9, 40.9, 40.9, 40.8, 40.8, 40.6, 32.9]
        
    # Cycle through range of number of masses to use in the estimate
    for i in range(len(Deltas)):
        print("\nDelta = {:.1f}".format(Deltas[i]))
        
        # Account for uncertainty due to the limited precision of values given in Table II of 2009.03204.
        for ln_mc in np.linspace(ln_mc_SLN[i]-0.005, ln_mc_SLN[i]+0.005, 10):
            
            for alpha in np.linspace(alphas_SLN[i]-0.005, alphas_SLN[i]+0.005, 5):
                
                for sigma in np.linspace(sigmas_SLN[i]-0.005, sigmas_SLN[i]+0.005, 5):
                    
                    #print((m_max_SLN_est - mp_SL[i]))
                    
                    # Estimated peak mass of the SLN mass function.
                    m_max_SLN_est = m_max_SLN(np.exp(ln_mc), sigma, alpha, log_m_factor=n_sigma_min[i], n_steps=int(n_steps_min[i]))
                    
                    # Compare to peak mass given in Table II of 2009.03204
                    if abs(m_max_SLN_est - mp_SL[i]) < 0.05:
                        
                        print("Success")
                        
                        # Calculate and print fractional difference
                        frac_diff_val = abs((m_max_SLN_est - mp_SL[i]) / mp_SL[i])
                        print("Fractional difference = {:.2e}".format(frac_diff_val))
                        
#%% Plot the CC3 mass function, and compare to the limits at small and large (m / m_p) [m_p = peak mass]

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)    
    # Peak mass
    
    filepath = "./Data/psiData/"
    
    for i in range(len(Deltas[0:])):
        
        fig, ax = plt.subplots(figsize=(7, 7))
        
        m_p=mp_CC3[i]
        
        if i == 0:
            log_m_pbh_values, mf_values = np.genfromtxt(filepath + "psiData_Delta_k35.txt", unpack=True, skip_header=1)
            m_pbh_values = np.exp(log_m_pbh_values)
        elif i >= 5:
            log_m_pbh_values_data, mf_values_data = np.genfromtxt(filepath + "psiData_Lognormal_D-{:.1f}.txt".format(Deltas[i]), unpack=True, skip_header=1)
            # For the Delta = 2 and Delta = 5 cases, load the data from Fig. 5, since this covers a wider PBH mass range than that provided by Andrew Gow
            m_pbh_values, mf_values_Fig5 = load_data("2009.03204/Delta_{:.1f}_numeric.csv".format(Deltas[i]))
            # Rescale the MF data from Fig. 5 (which is scaled by the maximum of the MF) so that its maximum matches the maximum from the data provided by Andrew Gow
            mf_values = mf_values_Fig5 * max(mf_values_data) / max(mf_values_Fig5)
        else:
            log_m_pbh_values, mf_values = np.genfromtxt(filepath + "psiData_Lognormal_D-{:.1f}.txt".format(Deltas[i]), unpack=True, skip_header=1)
            m_pbh_values = np.exp(log_m_pbh_values)        

        ax.plot(m_pbh_values, mf_values, color="k", label="Numeric")
        
        alpha = alphas_CC3[i]
        beta = betas[i]
    
        m_pbh_values = np.logspace(np.log10(m_p) - 3, np.log10(m_p) + 3, 1000)
        mf_CC3_exact = CC3(m_pbh_values, m_p, alpha=alphas_CC3[i], beta=betas[i])
        
        m_f = m_p * np.power(beta/alpha, 1/beta)
        
        log_mf_approx_lowmass = np.log(beta/m_f) - loggamma((alpha+1) / beta) + (alpha * np.log(m_pbh_values/m_f))
        mf_CC3_approx_lowmass = np.exp(log_mf_approx_lowmass)
        
        log_mf_approx_highmass = np.log(beta/m_f) - loggamma((alpha+1) / beta)- np.power(m_pbh_values/m_f, beta)
        mf_CC3_approx_highmass = np.exp(log_mf_approx_highmass)
        
        ax.plot(m_pbh_values, mf_CC3_exact, label="GCC")
        ax.plot(m_pbh_values, mf_CC3_approx_lowmass, label=r"$\propto (m/m_p)^\alpha$", linestyle="dotted")  
        ax.plot(m_pbh_values, mf_CC3_approx_highmass, label=r"$\propto \exp\left(-\frac{\alpha}{\beta}\frac{m}{m_p}\right)$", linestyle="dotted")  

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize="small", title="$\Delta={:.1f}$".format(Deltas[i]))
        ax.set_xlabel("$m~[M_\odot]$")
        ax.set_xlim(min(m_pbh_values), max(m_pbh_values)/30)
        ax.set_ylim(mf_CC3_exact[0]/100, 3*max(mf_CC3_exact))
        ax.set_ylabel("$\psi~[M_\odot^{-1}]$")
        fig.tight_layout()
                        
        
#%% Plot the SLN at small masses, compare to (m/m_p)^|alpha|.

if "__main__" == __name__:
    
    # Mass function parameter values, from 2009.03204.
    [Deltas, sigmas_LN, ln_mc_SL, mp_SL, sigmas_SLN, alphas_SLN, mp_CC, alphas_CC, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)

    for Delta_index in range(len(Deltas)):
        
        print(Delta_index)
        
        fig, ax = plt.subplots(figsize=(6,6))
        m_c = 1e20
        m_p = m_max_SLN(m_c, sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index], log_m_factor=3, n_steps=1000)
        m_pbh_values = np.logspace(np.log10(m_p)-5, np.log10(m_p)+5, 1000)
        
        SLN_actual = SLN(m_pbh_values, m_c, sigmas_SLN[Delta_index], alphas_SLN[Delta_index])
        SLN_fit_alpha_power = SLN_actual[0] * np.power(m_pbh_values/m_pbh_values[0], np.abs(alphas_SLN[Delta_index]))
        
        ax.plot(m_pbh_values[SLN_actual > 0], SLN_actual[SLN_actual > 0])
        ax.plot(m_pbh_values, SLN_fit_alpha_power, linestyle="dotted", label=r"$\propto (m / m_\mathrm{p})^{|\alpha|}$")
        ax.set_xlim(min(m_pbh_values), max(m_pbh_values))
        ax.set_ylim(SLN_actual[0], 10*max(SLN_actual))
        ax.set_title("SLN, $\Delta={:.1f}$".format(Deltas[Delta_index]))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$m~[\mathrm{g}]$")
        ax.set_ylabel("$\psi~[\mathrm{g}^{-1}]$")
        ax.legend()
        fig.tight_layout()

                        
#%% Plot the fitting functions for Delta = 5.0, showing the behaviour deep into the high-mass tail.
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
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.grid()
                ax.legend(fontsize="small")
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
           
            
#%% Reproduce Fig. 1 of Mosbech & Picker (2022), using different forms of alpha_eff.

if "__main__" == __name__:
    
    def alpha_eff(tau, M_0):
        """
        Calculate alpha_eff from BlackHawk output files.

        Parameters
        ----------
        tau : Array-like
            PBH lifetimes, in seconds.
        M_0 : Array-like
            Initial PBH masses, in grams.

        Returns
        -------
        Array-like
            Values of alpha_eff.

        """
        return (1/3) * (t_Pl/tau) * (M_0 / m_Pl)**3
    
    m_pbh_values_formation_BlackHawk = np.logspace(np.log10(4e14), 16, 50)   # Range of formation masses for which the lifetime is calculated using BlackHawk (primary particle energy range from 1e-6 to 1e5 GeV)
    m_pbh_values_formation_wide = np.logspace(8, 18, 100)    # Test range of initial masses
    pbh_lifetimes = []
    M0_min_BH, M0_max_BH = min(m_pbh_values_formation_BlackHawk), max(m_pbh_values_formation_BlackHawk)
    
    # Find PBH lifetimes corresponding to PBH formation masses in m_pbh_values_formation_BlackHawk
    for j in range(len(m_pbh_values_formation_BlackHawk)):
        
        destination_folder = "mass_evolution_v2" + "_{:.0f}".format(j+1)
        filename = os.path.expanduser('~') + "/Downloads/version_finale/results/" + destination_folder + "/life_evolutions.txt"
        data = np.genfromtxt(filename, delimiter="    ", skip_header=4, unpack=True, dtype='str')
        times = data[0]
        tau = float(times[-1])
        pbh_lifetimes.append(tau)   # add the last time value at which BlackHawk calculates the PBH mass, corresponding to the PBH lifetime (when estimated PBH mass ~= Planck mass [see paragraph after Eq. 43b in manual, line 490 of evolution.c])
    
    # Values of alpha_eff calculated directly using BlackHawk
    alpha_eff_values_BlackHawk = alpha_eff(np.array(pbh_lifetimes), m_pbh_values_formation_BlackHawk)
    # Values of alpha_eff extracted from Fig. 1 of Mosbech & Picker (2022)
    alpha_eff_extracted_values = alpha_eff_extracted(m_pbh_values_formation_wide)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(m_pbh_values_formation_BlackHawk, alpha_eff_values_BlackHawk, label="Calculated using BlackHawk")
    ax.plot(m_pbh_values_formation_wide, alpha_eff_extracted_values, linestyle="None", marker="x", label="Extracted (Fig. 1 MP '22)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Formation mass $M_{\rm i} \, [{\rm g}$]")
    ax.set_ylabel(r"$\alpha_{\rm eff}$")
    ax.legend(fontsize="xx-small")
    fig.tight_layout()
    
  
#%% Reproduce Fig. 2 of Mosbech & Picker (2022)

if "__main__" == __name__:
    
    m_pbh_values_formation = np.logspace(11, 17, 500)
    # Initial PBH mass values includes values close to the initial mass of a PBH with lifetime equal to the age of the Universe,
    # corresponding to evolved masses at t=t_0 as low as a few times 10^11 g.
    #m_pbh_values_formation_to_evolve = np.concatenate((np.arange(7.473420349255e+14, 7.4734203494e+14, 5e2), np.arange(7.4734203494e+14, 7.47344e+14, 1e7), np.logspace(np.log10(7.474e14), 17, 500)))
    m_pbh_values_formation_to_evolve = np.concatenate((np.arange(m_star, m_star*(1+1e-11), 5e2), np.arange(m_star*(1+1e-11), m_star*(1+1e-6), 1e7), np.logspace(np.log10(m_star*(1+1e-4)), 17, 500)))

    m_pbh_values_evolved = mass_evolved(m_pbh_values_formation_to_evolve, t_0)
    m_pbh_t_init = mass_evolved(m_pbh_values_formation_to_evolve, 0)

    m_c = 1e15
    
    for sigma in [0.1, 0.5, 1, 1.5]:
        
        if sigma < 1:
           m_extracted_evolved, phi_extracted_evolved = load_data("2203.05743/2203.05743_Fig2_sigma0p{:.0f}.csv".format(10*sigma))
        else:
           m_extracted_evolved, phi_extracted_evolved = load_data("2203.05743/2203.05743_Fig2_sigma1p{:.0f}.csv".format(10*(sigma-1)))

        
        phi_initial = LN(m_pbh_values_formation, m_c, sigma)
        phi_initial_to_evolve = LN(m_pbh_values_formation_to_evolve, m_c, sigma)
                
        phi_present = psi_evolved(phi_initial_to_evolve, m_pbh_values_evolved, m_pbh_values_formation_to_evolve) / (m_pbh_values_evolved / m_pbh_values_formation_to_evolve)
        # test that the "evolved" mass function at t=0 matches the initial mass function.
        phi_test = psi_evolved(phi_initial_to_evolve, m_pbh_t_init, m_pbh_values_formation_to_evolve) / (m_pbh_t_init / m_pbh_values_formation_to_evolve)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(m_pbh_values_formation, phi_initial, label="$t=0$")
        ax.plot(m_pbh_values_evolved, phi_present, label="$t=t_0$", marker="x")
        ax.plot(m_pbh_values_formation_to_evolve, phi_test, label="$t=0$ (test)")
        ax.plot(m_extracted_evolved, phi_extracted_evolved, linestyle="None", marker="+", label="$t=t_0$ (extracted)")

        ax.set_xlabel("$M~[\mathrm{g}]$")
        ax.set_ylabel("$\phi(M)~[\mathrm{g}]^{-1}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(title="$\sigma={:.1f}$".format(sigma), fontsize="xx-small")
        ax.set_xlim(1e11, max(m_pbh_values_formation))
        ax.set_ylim(1e-21, 1e-12)
        fig.tight_layout()

        fig, ax = plt.subplots(figsize=(6, 6))
        phi_extracted_evolved_interp = 10**np.interp(np.log10(m_pbh_values_evolved), np.log10(m_extracted_evolved), np.log10(phi_extracted_evolved))
        ratio = phi_present/phi_extracted_evolved_interp
        ax.plot(m_pbh_values_evolved, ratio-1, marker="x", linestyle="None")
        ax.set_xlabel("$M~[\mathrm{g}]$")
        ax.set_ylabel("$\phi(M, t)$ (reproduction / extracted - 1)")
        ax.set_xscale("log")
        ax.set_title("$\sigma={:.1f}$".format(sigma))
        ax.set_xlim(min(m_extracted_evolved), max(m_extracted_evolved))
        ax.set_ylim(-0.2, 0.2)
        fig.tight_layout()

                       
#%% Plot the mass function for Delta = 0, 2 and 5, showing the evovled and unevolved MFs for a peak mass m_p = 1e16g. Plots show different fitting functions for the same Delta.
if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    for i in range(len(Deltas)):
        
        if i in (0, 5, 6):
           
            m_pbh_values_init = np.sort(np.concatenate((np.logspace(np.log10(m_star)-5, np.log10(m_star), 100), np.arange(m_star, m_star*(1+1e-11), 5e2), np.arange(m_star*(1+1e-11), m_star*(1+1e-6), 1e7),  np.logspace(15, 21, 100))))
            fig, ax = plt.subplots(figsize=(6, 6))
            
            if Deltas[i] == 5:
                mc_SLN = 6.8e16
            elif Deltas[i] == 2:
                mc_SLN = 3.24e16
            elif Deltas[i] == 0:
                mc_SLN = 1.53e16
                
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
            ax.set_ylim(1e-23, 5e-16)
            ax.set_title("$\Delta={:.1f},~m_p={:.1e}$".format(Deltas[i], m_p) + "$~\mathrm{g}$", fontsize="small")
            ax.set_ylabel("$\psi_\mathrm{N}$")

            fig.tight_layout()
            

#%% Plot the mass function for Delta = 0, 2 and 5, showing the mass range relevant for the Galactic Centre photon constraints from Isatis (1e16g). Plots show the same fitting functions for different Delta.
 
if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, has_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    plot_LN = False
    plot_SLN = False
    plot_CC3 = True
    
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
    
    
#%% Plot the mass function for Delta = 0, 2 and 5, showing the ratio of the evolved and unevolved MFs.
 
if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    plot_LN = False
    plot_SLN = True
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


#%% Plot the evolved mass function for Delta = 0, 2 and 5, showing the mass range relevant for the prospective microlensing constraints (m ~ 1e22g)
 
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
    
    
    
    #for Delta = 5
    i=6
    
    # gives m_p = 1e16g
    #mc_SLN = 3.1e14*np.exp(ln_mc_SLN[i])
    #m_p = 2.9e14*mp_CC3[i]
 
    # gives m_p = 1e18g
    #mc_SLN = 3.1e16*np.exp(ln_mc_SLN[i])
    #m_p = 2.9e16*mp_CC3[i]   
    
    # gives m_p = 1e19g
    #mc_SLN = 3.1e17*np.exp(ln_mc_SLN[i])
    #m_p = 2.9e17*mp_CC3[i]

    # gives m_p = 1e22g
    mc_SLN = 3.1e20*np.exp(ln_mc_SLN[i])
    m_p = 2.9e20*mp_CC3[i]
 
    # gives m_p = 2e25g
    #mc_SLN = 5.6e23*np.exp(ln_mc_SLN[i])
    #m_p = 5.25e23*mp_CC3[i]
    
           
    m_pbh_values_init = np.sort(np.concatenate((np.arange(m_star, m_star*(1+1e-11), 5e2), np.arange(m_star*(1+1e-11), m_star*(1+1e-6), 1e7),  np.logspace(np.log10(m_p)-7, np.log10(m_p)+7, 1000))))
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
    ax.set_xlim(1e14, 1e23)
    ax.set_ylim(1e-27, 1e-19)
    ax.set_title("$\Delta={:.1f},~m_p={:.1e}$".format(Deltas[i], m_p) + "$~\mathrm{g}$", fontsize="small")
    ax.set_ylabel("$\psi_\mathrm{N}~\mathrm{[g^{-1}]}$")

    fig.tight_layout()


#%% Find characteristic mass for which the minimum mass to include in a calculation is larger than 1e21g, the maximum mass for which I have calculated using isatis_reproduction.py.

if "__main__" == __name__:
    
    m_sig = 1e22  # PBH mass above which results have not been calculated using isatis_reproduction.py
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
            
            
#%% Test evolved MF calculation methods
if "__main__" == __name__:   
    # Maximum mass that the Korwar & Profumo (2023) delta-function MF constraint is calculated at
    m_delta_max_KP23 = 3e17
    # Initial PBH mass values includes values close to the initial mass of a PBH with lifetime equal to the age of the Universe,
    # corresponding to evolved masses at t=t_0 as low as a few times 10^11 g.
    m_pbh_values_formation = np.concatenate((np.logspace(np.log10(m_star) - 3, np.log10(m_star)), np.arange(m_star, m_star*(1+1e-11), 5e2), np.arange(m_star*(1+1e-11), m_star*(1+1e-6), 1e7), np.logspace(np.log10(m_star*(1+1e-4)), np.log10(m_delta_max_KP23)+4, 1000)))
    m_pbh_values_evolved = mass_evolved(m_pbh_values_formation, t_0)
    m_pbh_values_evolved_t_zero = mass_evolved(m_pbh_values_formation, 0)
    m_c = 1e17
    
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)

    plot_LN = False
    plot_SLN = True
    plot_CC3 = False

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

    
#%% Plot the initial PBH mass against the evolved mass, using Eq. 7 of Mosbech & Picker (2022) [2203.05743].
if "__main__" == __name__:
    
    n_steps = 1000
    m_delta = [1e14, 1e16]
    m_init_values = np.sort(np.concatenate((np.logspace(np.log10(min(m_delta)), np.log10(m_star), n_steps), np.arange(m_star, m_star*(1+1e-11), 5e2), np.arange(m_star*(1+1e-11), m_star*(1+1e-6), 1e7), np.logspace(np.log10(m_star*(1+1e-4)), np.log10(max(m_delta))+4, n_steps))))
    m_evolved_values = mass_evolved(m_init_values, t=t_0)
    
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(m_init_values, m_evolved_values)
    ax.plot(m_init_values, m_init_values, linestyle="dotted", color="k")
    ax.set_xlabel("$m_i~[\mathrm{g}]$")
    ax.set_ylabel("$m(t=t_0)~[\mathrm{g}]$")
    ax.set_xlim(min(m_init_values), max(m_init_values))
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.tight_layout()
    
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(m_init_values, m_evolved_values/m_init_values - 1, marker="x", linestyle="None")
    ax.set_xlabel("$m_i~[\mathrm{g}]$")
    ax.set_ylabel("$m(t_0)/m_i - 1$")
    ax.set_xlim(min(m_init_values), max(m_init_values))
    ax.set_xscale("log")
    fig.tight_layout()
    
    
#%% Find the slope of the numerical MFs obtained by Andrew Gow at masses much smaller than the peak mass. Other tests of the numerical MF calculation and extrapolation.

if "__main__" == __name__:
    # Load the data from the numerical MFs from Andrew Gow
    
    filepath = "./Data/psiData/"
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    
    gamma = 0.36
    PL_exp = 1/gamma
    
    colormap = mpl.colormaps['rainbow'].resampled(7)
    colors = colormap(range(7))
    
    for i in range(len(Deltas)):
                
        if i == 0:
            log_m_pbh_values, mf_values = np.genfromtxt(filepath + "psiData_Delta_k35.txt", unpack=True, skip_header=1)
            m_pbh_values = np.exp(log_m_pbh_values)
        elif i >= 5:
            log_m_pbh_values_data, mf_values_data = np.genfromtxt(filepath + "psiData_Lognormal_D-{:.1f}.txt".format(Deltas[i]), unpack=True, skip_header=1)
            # For the Delta = 2 and Delta = 5 cases, load the data from Fig. 5, since this covers a wider PBH mass range than that provided by Andrew Gow
            m_pbh_values, mf_values_Fig5 = load_data("2009.03204/Delta_{:.1f}_numeric.csv".format(Deltas[i]))
            # Rescale the MF data from Fig. 5 (which is scaled by the maximum of the MF) so that its maximum matches the maximum from the data provided by Andrew Gow
            mf_values = mf_values_Fig5 * max(mf_values_data) / max(mf_values_Fig5)
        else:
            log_m_pbh_values, mf_values = np.genfromtxt(filepath + "psiData_Lognormal_D-{:.1f}.txt".format(Deltas[i]), unpack=True, skip_header=1)
            m_pbh_values = np.exp(log_m_pbh_values)        
        
        ax.plot(m_pbh_values, mf_values, color=colors[i], label="${:.1f}$".format(Deltas[i]))
        ax.plot(m_pbh_values, mf_values[0] * np.power(m_pbh_values/m_pbh_values[0], PL_exp), color=colors[i], linestyle="dotted")
 
        if i in (0, 1, 2, 3, 4, 5, 6):
            fig1, ax1 = plt.subplots(figsize=(6.5, 5.5))
            #ax1.plot(m_pbh_values, mf_values[0] * np.power(m_pbh_values/m_pbh_values[0], PL_exp), color="k", linestyle="dotted")
            
            # Find m_c for the lognormal fit by finding the PBH mass where the numerical MF is maximal
            mp_LN = m_pbh_values[np.argmax(mf_values)]
            mc_LN = mp_LN * np.exp(sigmas_LN[i]**2)
            
            # Range of PBH mass values to show for the fitting functions
            m_pbh_values_fits = np.logspace(np.log10(min(m_pbh_values))-2, np.log10(max(m_pbh_values))+3, 1000)

            ax1.plot(m_pbh_values_fits, LN(m_pbh_values_fits, m_c=mc_LN, sigma=sigmas_LN[i]), color="tab:red", dashes=[6, 2], label="LN")            
            ax1.plot(m_pbh_values_fits, SLN(m_pbh_values_fits, m_c=np.exp(ln_mc_SLN[i]), sigma=sigmas_SLN[i], alpha=alphas_SLN[i]), color="tab:blue", linestyle=(0, (5, 7)), label="SLN")
            ax1.plot(m_pbh_values_fits, CC3(m_pbh_values_fits, m_p=mp_CC3[i], alpha=alphas_CC3[i], beta=betas[i]), color="g", linestyle="dashed", label="CC3")
            # Plot the numerical MF obtained using mf_numeric(). Test the method when the booleans extrapolate_lower = extrapolate_upper_const = True
            ax1.plot(m_pbh_values_fits, mf_numeric(m_pbh_values_fits, mp_CC3[i], Deltas[i], normalised=True, extrap_lower=False, extrap_upper_const=True, n=1), color="tab:orange", linestyle="dotted")
            ax1.plot(m_pbh_values_fits, mf_numeric(m_pbh_values_fits, mp_CC3[i], Deltas[i], normalised=True, extrap_lower=False, extrap_upper_const=True, n=2), color="tab:orange", linestyle="dashdot")
            ax1.plot(m_pbh_values, mf_values, color="k", label="Numeric MF (data)")
            ax1.plot(m_pbh_values_fits, mf_numeric(m_pbh_values_fits, mp_CC3[i], Deltas[i], normalised=True), color="tab:orange", linestyle="solid", label="Numeric MF (calculated)")
                     
            ax1.set_xlabel(r"$m~[M_\odot]$")
            ax1.set_ylabel("$\psi(m)~[M_\odot^{-1}]$")
            ax1.set_title("$\Delta={:.1f}$".format(Deltas[i]))
            ax1.set_xlim(min(m_pbh_values)/5, 110*max(m_pbh_values))
            ax1.set_ylim(max(1e-8, min(mf_values)/100), 0.05)
            ax1.set_xscale("log")
            ax1.set_yscale("log")
            ax1.legend(fontsize="11")
            ax1.tick_params(pad=7)
            fig1.tight_layout()
                
    ax.set_xlabel(r"$m~[M_\odot]$")
    ax.set_ylabel("$\psi(m)~[M_\odot^{-1}]$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 200)
    ax.set_ylim(1e-6, 1e-1)
    ax.tick_params(pad=7)
    ax.legend(fontsize="xx-small", title="$\Delta$")
    fig.tight_layout()
    
    
#%% Compare numeric MF, GCC MF, and GCC MF with alpha=1/gamma.

if "__main__" == __name__:
    # Load the data from the numerical MFs from Andrew Gow
    
    filepath = "./Data/psiData/"
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    gamma = 0.36
    PL_exp = 1/gamma
    
    colormap = mpl.colormaps['rainbow'].resampled(7)
    colors = colormap(range(7))
    
    subplots_index = 0
    
    for i in range(len(Deltas)):
        
        if i in (0, 3, 4, 5):
            ax1 = axes.flatten()[subplots_index]
            
            if i == 0:
                log_m_pbh_values, mf_values = np.genfromtxt(filepath + "psiData_Delta_k35.txt", unpack=True, skip_header=1)
                m_pbh_values = np.exp(log_m_pbh_values)
            elif i >= 5:
                log_m_pbh_values_data, mf_values_data = np.genfromtxt(filepath + "psiData_Lognormal_D-{:.1f}.txt".format(Deltas[i]), unpack=True, skip_header=1)
                # For the Delta = 2 and Delta = 5 cases, load the data from Fig. 5, since this covers a wider PBH mass range than that provided by Andrew Gow
                m_pbh_values, mf_values_Fig5 = load_data("2009.03204/Delta_{:.1f}_numeric.csv".format(Deltas[i]))
                # Rescale the MF data from Fig. 5 (which is scaled by the maximum of the MF) so that its maximum matches the maximum from the data provided by Andrew Gow
                mf_values = mf_values_Fig5 * max(mf_values_data) / max(mf_values_Fig5)
            else:
                log_m_pbh_values, mf_values = np.genfromtxt(filepath + "psiData_Lognormal_D-{:.1f}.txt".format(Deltas[i]), unpack=True, skip_header=1)
                m_pbh_values = np.exp(log_m_pbh_values)        
            
            # Find m_c for the lognormal fit by finding the PBH mass where the numerical MF is maximal
            mp_LN = m_pbh_values[np.argmax(mf_values)]
            mc_LN = mp_LN * np.exp(sigmas_LN[i]**2)
            
            # Range of PBH mass values to show for the fitting functions
            m_pbh_values_fits = np.logspace(np.log10(min(m_pbh_values))-2, np.log10(max(m_pbh_values))+3, 1000)
            
            # Plot the numerical MF obtained using mf_numeric(). Test the method when the booleans extrapolate_lower = extrapolate_upper_const = True
            ax1.plot(m_pbh_values, mf_values, color="k", label="Numeric MF")
            ax1.plot(m_pbh_values_fits, CC3(m_pbh_values_fits, m_p=mp_CC3[i], alpha=alphas_CC3[i], beta=betas[i]), color="limegreen", linestyle="dashed", label=r"GCC (best-fit $\alpha, \,\beta$)")
            ax1.plot(m_pbh_values_fits, CC3(m_pbh_values_fits, m_p=mp_CC3[i], alpha=1/0.36, beta=betas[i]), color="limegreen", linestyle="dotted", label=r"GCC ($\alpha=1/\gamma$, best-fit $\beta$)")
                     
            ax1.set_xlabel(r"$m~[M_\odot]$")
            ax1.set_ylabel("$\psi(m)~[M_\odot^{-1}]$")
            ax1.set_title("$\Delta={:.1f}$".format(Deltas[i]))
            ax1.set_xlim(min(m_pbh_values), max(m_pbh_values))
            ax1.set_ylim(max(1e-8, min(mf_values)/10), 2*max(mf_values))
            ax1.set_xscale("log")
            ax1.set_yscale("log")
            if i == 0:
                ax1.legend(fontsize="xx-small")
            ax1.tick_params(pad=7)
            subplots_index += 1            

    fig.tight_layout(pad=0.3)                


#%% Microlensing constraints with numeric MF: compare evolved to unevolved MF

if "__main__" == __name__:
    
    m_delta_Subaru, f_max_Subaru = load_data("2007.12697/Subaru-HSC_2007.12697_dx=5.csv")
    
    # Peak mass, in grams
    m_p = 1e23
    
    Delta = 5
    
    extrap_lower = False
    extrap_upper_const = True
    normalised = True
    n = 1
    
    # Parameters for the numeric MF
    params_numeric = [Delta, extrap_lower, extrap_upper_const, normalised, n]
    
    f_pbh_Subaru_evolved = constraint_Carr([m_p], m_delta_Subaru, f_max_Subaru, mf_numeric, params_numeric, evolved=True)
    f_pbh_Subaru_unevolved = constraint_Carr([m_p], m_delta_Subaru, f_max_Subaru, mf_numeric, params_numeric, evolved=False)
    
    print("Using constraint_Carr()")
    print("f_PBH (evolved numeric MF) = {:.2e}".format(f_pbh_Subaru_evolved[0]))
    print("f_PBH (unevolved numeric MF) = {:.2e}".format(f_pbh_Subaru_unevolved[0]))
    
    print("\nRecalculated")    
    # Calculate numeric MF constraint without calling constraint_Carr()
    psi_initial_values = mf_numeric(m_delta_Subaru, m_p, Delta, normalised=normalised, extrap_upper_const=extrap_upper_const, n=n)
    f_pbh_Subaru_unevolved = 1 / np.trapz(psi_initial_values / f_max_Subaru, m_delta_Subaru)
    print("f_PBH (unevolved numeric MF) = {:.2e}".format(f_pbh_Subaru_unevolved))
    
    m_evolved = mass_evolved(m_delta_Subaru, t=t_0)
    psi_evolved_values = psi_evolved(psi_initial_values, m_evolved, m_delta_Subaru)
    f_pbh_Subaru_evolved = 1 / np.trapz(psi_evolved_values / f_max_Subaru, m_evolved)
    print("f_PBH (evolved numeric MF) = {:.2e}".format(f_pbh_Subaru_evolved))
    
    psi_evolved_values_normalised = psi_evolved_normalised(psi_initial_values, m_evolved, m_delta_Subaru)
    f_pbh_Subaru_evolved = 1 / np.trapz(psi_evolved_values_normalised / f_max_Subaru, m_evolved)
    # This value is different from the others since the integral of the numeric
    # MF over mass is only calculated for the range of masses where the 
    # Subaru-HSC delta-function MF constraint is known
    print("f_PBH (evolved numeric MF using psi_evolved_normalised()) = {:.2e}".format(f_pbh_Subaru_evolved))

    # Initial masses matching those used in constraint_Carr() when calculating constraints for evolved MFs.
    n_steps = 1000
    m_init_values = np.sort(np.concatenate((np.logspace(np.log10(min(m_delta_Subaru)), np.log10(m_star), n_steps), np.arange(m_star, m_star*(1+1e-11), 5e2), np.arange(m_star*(1+1e-11), m_star*(1+1e-6), 1e7), np.logspace(np.log10(m_star*(1+1e-4)), np.log10(max(m_delta_Subaru))+4, n_steps))))

    psi_initial_values = mf_numeric(m_init_values, m_p, Delta, normalised=normalised, extrap_upper_const=extrap_upper_const, n=n)
    print(psi_initial_values[0:50])
    m_evolved = mass_evolved(m_init_values, t=t_0)
    psi_evolved_values = psi_evolved_normalised(psi_initial_values, m_evolved, m_init_values)
    #psi_evolved_values_interpolated = 10**np.interp(np.log10(m_delta_Subaru), np.log10(m_evolved), np.log10(psi_evolved_values), left=-100, right=-100)
    psi_evolved_values_interpolated = 10**np.interp(np.log10(m_delta_Subaru), np.log10(m_evolved), np.log10(psi_evolved_values), left=-np.infty, right=-np.infty)   
    print(len(psi_evolved_values_interpolated[psi_evolved_values_interpolated>0]))
    f_pbh_Subaru_evolved = 1 / np.trapz(psi_evolved_values_interpolated / f_max_Subaru, m_delta_Subaru)

    print("f_PBH (evolved numeric MF following constraint_Carr) = {:.2e}".format(f_pbh_Subaru_evolved))


#%% Voyager-1 constraints with numeric MF: compare evolved to unevolved MF
if "__main__" == __name__:
    
    m_delta_loaded, f_max_loaded = load_data("1807.03075/1807.03075_prop_B_bkg_lower.csv")
    
    m_delta_extrapolated = 10**np.arange(11, np.log10(min(m_delta_loaded))+0.01, 0.1)
    f_max_extrapolated = min(f_max_loaded) * np.power(m_delta_extrapolated / min(m_delta_loaded), 2)

    f_max = np.concatenate((f_max_extrapolated, f_max_loaded))
    m_delta = np.concatenate((m_delta_extrapolated, m_delta_loaded))

    
    # Peak mass, in grams
    m_p = 1e18
    
    Delta = 5
    
    extrap_lower = False
    extrap_upper_const = True
    normalised = True
    n = 2
    
    # Parameters for the numeric MF
    params_numeric = [Delta, extrap_lower, extrap_upper_const, normalised, n]
    
    f_pbh_evolved = constraint_Carr([m_p], m_delta, f_max, mf_numeric, params_numeric, evolved=True)
    f_pbh_unevolved = constraint_Carr([m_p], m_delta, f_max, mf_numeric, params_numeric, evolved=False)
    
    print("Using constraint_Carr()")
    print("f_PBH (evolved numeric MF) = {:.2e}".format(f_pbh_evolved[0]))
    print("f_PBH (unevolved numeric MF) = {:.2e}".format(f_pbh_unevolved[0]))
    
    print("\nRecalculated")    
    # Calculate numeric MF constraint without calling constraint_Carr()
    psi_initial_values = mf_numeric(m_delta, m_p, Delta, normalised=normalised, extrap_upper_const=extrap_upper_const, n=n)
    f_pbh_unevolved = 1 / np.trapz(psi_initial_values / f_max, m_delta)
    print("f_PBH (unevolved numeric MF) = {:.2e}".format(f_pbh_unevolved))
    
    m_evolved = mass_evolved(m_delta, t=t_0)
    psi_evolved_values = psi_evolved(psi_initial_values, m_evolved, m_delta)
    f_pbh_evolved = 1 / np.trapz(psi_evolved_values / f_max, m_evolved)
    print("f_PBH (evolved numeric MF) = {:.2e}".format(f_pbh_evolved))
    
    psi_evolved_values_normalised = psi_evolved_normalised(psi_initial_values, m_evolved, m_delta)
    f_pbh_evolved = 1 / np.trapz(psi_evolved_values_normalised / f_max, m_evolved)
    # This value is different from the others since the integral of the numeric
    # MF over mass is only calculated for the range of masses where the 
    # delta-function MF constraint is known
    print("f_PBH (evolved numeric MF using psi_evolved_normalised()) = {:.2e}".format(f_pbh_evolved))

    # Initial masses matching those used in constraint_Carr() when calculating constraints for evolved MFs.
    n_steps = 1000
    m_init_values = np.sort(np.concatenate((np.logspace(np.log10(min(m_delta)), np.log10(m_star), n_steps), np.arange(m_star, m_star*(1+1e-11), 5e2), np.arange(m_star*(1+1e-11), m_star*(1+1e-6), 1e7), np.logspace(np.log10(m_star*(1+1e-4)), np.log10(max(m_delta))+4, n_steps))))

    psi_initial_values = mf_numeric(m_init_values, m_p, Delta, normalised=normalised, extrap_upper_const=extrap_upper_const, n=n)
    print(psi_initial_values[0:50])
    m_evolved = mass_evolved(m_init_values, t=t_0)
    psi_evolved_values = psi_evolved_normalised(psi_initial_values, m_evolved, m_init_values)
    #psi_evolved_values_interpolated = 10**np.interp(np.log10(m_delta), np.log10(m_evolved), np.log10(psi_evolved_values), left=-100, right=-100)
    psi_evolved_values_interpolated = 10**np.interp(np.log10(m_delta), np.log10(m_evolved), np.log10(psi_evolved_values), left=-np.infty, right=-np.infty)   
    print(len(psi_evolved_values_interpolated[psi_evolved_values_interpolated>0]))
    f_pbh_evolved = 1 / np.trapz(psi_evolved_values_interpolated / f_max, m_delta)

    print("f_PBH (evolved numeric MF following constraint_Carr) = {:.2e}".format(f_pbh_evolved))


#%% Convergence test for the normalisation factor of the numeric MF
if "__main__" == __name__:
    
    filepath = "./Data/psiData/"
    
    for i in range(len(Deltas)):
        
        Delta = Deltas[i]
        
        # Load the data from the numerical MFs from Andrew Gow    
        if Delta < 0.1:
            log_m_data, mf_data_loaded = np.genfromtxt("./Data/psiData/psiData_Delta_k35.txt", unpack=True, skip_header=1)
            # Only include the range of masses for which the numerical MF data has non-zero positive values.
            mf_data = mf_data_loaded[mf_data_loaded >= 0]
            m_data = np.exp(log_m_data[mf_data_loaded >= 0])
        elif Delta < 2:
            log_m_data, mf_data_loaded = np.genfromtxt("./Data/psiData/psiData_Lognormal_D-{:.1f}.txt".format(Delta), unpack=True, skip_header=1)
            # Only include the range of masses for which the numerical MF data has non-zero positive values.
            mf_data = mf_data_loaded[mf_data_loaded >= 0]
            m_data = np.exp(log_m_data[mf_data_loaded >= 0])
        else:
            log_m_data_tabulated, mf_data_tabulated = np.genfromtxt("./Data/psiData/psiData_Lognormal_D-{:.1f}.txt".format(Delta), unpack=True, skip_header=1)
            # For the Delta = 5 case, load the data from Fig. 5, since this covers a wider PBH mass range than that provided by Andrew Gow
            m_data, mf_data_Fig5 = load_data("2009.03204/Delta_{:.1f}_numeric.csv".format(Delta))
            # Rescale the MF data from Fig. 5 (which is scaled by the maximum of the MF) so that its maximum matches the maximum from the data provided by Andrew Gow
            mf_data = mf_data_Fig5 * max(mf_data_tabulated) / max(mf_data_Fig5)
        
        markers = ["x", "+", "1"]
        n_values = np.arange(0, 5, 0.25)
        colors = ["tab:blue", "tab:orange", "k"]
        
        if i in range(len(Deltas)):
            
            fig, ax = plt.subplots(figsize=(6, 6))
    
            for j, n_steps in enumerate((100, 1000, 10000)):
            #for j, n_steps in enumerate([10]):
                      
                normalisation_factors = []
                log_m_range = []
                
                # n = number of powers of ten in the mass to extrapolate the mass function to outside the range in which data is available
                for n in n_values:

                    m_data_lower = min(m_data) * np.logspace(-n, 0, n_steps)
                    m_data_upper = max(m_data) * np.logspace(0, n, n_steps)
                    m_data_total = np.concatenate((m_data_lower, m_data, m_data_upper))
                    n_steps_true = len(m_data_total)
                    
                    mf_values = mf_numeric(m_data_total, m_p=m_data[mf_data / max(mf_data) >= 1], custom_mp=True, Delta=Deltas[i], normalised=False, extrap_upper_const=True, n=n)

                    normalisation_factors.append(np.trapz(mf_values[mf_values>0], m_data_total[mf_values>0]))
                    log_m_range.append(np.log10(max(m_data_total)/min(m_data_total)))

                ax.plot(0, 0, marker=markers[j], label=len(m_data_total), color=colors[j], linestyle="None")
                ax.plot(log_m_range, normalisation_factors / normalisation_factors[-1], marker=markers[j], color=colors[j], linestyle="None")
                ax.legend(title="Number of steps", fontsize="x-small")
                
            ax.set_xlabel(r"$\log_{10}(m_{\rm max} / m_{\rm min})$")
            #ax.set_ylabel(r"Normalisation factor (normalised)")
            ax.set_ylabel(r"$\int {\rm d}m\psi(m)$" + " (normalised to most accurate)")
            ax.set_yscale("log")
            ax.set_title("$\Delta={:.1f}$".format(Deltas[i]))
            fig.tight_layout()
    
    
#%% Sanity check: calculate one value of the extended MF constraint (for Figs. 2-3)

if "__main__" == __name__:
    
    m_pbh_GECCO, f_max_GECCO = load_data("2101.01370/2101.01370_Fig9_GC_NFW.csv")
    m_pbh_Voyager, f_max_Voyager = load_data("1807.03075/1807.03075_prop_A_bkg.csv")
    m_pbh_KP23, f_max_KP23 = load_data("2302.04408/2302.04408_MW_diffuse_SPI.csv")
    
    Delta_index = 6
    if Delta_index < 6:
        m_p = 1e17
    else:
        m_p = 1e18
        
    calc_GCC = False
    calc_LN = False
    calc_SLN = True
        
    if calc_GCC:
        mf = CC3
        m_c = m_p
        params = [alphas_CC3[Delta_index], betas[Delta_index]]
    elif calc_LN:
        mf = LN
        m_c = m_p * np.exp(sigmas_LN[Delta_index]**2)
        params = [sigmas_LN[Delta_index]]
    elif calc_SLN:
        mf = SLN
        params = [sigmas_SLN[Delta_index], alphas_SLN[Delta_index]]
        if Delta_index == 0:
            m_c = 1.53 * m_p   # for Delta = 0
        elif Delta_index == 5:
            m_c = 3.24 * m_p   # for Delta = 2
        elif Delta_index == 6:
            m_c = 6.8 * m_p   # for Delta = 5
        print("Peak mass (SLN) = {:.2e} g".format(m_max_SLN(m_c, *params)))

    print("m_p = {:.2e} g".format(m_p))
    
    f_PBH_GECCO = 1 / np.trapz(mf(m_pbh_GECCO, m_c, *params) / f_max_GECCO, m_pbh_GECCO)
    f_PBH_Voyager = 1 / np.trapz(mf(m_pbh_Voyager, m_c, *params) / f_max_Voyager, m_pbh_Voyager)
    
    m_pbh_KP23_extrapolated = np.logspace(11, 16, 51)
    f_max_KP23_extrapolated = min(f_max_KP23) * np.power(m_pbh_KP23_extrapolated/min(m_pbh_KP23), 2)
    
    m_pbh_KP23_total = np.concatenate((m_pbh_KP23_extrapolated, m_pbh_KP23))
    f_max_KP23_total = np.concatenate((f_max_KP23_extrapolated, f_max_KP23))
    
    f_PBH_KP23 = 1 / np.trapz(mf(m_pbh_KP23, m_c, *params) / f_max_KP23, m_pbh_KP23)
    f_PBH_KP23_extrapolated = 1 / np.trapz(mf(m_pbh_KP23_total, m_c, *params) / f_max_KP23_total, m_pbh_KP23_total)

    print("f_PBH (GECCO) = {:.3e}".format(f_PBH_GECCO))
    print("f_PBH (Voyager 1) = {:.3e}".format(f_PBH_Voyager))
    print("f_PBH (INTEGRAL) = {:.3e}".format(f_PBH_KP23))
    print("f_PBH (INTEGRAL) [f_max extrapolated] = {:.3e}".format(f_PBH_KP23_extrapolated))
  
    m_pbh_HSC, f_max_HSC = load_data("2007.12697/Subaru-HSC_2007.12697_dx=5.csv")
    m_pbh_Sugiyama, f_max_Sugiyama = load_data("1905.06066/1905.06066_Fig8_finite+wave.csv")

    m_p = 1e23
    
    if calc_GCC:
        mf = CC3
        m_c = m_p
        params = [alphas_CC3[Delta_index], betas[Delta_index]]
    elif calc_LN:
        mf = LN
        m_c = m_p * np.exp(sigmas_LN[Delta_index]**2)
        params = [sigmas_LN[Delta_index]]
    elif calc_SLN:
        mf = SLN
        params = [sigmas_SLN[Delta_index], alphas_SLN[Delta_index]]
        if Delta_index == 0:
            m_c = 1.53 * m_p   # for Delta = 0
        elif Delta_index == 5:
            m_c = 3.24 * m_p   # for Delta = 2
        elif Delta_index == 6:
            m_c = 6.8 * m_p   # for Delta = 5
        print("Peak mass (SLN) = {:.2e} g".format(m_max_SLN(m_c, *params)))

    print("m_p = {:.2e} g".format(m_p))

    print("\nm_p = {:.2e} g".format(m_p))
    
    f_PBH_HSC = 1 / np.trapz(mf(m_pbh_HSC, m_c, *params) / f_max_HSC, m_pbh_HSC)
    f_PBH_Sugiyama = 1 / np.trapz(mf(m_pbh_Sugiyama, m_c, *params) / f_max_Sugiyama, m_pbh_Sugiyama)
    print("f_PBH (HSC) = {:.3e}".format(f_PBH_HSC))
    print("f_PBH (Sugiyama proposal) = {:.3e}".format(f_PBH_Sugiyama))


#%% Sanity check: calculate one value of the extended MF constraint (for Fig. 4)

if "__main__" == __name__:
    
    m_pbh_Voyager, f_max_Voyager = load_data("1807.03075/1807.03075_prop_A_bkg.csv")
    
    mf = LN
    
    for sigma in [sigmas_LN[-1], 2]:
        m_p = 1e18
        print("sigma = {:.2f}".format(sigma))

        m_c = m_p * np.exp(sigma**2)
        params = [sigma]
        
        m_delta_extrapolated = 10**np.arange(11, np.log10(min(m_pbh_Voyager))+0.01, 0.1)
        f_max_extrapolated = min(f_max_Voyager) * np.power(m_delta_extrapolated / min(m_pbh_Voyager), 2)
    
        f_max_total = np.concatenate((f_max_extrapolated, f_max_Voyager))
        m_delta_total = np.concatenate((m_delta_extrapolated, m_pbh_Voyager))

    
        print("m_p = {:.2e} g".format(m_p))
        f_PBH_Voyager = 1 / np.trapz(mf(m_pbh_Voyager, m_c, *params) / f_max_Voyager, m_pbh_Voyager)
        f_PBH_Voyager_extrapolated = 1 / np.trapz(mf(m_delta_total, m_c, *params) / f_max_total, m_delta_total)
        print("f_PBH (Voyager 1) = {:.3e}".format(f_PBH_Voyager))
        print("f_PBH (Voyager 1) [f_max extrapolated] = {:.3e}".format(f_PBH_Voyager_extrapolated))
    
        m_c = 2e19 
        print("m_c = {:.2e} g".format(m_c))
        f_PBH_Voyager = 1 / np.trapz(mf(m_pbh_Voyager, m_c, *params) / f_max_Voyager, m_pbh_Voyager)
        f_PBH_Voyager_extrapolated = 1 / np.trapz(mf(m_delta_total, m_c, *params) / f_max_total, m_delta_total)
        print("f_PBH (Voyager 1) = {:.3e}".format(f_PBH_Voyager))
        print("f_PBH (Voyager 1) [f_max extrapolated] = {:.3e}".format(f_PBH_Voyager_extrapolated))


#%% Calculate the skew of each mass function, for Delta=(0, 2,5)
# Should find that the Delta=5 SLN is the only MF that has positive skew in log-space

if "__main__" == __name__:
    
    m_p = 1e20
    print("m_p = {:.2e} g".format(m_p))
    m_values = np.logspace(np.log10(m_p)-6, np.log10(m_p)+6, 100000)
   
    calc_GCC = True
    calc_LN = False
    calc_SLN = False
    
    for Delta_index in [0, 5, 6]:
        print("\nDelta = {:.0f}".format(Deltas[Delta_index]))
                
        if calc_GCC:
            mf = CC3
            m_c = m_p
            params = [alphas_CC3[Delta_index], betas[Delta_index]]
            
        elif calc_LN:
            mf = LN
            m_c = m_p * np.exp(sigmas_LN[Delta_index]**2)
            params = [sigmas_LN[Delta_index]]
            print("sigma (parameter) = {:.2e}".format(sigmas_LN[Delta_index]))

        elif calc_SLN:
            mf = SLN
            params = [sigmas_SLN[Delta_index], alphas_SLN[Delta_index]]
            if Delta_index == 0:
                m_c = 1.53 * m_p   # for Delta = 0
            elif Delta_index == 5:
                m_c = 3.24 * m_p   # for Delta = 2
            elif Delta_index == 6:
                m_c = 6.8 * m_p   # for Delta = 5
            print("Peak mass (SLN) = {:.2e} g".format(m_max_SLN(m_c, *params)))

        mean = np.trapz(mf(m_values, m_c, *params) * np.log(m_values), m_values)
        print("mean = {:.2e} g".format(np.exp(mean)))
        variance = np.trapz(mf(m_values, m_c, *params) * np.log(m_values)**2, m_values) - mean**2
        print("sigma (calculated) = {:.2e}".format(np.sqrt(variance)))
        
        # Expression for skewness from https://en.wikipedia.org/wiki/Skewness, adapted to log-space scenario
        # Sanity check: values for log-normal should be negligible
        skew = np.trapz(mf(m_values, m_c, *params) * np.power((np.log(m_values) - mean) / np.sqrt(variance), 3), m_values)
        print("skew = {:.2e}".format(skew))