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
# Results for lognormal from Andrew Gow.
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


def PL_MF(m_values, m_min, m_max, gamma=-1/2):
    """
    Power-law mass function, defined in e.g. Eq. (3.1) of Bellomo et al. (2018)
    [1709.07467].

    Parameters
    ----------
    m_values : Array-like
        PBH mass values.
    m_min : Float
        Minimum mass at which the power-law mass function is defined.
    m_max : Float
        Maximum mass at which the power-law mass function is defined.
    gamma : Float, optional
        Power law exponent. The default is -1/2 (the value for PBHs formed 
        during the radiation-dominated epoch).

    Returns
    -------
    Array-like
        Value of the power-law mass function.

    """
    if gamma == 0:
        normalisation = 1 / np.log(m_max/m_min)
    else:
        normalisation = gamma / (np.power(m_max, gamma) - np.power(m_min, gamma))

    PL_MF_values = []
    for m in m_values:
        if m < m_min or m > m_max:
            PL_MF_values.append(0)
        else:
            PL_MF_values.append(normalisation / np.power(m, 1-gamma))
    
    return np.array(PL_MF_values)


def mf_numeric(m, m_p, Delta, extrap_lower=False, extrap_upper_const=False, normalised=False, n=1, gamma=0.36, custom_mp=True, log_interp=True):
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
        The default is False.
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
        Number of multiples of sigma (in log-space) of masses around m_c to 
        consider when estimating the maximum. The default is 5.
    n_steps : Integer, optional
        Number of masses to use for estimating the peak mass of the skew-
        lognormal mass function. The default is 100000.
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


def envelope(constraints, save_argmin=False, fname=None):
    """
    Calculate the tightest constraint at a given mass, from a set of 
    constraints.

    Parameters
    ----------
    constraints : Array-like
        Constraints on PBH abundance. All should have the same length and be
        evaluated at the same PBH mass.
    save_argmin : Boolean, optional
        If True, save the index of the constraint that places the tightest
        constraint. The default is False.
    fname : String, optional
        Name of file containing index of constraint that places tightest 
        constraint. Not needed if save == False. The default is None.

    Returns
    -------
    tightest : Array-like
        Tightest constraint, from the constraints given in the input.

    """
    tightest = np.ones(len(constraints[0]))
    tightest_arg = np.ones(len(constraints[0]))
    
    for i in range(len(constraints[0])):

        constraints_values = []

        for j in range(len(constraints)):
            if constraints[j][i] <= 0:
                constraints_values.append(1e100)
            else:
                constraints_values.append(abs(constraints[j][i]))
        tightest_arg[i] = np.argmax(constraints_values)
        tightest[i] = min(constraints_values)

    if save_argmin:
        np.savetxt(fname, tightest_arg)

    return tightest


def load_results_Isatis(mf_string="mono_E500", modified=True, test_mass_range=False, wide=False):
    """
    Read in constraints on f_PBH, obtained using Isatis, with a monochromatic PBH mass function.
    Parameters
    ----------
    mf_string : String, optional
        The mass function to load constraints for. Acceptable inputs are "mono" 
        (monochromatic), "LN" (log-normal), "SLN" (skew-lognormal) and "CC3" 
        (critical collapse 3), plus the value of the power spectrum width Delta. 
    modified : Boolean, optional
        If True, use data from the modified version of Isatis. The modified 
        version corrects a typo in the original version on line 1697 in 
        Isatis.c which means that the highest-energy bin in the observational 
        data set is not included. Otherwise, use the version of Isatis 
        containing the typo. The default is True.
    test_mass_range : Boolean, optional
        If True, use data obtained using the same PBH mass range for all Delta 
        (1000 BHs evenly spaced in log space between 1e11-1e21g, if wide=False).
    wide : Boolean, optional
        If True, use the 'wider' PBH mass range for all Delta (1000 BHs evenly 
        spaced in log space between 1e11-1e22g).
        
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
                        
                        
#%% Plot the mass function for Delta = 5.0, showing the mass range relevant for the Subaru-HSC microlensing constraints.
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
            
            
#%% Plot the mass function for Delta = 5.0, showing the mass range relevant for the Korwar & Profumo (2023) constraints.
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
            
            
#%% Plot the mass function for Delta = 0 and 5, showing the mass range relevant for the Galactic Centre photon constraints from Isatis (1e14g)
 
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


#%% Plot the mass function for Delta = 0, 2 and 5, showing the mass range relevant for the Galactic Centre photon constraints from Isatis (1e16g)

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
            

#%% Plot the mass function for Delta = 0, 2 and 5, showing the mass range relevant for the Galactic Centre photon constraints from Isatis (1e16g)
 
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
    
    
#%% Plot the mass function for Delta = 0, 2 and 5, showing the mass range relevant for the Galactic Centre photon constraints from Isatis (1e16g) - ratio plots.
 
if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    plot_LN = False
    plot_SLN = False
    plot_CC3 = True
    
    
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


#%% Plot the mass function for Delta = 0, 2 and 5, showing the mass range relevant for the prospective microlensing constraints (m ~ 1e22g)
 
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
 
def extract_GC_Isatis(j, k, exponent_PL_lower=2, include_extrapolated=True):
    """
    Load delta-function MF constraint on f_PBH from Galactic Centre photons.

    Parameters
    ----------
    j : Integer
        Index for which instrument to load data from (0 for COMPTEL, 1 for EGRET, 2 for Fermi-LAT, 3 for INTEGRAL).
    k : Integer
        Index for which energy bin of the instrument to load data from.
    exponent_PL_lower : Float
        Power-law exponent to use between 1e11g and 1e13g.
    include_extrapolated : Boolean
        If True, extrapolate delta-function MF constraints using a power-law below 1e13g. The default is True.

    Returns
    -------
    f_max : Array-like
        Values of the delta-function mass function constraint.
    m_pbh_values : Array-like
        PBH masses the delta-function mass function constraint is evaluated at, in grams.

    """
    m_delta_values_loaded = np.logspace(11, 22, 1000)            
    m_delta_extrapolated = np.logspace(11, 13, 21)
    
    constraints_names_short = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]

    f_max_all = np.genfromtxt("./Data/fPBH_GC_full_all_bins_%s_monochromatic_wide.txt" % constraints_names_short[j], unpack=True)
                                  
    # Set non-physical values of f_max (-1 or np.infty) to 1e100 from the f_max values calculated using Isatis
    f_max_allpositive = []

    for f_max in f_max_all[k]:
        if f_max == -1 or f_max == np.infty:
            f_max_allpositive.append(1e100)
        else:
            f_max_allpositive.append(f_max)
    
    # Extrapolate f_max at masses below 1e13g using a power-law
    if include_extrapolated:
        f_max_loaded_truncated = np.array(f_max_allpositive)[m_delta_values_loaded > 1e13]
        f_max_extrapolated = f_max_loaded_truncated[0] * np.power(m_delta_extrapolated / 1e13, exponent_PL_lower)
        f_max_k = np.concatenate((f_max_extrapolated, f_max_loaded_truncated))
        m_delta_values = np.concatenate((m_delta_extrapolated, m_delta_values_loaded[m_delta_values_loaded > 1e13]))
    else:
        f_max_k = f_max_allpositive
        m_delta_values = m_delta_values_loaded
                        
    return f_max_k, m_delta_values


if "__main__" == __name__:
    
    plot_KP23 = True
    plot_GC_Isatis = False
    plot_BC19 = False
    plot_Subaru = False
    plot_Sugiyama19 = False
    Delta = 0
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    # Peak mass, in grams
    m_p = 5e16
    
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
        
        m_delta_extrapolated_upper = np.logspace(15, np.log10(min(m_delta_values_loaded)), 11)
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
        # Power-law exponent to use between 1e11g and 1e13g.
        exponent_PL_lower = 0
        
        # Constraints from Galactic Centre photons from Isatis (see Auffinger (2022) [2201.01265])
        
        # Select which instrument places the tightest constraint, for the evolved MF constraints (depends on Delta and the peak mass). Values are for a peak mass m_p = 1e16g
        # 0 for COMPTEL, 1 for EGRET, 2 for Fermi-LAT, 3 for INTEGRAL
        # Note: the defaults here apply to the evolved MF constraints and a peak mass m_p = 1e16g, and do not depend on the fitting function at m_p = 1e16g 
        
        if Delta == 0:
            j = 0 # for m_p = 1e16g
        elif Delta == 2:
            if m_p == 1e16:
                j = 0 # for m_p = 1e16g
            elif m_p == 1e17:
                j = 3 # for m_p = 1e17g
        elif Delta == 5:
            if m_p == 1e16:
                j = 2 # for m_p = 1e16g
            elif 5e16 <= m_p <= 1e17: 
                j = 1 # for m_p = 5e16g and m_p = 1e17g
        
        constraints_names, f_max_Isatis = load_results_Isatis(modified=True)
        colours_GC_fit = ["tab:orange", "tab:green", "tab:red", "tab:blue"]
        
        mc_values = np.logspace(14, 20, 120)
        
        # Find index of mc_values for which the peak mass is closest to the desired m_p:
        mp_values_SLN = np.array([m_max_SLN(mc_SLN, sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=4, n_steps=1000) for mc_SLN in mc_values])
        mp_values_LN = mc_values * np.exp(-sigmas_LN[i]**2)
        mp_values_CC3 = mc_values
        argmin_SLN = np.argmax(mp_values_SLN > m_p)
        argmin_LN = np.argmax(mp_values_LN > m_p)
        argmin_CC3 = np.argmax(mp_values_CC3 > m_p)
        
        print("SLN \n m_p[arg - 1] = {:.3e}g, m_p[arg] = {:.3e} g.".format(mp_values_SLN[argmin_SLN-1], mp_values_SLN[argmin_SLN]))
        print("LN \n m_p[arg - 1] = {:.3e}g, m_p[arg] = {:.3e} g.".format(mp_values_LN[argmin_LN-1], mp_values_LN[argmin_LN]))
        print("CC3 \n m_p[arg - 1] = {:.3e}g, m_p[arg] = {:.3e} g.".format(mp_values_CC3[argmin_CC3-1], mp_values_CC3[argmin_CC3]))
               
        constraints_names_short = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
        # Find the energy bin that places the tightest constraint on f_PBH at the given peak mass
        data_folder_base = "./Data-tests/"
        data_folder = data_folder_base + "/PL_exp_{:.0f}/argmin/".format(exponent_PL_lower)
        fname_LN = data_folder + "LN_GC_%s" % constraints_names_short[j] + "_Delta={:.1f}.txt".format(Deltas[j])
        fname_SLN = data_folder + "SLN_GC_%s" % constraints_names_short[j] + "_Delta={:.1f}.txt".format(Deltas[j])
        fname_CC3 = data_folder + "CC3_GC_%s" % constraints_names_short[j] + "_Delta={:.1f}.txt".format(Deltas[j])
        
        k_LN = int(np.genfromtxt(fname_LN)[argmin_LN])
        k_SLN = int(np.genfromtxt(fname_LN)[argmin_SLN])
        k_CC3 = int(np.genfromtxt(fname_LN)[argmin_CC3])
        k_LN = k_SLN = k_CC3 = 2
        k_EGRET = 1
        
        if k_LN == k_SLN == k_CC3:
            
            f_max, m_pbh_values = extract_GC_Isatis(j, k_LN, exponent_PL_lower)

            f_max_EGRET, m_pbh_values_EGRET = extract_GC_Isatis(1, k_EGRET, exponent_PL_lower)
            f_max_FermiLAT, m_pbh_values_FermiLAT = extract_GC_Isatis(2, k_LN, exponent_PL_lower)
            
            # For Delta = 0 LN constraints
            f_max_COMPTEL_k1, m_pbh_values_COMPTEL_k1 = extract_GC_Isatis(0, 1, exponent_PL_lower)

        else:
            print("Error: different energy bins place constraint for different fitting functions.")
        
    elif plot_BC19:
        # Voyager 1 cosmic ray constraints, from Boudaud & Cirelli (2019) [1807.03075]
        
        # Boolean determines which propagation model to load data from
        prop_A = False
        prop_B = not prop_A
 
        # If True, use constraints obtained with background subtraction
        with_bkg_subtr = True
        
        # If True, load the more stringent "prop B" constraint
        prop_B_lower = True

        
        if with_bkg_subtr:
            if not prop_B_lower:
                m_pbh_values_loaded, f_max_loaded = load_data("1807.03075/1807.03075_prop_B_bkg_upper.csv")
            else:
                m_pbh_values_loaded, f_max_loaded = load_data("1807.03075/1807.03075_prop_B_bkg_lower.csv")                        
        else:
            if not prop_B_lower:
                m_pbh_values_loaded, f_max_loaded = load_data("1807.03075/1807.03075_prop_B_nobkg_upper.csv")
            else:
                m_pbh_values_loaded, f_max_loaded = load_data("1807.03075/1807.03075_prop_B_nobkg_lower.csv")                        
    
        exponent_PL_lower = 2
    
        m_delta_extrapolated = 10**np.arange(11, np.log10(min(m_pbh_values_loaded))+0.01, 0.1)
        f_max_extrapolated = min(f_max_loaded) * np.power(m_delta_extrapolated / min(m_pbh_values_loaded), exponent_PL_lower)
    
        f_max = np.concatenate((f_max_extrapolated, f_max_loaded))
        m_pbh_values = np.concatenate((m_delta_extrapolated, m_pbh_values_loaded))

    
    mp_SLN = m_max_SLN(mc_SLN, sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=4, n_steps=1000)
    mc_LN = m_p * np.exp(+sigmas_LN[i]**2)
    print("m_p (CC3) = {:.2e}".format(m_p))
    print("m_p (SLN) = {:.2e}".format(mp_SLN))
     
    m_pbh_values_init = np.sort(np.concatenate((np.arange(m_star, m_star*(1+1e-11), 5e2), np.arange(m_star*(1+1e-11), m_star*(1+1e-6), 1e7),  np.logspace(np.log10(m_p)-3, np.log10(m_p)+3, 100))))
    n_steps = 1000
    #m_init_values_input = np.sort(np.concatenate((np.logspace(np.log10(min(m_pbh_values)), np.log10(m_star), n_steps), np.arange(m_star, m_star*(1+1e-11), 5e2), np.arange(m_star*(1+1e-11), m_star*(1+1e-6), 1e7), np.logspace(np.log10(m_star*(1+1e-4)), np.log10(max(m_pbh_values))+4, n_steps))))
    
    mf_LN_init = LN(m_pbh_values_init, mc_LN, sigma=sigmas_LN[i])
    mf_SLN_init = SLN(m_pbh_values_init, mc_SLN, sigma=sigmas_SLN[i], alpha=alphas_SLN[i])
    mf_CC3_init = CC3(m_pbh_values_init, m_p, alpha=alphas_CC3[i], beta=betas[i])
    
    m_pbh_values_evolved = mass_evolved(m_pbh_values_init, t_0)
    mf_LN_evolved = 10**np.interp(np.log10(m_pbh_values), np.log10(m_pbh_values_evolved), np.log10(psi_evolved_normalised(mf_LN_init, m_pbh_values_evolved, m_pbh_values_init)))
    mf_SLN_evolved = 10**np.interp(np.log10(m_pbh_values), np.log10(m_pbh_values_evolved), np.log10(psi_evolved_normalised(mf_SLN_init, m_pbh_values_evolved, m_pbh_values_init)))
    mf_CC3_evolved = 10**np.interp(np.log10(m_pbh_values), np.log10(m_pbh_values_evolved), np.log10(psi_evolved_normalised(mf_CC3_init, m_pbh_values_evolved, m_pbh_values_init)))

    # Unevolved MF, evaluated at the masses the delta-function MF constraint is evaluated at
    if plot_GC_Isatis and Delta == 5 and m_p == 1e16:
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

    if plot_GC_Isatis and Delta == 0:
        ax.plot(m_pbh_values, mf_SLN_evolved / f_max, color="b", label="SLN", linestyle=(0, (5, 7)))
        ax.plot(m_pbh_values, mf_CC3_evolved / f_max, color="g", label="CC3", linestyle="dashed")
        ax.plot(m_pbh_values, mf_LN_evolved / f_max_COMPTEL_k1, color="r", label="LN", dashes=[6, 2])

    if plot_GC_Isatis and Delta == 5:
        ax.plot(m_pbh_values, mf_SLN_evolved / f_max_EGRET, color="b", label="SLN", linestyle=(0, (5, 7)))
        ax.plot(m_pbh_values, mf_CC3_evolved / f_max, color="g", label="CC3", linestyle="dashed")
        ax.plot(m_pbh_values, mf_LN_evolved / f_max, color="r", label="LN", dashes=[6, 2])

    else:
        print("Here too!")
        ax.plot(m_pbh_values, mf_SLN_evolved / f_max, color="b", label="SLN", linestyle=(0, (5, 7)))
        ax.plot(m_pbh_values, mf_CC3_evolved / f_max, color="g", label="CC3", linestyle="dashed")
        ax.plot(m_pbh_values, mf_LN_evolved / f_max, color="r", label="LN", dashes=[6, 2])
    
    # Plot the integrand obtained with the unevolved MF
    if plot_KP23:
        ax.plot(m_pbh_values_upper, SLN(m_pbh_values_upper, mc_SLN, sigma=sigmas_SLN[i], alpha=alphas_SLN[i]) / f_max_upper, color="b", linestyle="dotted")
        ax.plot(m_pbh_values_upper, CC3(m_pbh_values_upper, m_p, alpha=alphas_CC3[i], beta=betas[i]) / f_max_upper, color="g", linestyle="dotted")
        ax.plot(m_pbh_values_upper, LN(m_pbh_values_upper, mc_LN, sigma=sigmas_LN[i]) / f_max_upper, color="r", linestyle="dotted")

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
        if Delta == 2:
            psinorm_fit_2 = mf_CC3_evolved[44] * np.power(m_pbh_values/m_pbh_values[44], 2)
        elif Delta == 5:
            psinorm_fit_2 = mf_CC3_evolved[39] * np.power(m_pbh_values/m_pbh_values[39], 2)

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
    """
    for ax in axes.flatten():
        ax.grid('on', linestyle='--')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    """
    axes[0].plot(m_pbh_values, mf_CC3_evolved, color="g", label="CC3", linestyle="dashed")
    axes[0].plot(m_pbh_values, mf_SLN_evolved, color="b", label="SLN", linestyle=(0, (5, 7)))
    axes[0].plot(m_pbh_values, mf_LN_evolved, color="r", label="LN", dashes=[6, 2])
    """
    axes[0].plot(m_pbh_values, mf_SLN_unevolved, color="b", linestyle="dotted", alpha=0.5)
    axes[0].plot(m_pbh_values, mf_CC3_unevolved, color="g", linestyle="dotted", alpha=0.5)
    axes[0].plot(m_pbh_values, mf_LN_unevolved, color="r", linestyle="dotted", alpha=0.5)
    """
    #axes[0].plot(m_pbh_values, psinorm_fit, linestyle="dotted", color="k", label="$m^3$ fit")
        
    """
    if Delta >= 2 and plot_BC19:
        axes[0].plot(m_pbh_values, psinorm_fit_2, linestyle="dotted", color="magenta", label="$m^2$ fit")
    """
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
            #axes[1].text(2e14, 2e8,"$\propto m^{-2}$", fontsize="x-small")
            axes[1].set_ylim(1e4, 1e12)

        if Delta == 0:
            axes[2].set_ylim(1e-14, 1e-11)

        elif Delta == 2:
            axes[2].set_ylim(1e-13, 1e-11)
            
        elif Delta == 5:
            axes[2].set_ylim(1e-13, 1e-10)

    elif plot_GC_Isatis:
               
        axes[1].plot(m_pbh_values, 1 / f_max, color=colours_GC_fit[j], label=constraints_names_short[j])
        
        if Delta == 0:
            axes[1].plot(m_pbh_values_COMPTEL_k1, 1 / f_max_COMPTEL_k1, label="COMPTEL (middle energy bin)", linestyle="dashed", color="tab:orange")
            ax.plot(m_pbh_values, mf_SLN_evolved / f_max, color="b", label="SLN", linestyle=(0, (5, 7)))
            ax.plot(m_pbh_values, mf_CC3_evolved / f_max, color="g", label="CC3", linestyle="dashed")
            ax.plot(m_pbh_values, mf_LN_evolved / f_max_COMPTEL_k1, color="r", label="LN", dashes=[6, 2])
            axes[1].legend(fontsize="x-small")
        
        if Delta >= 5 and m_p < 1e17:
            axes[1].plot(m_pbh_values_EGRET, 1 / f_max_EGRET, color="tab:green", label="EGRET")
            ax.plot(m_pbh_values, mf_SLN_evolved / f_max_EGRET, color="b", label="SLN", linestyle=(0, (5, 7)))
            ax.plot(m_pbh_values, mf_CC3_evolved / f_max, color="g", label="CC3", linestyle="dashed")
            ax.plot(m_pbh_values, mf_LN_evolved / f_max, color="r", label="LN", dashes=[6, 2])
            axes[1].legend(fontsize="x-small")

        axes[2].set_ylim(1e-15, 1e-12)
        exponent_fit = 2.8
        fmax_fit = f_max[90] * np.power(m_pbh_values/m_pbh_values[90], 2.8)
        axes[1].plot(m_pbh_values, 1 / fmax_fit, color="k", linestyle="dotted", label="$m^{-2.8}$ fit")
            
        axes[1].legend(fontsize="xx-small")
        axes[1].set_ylim(1, 1e14)
        
        if Delta == 0:
            axes[2].set_ylim(1e-14, 2e-12)        
            
        elif Delta >= 2:
            axes[2].set_ylim(1e-14, 2e-12)        
            
        if Delta == 5:
            axes[2].set_ylim(1e-14, 1e-8)        
        
    elif plot_BC19:
        
        if with_bkg_subtr:
            linestyle = "dashed"
        else:
            linestyle = "solid"
        if prop_A:
            axes[1].plot(m_pbh_values, 1 / f_max, color="b", linestyle=linestyle)
        elif prop_B:
            axes[1].plot(m_pbh_values, 1 / f_max, color="r", linestyle=linestyle)
        
        if Delta == 0:                
            axes[0].set_ylim(1e-24, 5e-18)    
            axes[1].set_ylim(1, 1e11)
            axes[2].set_ylim(1e-17, 1e-12)
            
        elif Delta == 5:
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

    if plot_GC_Isatis and Delta == 0:
        axes[2].plot(m_pbh_values, mf_SLN_evolved / f_max, color="b", label="SLN", linestyle=(0, (5, 7)))
        axes[2].plot(m_pbh_values, mf_CC3_evolved / f_max, color="g", label="CC3", linestyle="dashed")
        axes[2].plot(m_pbh_values, mf_LN_evolved / f_max_COMPTEL_k1, color="r", label="LN", dashes=[6, 2])

    elif plot_GC_Isatis and Delta >= 5:
        axes[2].plot(m_pbh_values, mf_SLN_evolved / f_max_EGRET, color="b", label="SLN", linestyle=(0, (5, 7)))
        axes[2].plot(m_pbh_values, mf_CC3_evolved / f_max, color="g", label="CC3", linestyle="dashed")
        axes[2].plot(m_pbh_values, mf_LN_evolved / f_max, color="r", label="LN", dashes=[6, 2])
        
    else:
        print("Here!")
        axes[2].plot(m_pbh_values, mf_SLN_evolved / f_max, color="b", label="SLN", linestyle=(0, (5, 7)))
        axes[2].plot(m_pbh_values, mf_CC3_evolved / f_max, color="g", label="CC3", linestyle="dashed")
        axes[2].plot(m_pbh_values, mf_LN_evolved / f_max, color="r", label="LN", dashes=[6, 2])
    
    """
    if plot_GC_Isatis and Delta == 5:
        axes[2].plot(m_pbh_values_FermiLAT, mf_SLN_unevolved / f_max_FermiLAT, color="b", linestyle="dotted")
        axes[2].plot(m_pbh_values_FermiLAT, mf_CC3_unevolved / f_max_FermiLAT, color="g", linestyle="dotted")
        axes[2].plot(m_pbh_values_FermiLAT, mf_LN_unevolved / f_max_FermiLAT, color="r", linestyle="dotted")

    else:
        axes[2].plot(m_pbh_values, mf_SLN_unevolved / f_max, color="b", linestyle="dotted")
        axes[2].plot(m_pbh_values, mf_CC3_unevolved / f_max, color="g", linestyle="dotted")
        axes[2].plot(m_pbh_values, mf_LN_unevolved / f_max, color="r", linestyle="dotted")
    """
    axes[2].set_ylabel("$\psi_\mathrm{N} / f_\mathrm{max}~[\mathrm{g}^{-1}]$")
    axes[2].set_xlabel("$m~[\mathrm{g}]$")
    
    if plot_KP23:
        fig.suptitle("KP '23 constraints, $\Delta={:.0f}$".format(Delta))

    elif plot_GC_Isatis:
        fig.suptitle("GC photon constraints, $\Delta={:.0f}$".format(Delta))
        
    elif plot_BC19:
        """
        if m_p > 1e17:
            axes[0].set_ylim(1e-24, 3e-21)
            axes[1].set_ylim(1, 1e9)
            axes[2].set_ylim(1e-20, 1e-14)
            
        else:
            if Delta == 5:
                axes[0].set_ylim(1e-23, 1e-17)
                axes[1].set_ylim(1e3, 1e11)
                axes[2].set_ylim(1e-15, 1e-11)

            elif Delta == 2:
                axes[0].set_ylim(1e-23, 1e-17)
                axes[1].set_ylim(1e3, 1e11)
                axes[2].set_ylim(1e-15, 1e-11)
        """
            
        fig.suptitle("Voyager 1 electron/positron constraints \n $\Delta={:.0f}$".format(Delta), fontsize="small")

    axes[0].legend(fontsize="xx-small")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[2].set_yscale("linear")

    for ax in axes:
        
        ax.set_xlim(1e14, max(m_pbh_values[f_max < 1]))
        
        if m_p > 1e18:
            ax.set_xlim(5e14, 1e18)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid('on')
        ax.set_xlabel("$m~[\mathrm{g}]$")
        
        # set x-axis and y-axis ticks
        # see https://stackoverflow.com/questions/30887920/how-to-show-minor-tick-labels-on-log-scale-with-matplotlib
        
        x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
        ax.xaxis.set_major_locator(x_major)
        x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 5)
        ax.xaxis.set_minor_locator(x_minor)
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4)
   
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
    print("constraint_Carr result [CC3] = {:.2e}".format(constraint_Carr([m_p], m_pbh_values, f_max, CC3, [alphas_CC3[i], betas[i]], evolved=True)[0]))

    integral_lower = np.trapz(mf_SLN_evolved[m_pbh_values<m_min] / f_max[m_pbh_values<m_min], m_pbh_values[m_pbh_values<m_min])
    integral_upper = np.trapz(mf_SLN_evolved[m_pbh_values<m_max] / f_max[m_pbh_values<m_max], m_pbh_values[m_pbh_values<m_max])
    integral_total = np.trapz(mf_SLN_evolved / f_max, m_pbh_values)
    print("\n Evolved MFs")
    print("Integral (M < {:.1e}g) / total integral [SLN] = {:.3f}".format(m_min, integral_lower / integral_total))
    print("Integral ({:.1e}g < M < {:.1e}g) / total integral [SLN] = {:.3f}".format(m_min, m_max, (integral_upper - integral_lower) / integral_total))
    print("1 / total integral [SLN] = {:.2e}".format(1 / integral_total))
    print("constraint_Carr result [SLN] = {:.2e}".format(constraint_Carr([mc_SLN], m_pbh_values, f_max, SLN, [sigmas_SLN[i], alphas_SLN[i]], evolved=True)[0]))

    integral_lower = np.trapz(mf_LN_evolved[m_pbh_values<m_min] / f_max[m_pbh_values<m_min], m_pbh_values[m_pbh_values<m_min])
    integral_upper = np.trapz(mf_LN_evolved[m_pbh_values<m_max] / f_max[m_pbh_values<m_max], m_pbh_values[m_pbh_values<m_max])
    integral_total = np.trapz(mf_LN_evolved / f_max, m_pbh_values)
    print("\n Evolved MFs")
    print("Integral (M < {:.1e}g) / total integral [LN] = {:.3f}".format(m_min, integral_lower / integral_total))
    print("Integral ({:.1e}g < M < {:.1e}g) / total integral [LN] = {:.3f}".format(m_min, m_max, (integral_upper - integral_lower) / integral_total))
    print("1 / total integral [LN] = {:.2e}".format(1 / integral_total))
    print("constraint_Carr result [LN] = {:.2e}".format(constraint_Carr([m_p*np.exp(sigmas_LN[i]**2)], m_pbh_values, f_max, LN, [sigmas_LN[i]], evolved=True)[0]))
    
    # For initial MFs, restrict the mass range to that loaded from the data itself
    
    if plot_KP23:
        f_max = np.array(f_max_upper)
        m_pbh_values = np.array(m_pbh_values_upper)
    
    if plot_GC_Isatis and Delta == 5 and m_p == 1e16:
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
    print("constraint_Carr result [CC3] = {:.2e}".format(constraint_Carr([m_p], m_pbh_values, f_max, CC3, [alphas_CC3[i], betas[i]], evolved=False)[0]))
    
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
    print("constraint_Carr result [LN] = {:.2e}".format(constraint_Carr([m_p*np.exp(sigmas_LN[i]**2)], m_pbh_values, f_max, LN, [sigmas_LN[i]], evolved=False)[0]))

#%% Version for approximate GC photon constraints

def extract_GC_Isatis(j, f_max_Isatis, exponent_PL_lower=2):
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
    
    Delta = 5
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    # Constraints for monochromatic MF
                
    # Power-law exponent to use between 1e11g and 1e13g.
    exponent_PL_lower = 2
    
    # Constraints from Galactic Centre photons from Isatis (see Auffinger (2022) [2201.01265])
    
    # Select which instrument places the tightest constraint, for the evolved MF constraints (depends on Delta and the peak mass). Values are for a peak mass m_p = 1e16g
    # 0 for COMPTEL, 1 for EGRET, 2 for Fermi-LAT, 3 for INTEGRAL
    # Note: the defaults here apply to the evolved MF constraints and a peak mass m_p = 1e16g, and do not depend on the fitting function at m_p = 1e16g 
    if Delta == 0:
        j = 0
    elif Delta == 2:
        j = 1
    elif Delta == 5:
        j = 1
    
    constraints_names, f_max_Isatis = load_results_Isatis(modified=True)
    colours_GC_fit = ["tab:orange", "tab:green", "tab:red", "tab:blue"]
    
    f_max, m_pbh_values = extract_GC_Isatis(j, f_max_Isatis, exponent_PL_lower)
    # For the unevolved MF constraints, j=2 when m_p = 1e16g
    f_max_FermiLAT, m_pbh_values_FermiLAT = extract_GC_Isatis(2, f_max_Isatis, exponent_PL_lower)
        
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
    n_steps = 1000
    #m_init_values_input = np.sort(np.concatenate((np.logspace(np.log10(min(m_pbh_values)), np.log10(m_star), n_steps), np.arange(m_star, m_star*(1+1e-11), 5e2), np.arange(m_star*(1+1e-11), m_star*(1+1e-6), 1e7), np.logspace(np.log10(m_star*(1+1e-4)), np.log10(max(m_pbh_values))+4, n_steps))))
    
    mf_LN_init = LN(m_pbh_values_init, mc_LN, sigma=sigmas_LN[i])
    mf_SLN_init = SLN(m_pbh_values_init, mc_SLN, sigma=sigmas_SLN[i], alpha=alphas_SLN[i])
    mf_CC3_init = CC3(m_pbh_values_init, m_p, alpha=alphas_CC3[i], beta=betas[i])
    
    m_pbh_values_evolved = mass_evolved(m_pbh_values_init, t_0)
    mf_LN_evolved = 10**np.interp(np.log10(m_pbh_values), np.log10(m_pbh_values_evolved), np.log10(psi_evolved_normalised(mf_LN_init, m_pbh_values_evolved, m_pbh_values_init)))
    mf_SLN_evolved = 10**np.interp(np.log10(m_pbh_values), np.log10(m_pbh_values_evolved), np.log10(psi_evolved_normalised(mf_SLN_init, m_pbh_values_evolved, m_pbh_values_init)))
    mf_CC3_evolved = 10**np.interp(np.log10(m_pbh_values), np.log10(m_pbh_values_evolved), np.log10(psi_evolved_normalised(mf_CC3_init, m_pbh_values_evolved, m_pbh_values_init)))

    # Unevolved MF, evaluated at the masses the delta-function MF constraint is evaluated at
    if Delta == 5:
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
    
    if Delta == 5:
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

    ax.set_xlim(0, 5e15)
    
    if Delta == 0:
        ax.set_xlim(0, 2e16)
       
    if Delta == 2:
        ax.set_ylim(0, 5e-12)

    elif Delta == 5:
        ax.set_ylim(0, 1e-11)
        
    ax.set_title("GC Isatis constraints, $\Delta={:.1f},~m_p={:.0e}$".format(Deltas[i], m_p) + "$~\mathrm{g}$", fontsize="x-small")
    ax.set_ylabel("$\psi_\mathrm{N} / f_\mathrm{max}~[\mathrm{g}^{-1}]$")
    
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
            
    if m_p > 1e17:
        axes[0].set_ylim(1e-24, 3e-21)
    else:
        axes[0].set_ylim(1e-23, 1e-15)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("$\psi_\mathrm{N}~[\mathrm{g}^{-1}]$")
                   
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
    
    if Delta == 5:
        axes[2].plot(m_pbh_values_FermiLAT, mf_SLN_unevolved / f_max_FermiLAT, color="b", linestyle="dotted")
        axes[2].plot(m_pbh_values_FermiLAT, mf_CC3_unevolved / f_max_FermiLAT, color="g", linestyle="dotted")
        axes[2].plot(m_pbh_values_FermiLAT, mf_LN_unevolved / f_max_FermiLAT, color="r", linestyle="dotted")

    else:
        axes[2].plot(m_pbh_values, mf_SLN_unevolved / f_max, color="b", linestyle="dotted")
        axes[2].plot(m_pbh_values, mf_CC3_unevolved / f_max, color="g", linestyle="dotted")
        axes[2].plot(m_pbh_values, mf_LN_unevolved / f_max, color="r", linestyle="dotted")
    
    axes[2].set_ylabel("$\psi_\mathrm{N} / f_\mathrm{max}~[\mathrm{g}^{-1}]$")
    axes[2].set_xlabel("$m~[\mathrm{g}]$")
    
    fig.suptitle("GC photon constraints, $\Delta={:.0f}$".format(Delta))

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
    print("constraint_Carr result [CC3] = {:.2e}".format(constraint_Carr([m_p], m_pbh_values, f_max, CC3, [alphas_CC3[i], betas[i]], evolved=True)[0]))

    integral_lower = np.trapz(mf_SLN_evolved[m_pbh_values<m_min] / f_max[m_pbh_values<m_min], m_pbh_values[m_pbh_values<m_min])
    integral_upper = np.trapz(mf_SLN_evolved[m_pbh_values<m_max] / f_max[m_pbh_values<m_max], m_pbh_values[m_pbh_values<m_max])
    integral_total = np.trapz(mf_SLN_evolved / f_max, m_pbh_values)
    print("\n Evolved MFs")
    print("Integral (M < {:.1e}g) / total integral [SLN] = {:.3f}".format(m_min, integral_lower / integral_total))
    print("Integral ({:.1e}g < M < {:.1e}g) / total integral [SLN] = {:.3f}".format(m_min, m_max, (integral_upper - integral_lower) / integral_total))

    integral_lower = np.trapz(mf_LN_evolved[m_pbh_values<m_min] / f_max[m_pbh_values<m_min], m_pbh_values[m_pbh_values<m_min])
    integral_upper = np.trapz(mf_LN_evolved[m_pbh_values<m_max] / f_max[m_pbh_values<m_max], m_pbh_values[m_pbh_values<m_max])
    integral_total = np.trapz(mf_LN_evolved / f_max, m_pbh_values)
    print("\n Evolved MFs")
    print("Integral (M < {:.1e}g) / total integral [LN] = {:.3f}".format(m_min, integral_lower / integral_total))
    print("Integral ({:.1e}g < M < {:.1e}g) / total integral [LN] = {:.3f}".format(m_min, m_max, (integral_upper - integral_lower) / integral_total))
    print("1 / total integral [LN] = {:.2e}".format(1 / integral_total))
    print("constraint_Carr result [LN] = {:.2e}".format(constraint_Carr([m_p*np.exp(sigmas_LN[i]**2)], m_pbh_values, f_max, LN, [sigmas_LN[i]], evolved=True)[0]))
    
    # For initial MFs, restrict the mass range to that loaded from the data itself
        
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
    print("constraint_Carr result [CC3] = {:.2e}".format(constraint_Carr([m_p], m_pbh_values, f_max, CC3, [alphas_CC3[i], betas[i]], evolved=False)[0]))
    
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
    print("constraint_Carr result [LN] = {:.2e}".format(constraint_Carr([m_p*np.exp(sigmas_LN[i]**2)], m_pbh_values, f_max, LN, [sigmas_LN[i]], evolved=False)[0]))

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

   
#%% Calculate the constraint in the simplified problem using GC_constraint_Carr, and compare to f_max evaluated at that peak mass. Plot results. 
 
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
    f_max_extrapolated_upper = min(f_max_loaded) * np.power(m_delta_extrapolated_upper / min(m_delta_loaded), exponent_PL_upper)
    f_max_extrapolated_lower = min(f_max_extrapolated_upper) * np.power(m_delta_extrapolated_lower / min(m_delta_extrapolated_upper), exponent_PL_lower)
    f_max_total = np.concatenate((f_max_extrapolated_lower, f_max_extrapolated_upper, f_max_loaded))
            
    # If True, use LN MF
    use_LN = True
    use_CC3 = not use_LN
                                
    if use_LN:
        
        mp_values = np.logspace(14.5, 17, 25)
        psi_initial = LN
        
        for i in range(len(Deltas)):
                    
            frac_diff_approx_both = []
            frac_diff_approx_fmax = []
            frac_diff_approx_LN = []
            frac_diff_full = []
            
            for m_p in mp_values:
                f_PBH_simplified = f_max_loaded[0] * np.power(m_p / m_delta_loaded[0], 2)   # Find value of the simplified constraint

                params = [sigmas_LN[i]]
                mc_values = [m_p*np.exp(sigmas_LN[i]**2)]
                
                f_PBH_approx_both = constraint_Carr(mc_values, m_delta=m_delta_total, f_max=f_max_simplified, psi_initial=psi_initial, params=params, evolved=False)
                f_PBH_approx_fmax = constraint_Carr(mc_values, m_delta=m_delta_total, f_max=f_max_simplified, psi_initial=psi_initial, params=params, evolved=True)
                f_PBH_approx_LN = constraint_Carr(mc_values, m_delta=m_delta_total, f_max=f_max_total, psi_initial=psi_initial, params=params, evolved=False)
                f_PBH_full = constraint_Carr(mc_values, m_delta=m_delta_total, f_max=f_max_total, psi_initial=psi_initial, params=params, evolved=True)
                
                frac_diff_approx_both.append(f_PBH_approx_both[0] / f_PBH_simplified - 1)
                frac_diff_approx_fmax.append(f_PBH_approx_fmax[0] / f_PBH_simplified - 1)
                frac_diff_approx_LN.append(f_PBH_approx_LN[0] / f_PBH_simplified - 1)
                frac_diff_full.append(f_PBH_full[0] / f_PBH_simplified - 1)
            
            fig, ax = plt.subplots(figsize=(9,6))
            ax.plot(mp_values, frac_diff_full, marker="x", linestyle="None", label="$f_\mathrm{max}$ from KP '23, evolved $\psi_\mathrm{N}$", color="k")
            ax.plot(mp_values, frac_diff_approx_fmax, marker="+", linestyle="None", label="$f_\mathrm{max} \propto m^2$, evolved $\psi_\mathrm{N}$", color="k")
            ax.plot(mp_values, frac_diff_approx_LN, marker="x", linestyle="None", label="$f_\mathrm{max}$ from KP '23, unevolved $\psi_\mathrm{N}$", color="tab:red")
            #ax.plot(mp_values, frac_diff_approx_both, marker="+", linestyle="None", label="$f_\mathrm{max} \propto m^2$, unevolved $\psi_\mathrm{N}$", color="tab:red")
            ax.legend(title="Frac. diff. from $f_\mathrm{max}(m_p)$", fontsize="x-small")
            ax.set_xlabel("$m_p~[\mathrm{g}]$")
            ax.set_ylabel("$\Delta f_\mathrm{PBH} / f_\mathrm{PBH}$")
            ax.set_ylim(-0.2, 0.2)
            ax.set_xlim(min(mp_values), max(mp_values))
            ax.grid()
            ax.set_xscale("log")
            ax.set_title("LN MF, $\Delta={:.0f}$".format(Deltas[i]))
            fig.tight_layout()
            fig.savefig("Frac_diff_fPBH_LN_Delta={:.1f}.png".format(Deltas[i]))
        
        
        m_p = 1e15
        f_PBH_simplified = f_max_loaded[0] * np.power(m_p / m_delta_loaded[0], 2)   # Find value of the simplified constraint
    
        frac_diff_approx_both = []
        frac_diff_approx_fmax = []
        frac_diff_approx_LN = []
        frac_diff_full = []
    
        for i in range(len(Deltas)):
                           
            psi_initial = LN
            params=[sigmas_LN[i]]
            mc_values = [m_p * np.exp(sigmas_LN[i]**2)]
            
            #f_PBH_approx_both = constraint_Carr(mc_values, m_delta=m_delta_total, f_max=f_max_simplified, psi_initial=psi_initial, params=params, evolved=False)
            f_PBH_approx_fmax = constraint_Carr(mc_values, m_delta=m_delta_total, f_max=f_max_simplified, psi_initial=psi_initial, params=params, evolved=True)
            f_PBH_approx_LN = constraint_Carr(mc_values, m_delta=m_delta_total, f_max=f_max_total, psi_initial=psi_initial, params=params, evolved=False)
            f_PBH_full = constraint_Carr(mc_values, m_delta=m_delta_total, f_max=f_max_total, psi_initial=psi_initial, params=params, evolved=True)
            
            #frac_diff_approx_both.append(f_PBH_approx_both[0] / f_PBH_simplified - 1)
            frac_diff_approx_fmax.append(f_PBH_approx_fmax[0] / f_PBH_simplified - 1)
            frac_diff_approx_LN.append(f_PBH_approx_LN[0] / f_PBH_simplified - 1)
            frac_diff_full.append(f_PBH_full[0] / f_PBH_simplified - 1)
            
        fig, ax = plt.subplots(figsize=(9,6))   
        ax.plot(Deltas, frac_diff_full, marker="x", markersize=10, linestyle="None", label="f_\mathrm{max}$ from KP '23, evolved $\psi_\mathrm{N}$", color="k")
        ax.plot(Deltas, frac_diff_approx_fmax, marker="+", markersize=10, linestyle="None", label="$f_\mathrm{max} \propto m^2$, evolved $\psi_\mathrm{N}$", color="k")
        ax.plot(Deltas, frac_diff_approx_LN, marker="x", linestyle="None", label="Full $f_\mathrm{max}$, unevolved $\psi_\mathrm{N}$", color="tab:red")
        #ax.plot(Deltas, frac_diff_approx_both, marker="+", linestyle="None", label="$f_\mathrm{max} \propto m^2$, LN $\psi_\mathrm{N}$", color="tab:red")
        ax.legend(title="Frac. diff. from $f_\mathrm{max}(m_p)$", fontsize="x-small")
        ax.set_xlabel("$\Delta$")
        ax.set_ylabel("$\Delta f_\mathrm{PBH} / f_\mathrm{PBH}$")
        ax.set_ylim(-0.2, 1)
        ax.grid()
        ax.set_title("$m_p$ = {:.1e} g".format(m_p))
        fig.tight_layout()
        
    elif use_CC3:
        
        mp_values = np.logspace(15, 17, 20)
        psi_initial = CC3
        
        for i in [4,5,6]:
                    
            frac_diff_approx_both = []
            frac_diff_approx_fmax = []
            frac_diff_approx_LN = []
            frac_diff_full = []
            
            for m_p in mp_values:
                f_PBH_simplified = f_max_loaded[0] * np.power(m_p / m_delta_loaded[0], 2)   # Find value of the simplified constraint

                params = [alphas_CC3[i], betas[i]]
                mc_values = [m_p]
                
                f_PBH_approx_both = constraint_Carr(mc_values, m_delta=m_delta_total, f_max=f_max_simplified, psi_initial=psi_initial, params=params, evolved=False)
                f_PBH_approx_fmax = constraint_Carr(mc_values, m_delta=m_delta_total, f_max=f_max_simplified, psi_initial=psi_initial, params=params, evolved=True)
                f_PBH_approx_LN = constraint_Carr(mc_values, m_delta=m_delta_total, f_max=f_max_total, psi_initial=psi_initial, params=params, evolved=False)
                f_PBH_full = constraint_Carr(mc_values, m_delta=m_delta_total, f_max=f_max_total, psi_initial=psi_initial, params=params, evolved=True)
                
                frac_diff_approx_both.append(f_PBH_approx_both[0] / f_PBH_simplified - 1)
                frac_diff_approx_fmax.append(f_PBH_approx_fmax[0] / f_PBH_simplified - 1)
                frac_diff_approx_LN.append(f_PBH_approx_LN[0] / f_PBH_simplified - 1)
                frac_diff_full.append(f_PBH_full[0] / f_PBH_simplified - 1)
            
            fig, ax = plt.subplots(figsize=(9,6))
            ax.plot(mp_values, frac_diff_full, marker="x", linestyle="None", label="Full calculation", color="k")
            ax.plot(mp_values, frac_diff_approx_fmax, marker="+", linestyle="None", label="f_\mathrm{max}$ from KP '23, evolved $\psi_\mathrm{N}$", color="k")
            ax.plot(mp_values, frac_diff_approx_LN, marker="x", linestyle="None", label="$f_\mathrm{max}$ from KP '23, CC3 $\psi_\mathrm{N}$", color="tab:green")
            #ax.plot(mp_values, frac_diff_approx_both, marker="+", linestyle="None", label="$f_\mathrm{max} \propto m^2$, CC3 $\psi_\mathrm{N}$", color="tab:green")
            ax.legend(title="Frac. diff. from $f_\mathrm{max}(m_p)$", fontsize="x-small")
            ax.set_xlabel("$m_p~[\mathrm{g}]$")
            ax.set_ylabel("$\Delta f_\mathrm{PBH} / f_\mathrm{PBH}$")
            ax.set_ylim(-0.1, 0.2)
            ax.set_xlim(min(mp_values), max(mp_values))
            ax.grid()
            ax.set_xscale("log")
            ax.set_title("CC3 MF, $\Delta={:.0f}$".format(Deltas[i]))
            fig.tight_layout()
         
        m_p = 1e17
        f_PBH_simplified = f_max_loaded[0] * np.power(m_p / m_delta_loaded[0], 2)   # Find value of the simplified constraint
    
        frac_diff_approx_both = []
        frac_diff_approx_fmax = []
        frac_diff_approx_LN = []
        frac_diff_full = []
    
        for i in range(len(Deltas)):
                           
            psi_initial = CC3
            params = [alphas_CC3[i], betas[i]]
            mc_values = [m_p]
            
            f_PBH_approx_both = constraint_Carr(mc_values, m_delta=m_delta_total, f_max=f_max_simplified, psi_initial=psi_initial, params=params, evolved=False)
            f_PBH_approx_fmax = constraint_Carr(mc_values, m_delta=m_delta_total, f_max=f_max_simplified, psi_initial=psi_initial, params=params, evolved=True)
            f_PBH_approx_LN = constraint_Carr(mc_values, m_delta=m_delta_total, f_max=f_max_total, psi_initial=psi_initial, params=params, evolved=False)
            f_PBH_full = constraint_Carr(mc_values, m_delta=m_delta_total, f_max=f_max_total, psi_initial=psi_initial, params=params, evolved=True)
            
            frac_diff_approx_both.append(f_PBH_approx_both[0] / f_PBH_simplified - 1)
            frac_diff_approx_fmax.append(f_PBH_approx_fmax[0] / f_PBH_simplified - 1)
            frac_diff_approx_LN.append(f_PBH_approx_LN[0] / f_PBH_simplified - 1)
            frac_diff_full.append(f_PBH_full[0] / f_PBH_simplified - 1)
            
        fig, ax = plt.subplots(figsize=(9,6))   
        ax.plot(Deltas, frac_diff_full, marker="x", markersize=10, linestyle="None", label="f_\mathrm{max}$ from KP '23, evolved $\psi_\mathrm{N}$", color="k")
        ax.plot(Deltas, frac_diff_approx_fmax, marker="+", markersize=10, linestyle="None", label="$f_\mathrm{max} \propto m^2$, evolved $\psi_\mathrm{N}$", color="k")
        ax.plot(Deltas, frac_diff_approx_LN, marker="x", linestyle="None", label="$f_\mathrm{max}$ from KP '23, CC3 $\psi_\mathrm{N}$", color="tab:green")
        ax.plot(Deltas, frac_diff_approx_both, marker="+", linestyle="None", label="$f_\mathrm{max} \propto m^2$, CC3 $\psi_\mathrm{N}$", color="tab:green")
        ax.legend(title="Frac. diff. from $f_\mathrm{max}(m_p)$", fontsize="x-small")
        ax.set_xlabel("$\Delta$")
        ax.set_ylabel("$\Delta f_\mathrm{PBH} / f_\mathrm{PBH}$")
        ax.set_ylim(-0.2, 1)
        ax.grid()
        ax.set_title("$m_p$ = {:.1e} g".format(m_p))
        fig.tight_layout()

#%% Plot fractional difference between the delta-function MF constraint and a power-law in m^2.

if "__main__" == __name__:
        
    fig, ax = plt.subplots(figsize=(7,7))
    
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
    
    # set x-axis and y-axis ticks
    # see https://stackoverflow.com/questions/30887920/how-to-show-minor-tick-labels-on-log-scale-with-matplotlib
    
    y_major = mpl.ticker.LinearLocator(numticks = 11)
    ax.yaxis.set_major_locator(y_major)
    y_minor = mpl.ticker.LinearLocator(numticks = 11)
    ax.yaxis.set_minor_locator(y_minor)
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    
    ax.plot(m_pbh_values, f_max / (min(f_max_loaded) * np.power(m_pbh_values / min(m_delta_values_loaded), exponent_PL_upper)) - 1, color=(0.5294, 0.3546, 0.7020))
    ax.set_ylabel("$f_\mathrm{max} / (km^2) - 1$")
    ax.set_xlabel("$m~[\mathrm{g}]$")
    ax.set_xlim(1e16, 1e17)
    ax.set_ylim(0, 1)
    ax.grid()
    fig.tight_layout()
    fig.savefig("./Tests/Figures/EMF_constraints_work/fmax_fracdifff_from_m^2.png")

#%% Plot the CC3 at small and large masses, compare to asymptotic fits at m << m_p and m >> m_p.

if "__main__" == __name__:
    
    # Mass function parameter values, from 2009.03204.
    [Deltas, sigmas_LN, ln_mc_SL, mp_SL, sigmas_SLN, alphas_SLN, mp_CC, alphas_CC, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)

    for Delta_index in range(len(Deltas)):
        
        print(Delta_index)
        
        fig, ax = plt.subplots(figsize=(6,6))
        m_p = 1e16
        m_pbh_values = np.logspace(np.log10(m_p)-5, np.log10(m_p)+5, 1000)
        
        alpha = alphas_CC3[Delta_index]
        beta = betas[Delta_index]
        prefactor = (beta / m_p) * np.power(alpha/beta, (alpha+1)/beta) / np.exp(loggamma((alpha+1) / beta))
        
        CC3_actual = CC3(m_pbh_values, m_p, alpha, beta)
        CC3_fit_lowmass = prefactor * np.power(m_pbh_values/m_p, alpha)
        CC3_fit_highmass = prefactor * np.exp(-(alpha/beta)*np.power(m_pbh_values/m_p, beta))
        
        ax.plot(m_pbh_values[CC3_actual > 0], CC3_actual[CC3_actual > 0])
        ax.plot(m_pbh_values, CC3_fit_lowmass, linestyle="dotted", label=r"$\propto (m / m_\mathrm{p})^{\alpha}$")
        ax.plot(m_pbh_values, CC3_fit_highmass, linestyle="dotted", label=r"$\propto \exp \left(-\frac{\alpha}{\beta} \left(\frac{m}{m_\mathrm{p}}\right)^\beta\right)$")
        ax.set_xlim(min(m_pbh_values), max(m_pbh_values))
        ax.set_ylim(CC3_actual[0], 10*max(CC3_actual))
        ax.set_title("CC3, $\Delta={:.1f}$".format(Deltas[Delta_index]))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$m~[\mathrm{g}]$")
        ax.set_ylabel("$\psi~[\mathrm{g}^{-1}]$")
        ax.legend(fontsize="xx-small")
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
            fig1, ax1 = plt.subplots(figsize=(6, 5.5))

            ax1.plot(m_pbh_values, mf_values, color="k", label="Numeric MF")
            #ax1.plot(m_pbh_values, mf_values[0] * np.power(m_pbh_values/m_pbh_values[0], PL_exp), color="k", linestyle="dotted")
            
            # Find m_c for the lognormal fit by finding the PBH mass where the numerical MF is maximal
            mp_LN = m_pbh_values[np.argmax(mf_values)]
            mc_LN = mp_LN * np.exp(sigmas_LN[i]**2)
            
            # Range of PBH mass values to show for the fitting functions
            m_pbh_values_fits = np.logspace(np.log10(min(m_pbh_values))-2, np.log10(max(m_pbh_values))+2, 1000)

            ax1.plot(m_pbh_values_fits, LN(m_pbh_values_fits, m_c=mc_LN, sigma=sigmas_LN[i]), color="r", dashes=[6, 2], label="LN")            
            ax1.plot(m_pbh_values_fits, SLN(m_pbh_values_fits, m_c=np.exp(ln_mc_SLN[i]), sigma=sigmas_SLN[i], alpha=alphas_SLN[i]), color="b", linestyle=(0, (5, 7)), label="SLN")
            ax1.plot(m_pbh_values_fits, CC3(m_pbh_values_fits, m_p=mp_CC3[i], alpha=alphas_CC3[i], beta=betas[i]), color="g", linestyle="dashed", label="CC3")
            # Plot the numerical MF obtained using mf_numeric(). Test the method when the booleans extrapolate_lower = extrapolate_upper_const = True
            ax1.plot(m_pbh_values_fits, mf_numeric(m_pbh_values_fits, mp_CC3[i], Deltas[i], normalised=True, extrap_lower=False, extrap_upper_const=True), color="k", linestyle="dotted")
            ax1.plot(m_pbh_values_fits, mf_numeric(m_pbh_values_fits, mp_CC3[i], Deltas[i], normalised=True), color="k", linestyle="solid")
                     
            ax1.set_xlabel(r"$m~[M_\odot]$")
            ax1.set_ylabel("$\psi(m)~[M_\odot^{-1}]$")
            ax1.set_title("$\Delta={:.1f}$".format(Deltas[i]))
            ax1.set_xlim(min(m_pbh_values)/5, 5*max(m_pbh_values))
            ax1.set_ylim(max(1e-8, min(mf_values)/10), 1e-1)
            ax1.set_xscale("log")
            ax1.set_yscale("log")
            ax1.legend(fontsize="xx-small")
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

#%% Plot and calculate fractional difference between the numerical MF using different extrapolations at masses larger than that for which data is available.

if "__main__" == __name__:
    
    # Peak mass values
    #mp_values = np.logspace(16, np.log10(5.05e17), 5)[0:1]
    #mp_values = np.logspace(14, 15, 10)
    mp_values = [1e22]    # Approximate minimum peak mass for which microlensing constraints apply
    
    # Initial and evolved PBH masses
    n_steps = 324
    # Range of masses to include for the delta-function MF constraint
    m_delta = [9.47e21, 5.38e28]
    # Initial masses matching those used in constraint_Carr() when calculating constraints for unevolved MFs.
    m_init_unevolved = np.logspace(np.log10(min(m_delta)), np.log10(max(m_delta)), n_steps)
    # Initial masses matching those used in constraint_Carr() when calculating constraints for evolved MFs.
    m_init_values = np.sort(np.concatenate((np.logspace(np.log10(min(m_delta)), np.log10(m_star), n_steps), np.arange(m_star, m_star*(1+1e-11), 5e2), np.arange(m_star*(1+1e-11), m_star*(1+1e-6), 1e7), np.logspace(np.log10(m_star*(1+1e-4)), np.log10(max(m_delta))+4, n_steps))))
    m_evolved_values = mass_evolved(m_init_values, t=t_0)
    
    for i in range(len(Deltas)):
        
        print("\nDelta={:.1f}".format(Deltas[i]))
        
        for m_p in mp_values:
            fig, ax = plt.subplots(figsize=(6, 5))
            
            mf_numeric_values_unevolved = mf_numeric(m_init_unevolved, m_p, Deltas[i], extrap_upper_const=True, normalised=True)
            mf_numeric_values_unevolved_no_extrap = mf_numeric(m_init_unevolved, m_p, Deltas[i], normalised=True)
           
            mf_numeric_values_init = mf_numeric(m_init_values, m_p, Deltas[i], extrap_upper_const=True, normalised=True)
            mf_numeric_values_evolved = psi_evolved_normalised(mf_numeric_values_init, m_evolved_values, m_init_values)
    
            mf_numeric_values_init_no_extrap = mf_numeric(m_init_values, m_p, Deltas[i], normalised=True)
            mf_numeric_values_evolved_no_extrap = psi_evolved_normalised(mf_numeric_values_init_no_extrap, m_evolved_values, m_init_values)      
    
            print("Fractional difference (initial MFs) = {:.1e}".format(max(mf_numeric_values_unevolved_no_extrap) / max(mf_numeric_values_unevolved) - 1))        
            print("Fractional difference (evolved MFs) = {:.3e}".format(max(mf_numeric_values_evolved_no_extrap) / max(mf_numeric_values_evolved) - 1))
        
            ax.plot(m_init_values, mf_numeric_values_init, linestyle="dotted", color="k", linewidth=3)
            ax.plot(m_evolved_values, mf_numeric_values_evolved, color="k", label="Numeric (extrapolated to large $m$)")
            ax.plot(m_init_values, mf_numeric_values_init_no_extrap, linestyle="dotted", color="tab:grey", linewidth=3)
            ax.plot(m_evolved_values, mf_numeric_values_evolved_no_extrap, color="tab:grey", label="Numeric (no extrapolation)")
            ax.set_ylabel(r"$\psi_{\rm N}(m)~[{\rm g}^{-1}]$")
            ax.set_xlabel(r"$m~[{\rm g}]$")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(min(m_init_values[mf_numeric_values_init > 0]), max(m_init_values[mf_numeric_values_init > 0]))
            ax.set_title(r"$m_{\rm p} = " + " {:.1e}".format(m_p) + r"~{\rm g}" + "~(\Delta={:.1f}$)".format(Deltas[i]))
            ax.tick_params(pad=7)
            ax.legend(fontsize="xx-small")
            x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
            ax.yaxis.set_major_locator(x_major)
            x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 5)
            ax.yaxis.set_minor_locator(x_minor)
            ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
            fig.tight_layout()
    
    
#%% Plot different ways of calculating the numerical MF, and the fractional difference from the 'true' value
if "__main__" == __name__:
    
    filepath = "./Data/psiData/"

    for i in range(len(Deltas)):
        
        # Load the data from the numerical MFs from Andrew Gow    
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

        fig, ax = plt.subplots(figsize=(6, 5.5))
        fig1, ax1 = plt.subplots(figsize=(8, 7))
        ax.plot(m_pbh_values, mf_values, color="k", label="Numeric MF")
        
        # Find m_c for the lognormal fit by finding the PBH mass where the numerical MF is maximal
        mp_LN = m_pbh_values[np.argmax(mf_values)]
        mc_LN = mp_LN * np.exp(sigmas_LN[i]**2)
        
        # Range of PBH mass values to show for the fitting functions
        m_pbh_values_fits = np.logspace(np.log10(min(m_pbh_values))-2, np.log10(max(m_pbh_values))+2, 500)

        ax.plot(m_pbh_values_fits, mf_numeric(m_pbh_values_fits, mp_SLN[i], Deltas[i], log_interp=False, extrap_lower=True), color="tab:orange", linestyle="dotted", label="Lin interp.")
        ax.plot(m_pbh_values_fits, mf_numeric(m_pbh_values_fits, mp_SLN[i], Deltas[i], log_interp=True, extrap_lower=True), linestyle="None", color="tab:orange", marker="+", label="Log interp.")
        ax.plot(m_pbh_values_fits, mf_numeric(m_pbh_values_fits, mp_CC3[i], Deltas[i], params=[alphas_CC3[i], betas[i]], normalise_to_CC3=True), color="g", linestyle="dashed", label="Normalised to CC3 max (lin interp.)")
        ax.plot(m_pbh_values_fits, mf_numeric(m_pbh_values_fits, mp_SLN[i], Deltas[i], params=[sigmas_SLN[i], alphas_SLN[i]], normalise_to_SLN=True, mc_SLN=np.exp(ln_mc_SLN[i])), color="b", linestyle="dashdot", label="Normalised to SLN max (lin interp.)")
        ax.plot(m_pbh_values_fits, mf_numeric(m_pbh_values_fits, mp_CC3[i], Deltas[i], params=[alphas_CC3[i], betas[i]], normalise_to_CC3=True, log_interp=True), linestyle="None", color="g", marker="x", label="Normalised to CC3 max (log interp.)")
        ax.plot(m_pbh_values_fits, mf_numeric(m_pbh_values_fits, mp_SLN[i], Deltas[i], params=[sigmas_SLN[i], alphas_SLN[i]], normalise_to_SLN=True, mc_SLN=np.exp(ln_mc_SLN[i]), log_interp=True), linestyle="None", color="b", marker="+", label="Normalised to SLN max (log interp.)")

        ax1.plot(m_pbh_values, np.abs(frac_diff(mf_values, mf_numeric(m_pbh_values_fits, mp_SLN[i], Deltas[i], log_interp=False), m_pbh_values, m_pbh_values_fits)), color="tab:orange", linestyle="dotted", label="Lin interp.")
        ax1.plot(m_pbh_values, np.abs(frac_diff(mf_values, mf_numeric(m_pbh_values_fits, mp_SLN[i], Deltas[i]), m_pbh_values, m_pbh_values_fits)), linestyle="None", color="tab:orange", marker="+", label="Log interp.")
        ax1.plot(m_pbh_values, np.abs(frac_diff(mf_values, mf_numeric(m_pbh_values_fits, mp_CC3[i], Deltas[i], params=[alphas_CC3[i], betas[i]], log_interp=False, normalise_to_CC3=True), m_pbh_values, m_pbh_values_fits)), color="g", linestyle="dashed", label="Normalised to CC3 max (lin interp.)")
        ax1.plot(m_pbh_values, np.abs(frac_diff(mf_values, mf_numeric(m_pbh_values_fits, mp_SLN[i], Deltas[i], params=[sigmas_SLN[i], alphas_SLN[i]], normalise_to_SLN=True, mc_SLN=np.exp(ln_mc_SLN[i]), log_interp=False), m_pbh_values, m_pbh_values_fits)), color="b", linestyle="dashdot", label="Normalised to SLN max (lin interp.)")
        ax1.plot(m_pbh_values, np.abs(frac_diff(mf_values, mf_numeric(m_pbh_values_fits, mp_CC3[i], Deltas[i], params=[alphas_CC3[i], betas[i]], normalise_to_CC3=True), m_pbh_values, m_pbh_values_fits)), linestyle="None", color="g", marker="x", label="Normalised to CC3 max (log interp.)")
        ax1.plot(m_pbh_values, np.abs(frac_diff(mf_values, mf_numeric(m_pbh_values_fits, mp_SLN[i], Deltas[i], params=[sigmas_SLN[i], alphas_SLN[i]], normalise_to_SLN=True, mc_SLN=np.exp(ln_mc_SLN[i])), m_pbh_values, m_pbh_values_fits)), linestyle="None", color="b", marker="+", label="Normalised to SLN max (log interp.)")
        """
        ax1.plot(m_pbh_values_fits, np.abs(frac_diff(mf_numeric(m_pbh_values_fits, mp_SLN[i], Deltas[i], log_interp=False), mf_values, m_pbh_values_fits, m_pbh_values)), color="tab:orange", linestyle="dotted", label="Lin interp.")
        ax1.plot(m_pbh_values_fits, np.abs(frac_diff(mf_numeric(m_pbh_values_fits, mp_SLN[i], Deltas[i]), mf_values, m_pbh_values_fits, m_pbh_values)), linestyle="None", color="tab:orange", marker="+", label="Log interp.")
        ax1.plot(m_pbh_values_fits, np.abs(frac_diff(mf_numeric(m_pbh_values_fits, mp_CC3[i], Deltas[i], params=[alphas_CC3[i], betas[i]], normalise_to_CC3=True, log_interp=False), mf_values, m_pbh_values_fits, m_pbh_values)), color="g", linestyle="dashed", label="Normalised to CC3 max (lin interp.)")
        ax1.plot(m_pbh_values_fits, np.abs(frac_diff(mf_numeric(m_pbh_values_fits, mp_SLN[i], Deltas[i], params=[sigmas_SLN[i], alphas_SLN[i]], normalise_to_SLN=True, mc_SLN=np.exp(ln_mc_SLN[i]), log_interp=False), mf_values, m_pbh_values_fits, m_pbh_values)), color="b", linestyle="dashdot", label="Normalised to SLN max (lin interp.)")
        ax1.plot(m_pbh_values_fits, np.abs(frac_diff(mf_numeric(m_pbh_values_fits, mp_CC3[i], Deltas[i], params=[alphas_CC3[i], betas[i]], normalise_to_CC3=True), mf_values, m_pbh_values_fits, m_pbh_values)), linestyle="None", color="g", marker="x", label="Normalised to CC3 max (log interp.)")
        ax1.plot(m_pbh_values_fits, np.abs(frac_diff(mf_numeric(m_pbh_values_fits, mp_SLN[i], Deltas[i], params=[sigmas_SLN[i], alphas_SLN[i]], normalise_to_SLN=True, mc_SLN=np.exp(ln_mc_SLN[i])), mf_values, m_pbh_values_fits, m_pbh_values)), linestyle="None", color="b", marker="+", label="Normalised to SLN max (log interp.)")
        """
        for a in [ax, ax1]:
            a.set_xlabel(r"$m~[M_\odot]$")
            a.set_title("$\Delta={:.1f}$".format(Deltas[i]))
            a.set_xlim(min(m_pbh_values), max(m_pbh_values))
            a.set_xscale("log")
            a.set_yscale("log")
            a.legend(fontsize="xx-small")
            a.tick_params(pad=7)
            
        ax.set_ylabel("$\psi(m)~[M_\odot^{-1}]$")
        ax.set_ylim(1e-6, 1e-1)

        ax1.set_ylabel("$|\Delta \psi / \psi|$")
        ax1.set_ylim(1e-4, 1)

        fig.tight_layout()
        fig1.tight_layout()

#%% Microlensing constraints with numeric MF: compare evolved to unevolved MF
if "__main__" == __name__:
    
    m_delta_Subaru, f_max_Subaru = load_data("2007.12697/Subaru-HSC_2007.12697_dx=5.csv")
    
    # Peak mass, in grams
    m_p = 1e22
    
    Delta = 0
    
    extrap_lower = False
    extrap_upper_const = False
    extrap_upper_PL = False
    normalised = True
    
    # Parameters for the numeric MF
    params_numeric = [Delta, extrap_lower, extrap_upper_const, extrap_upper_PL, normalised]
    
    f_pbh_Subaru_evolved = constraint_Carr([m_p], m_delta_Subaru, f_max_Subaru, mf_numeric, params_numeric, evolved=True)
    f_pbh_Subaru_unevolved = constraint_Carr([m_p], m_delta_Subaru, f_max_Subaru, mf_numeric, params_numeric, evolved=False)
    
    print("Using constraint_Carr()")
    print("f_PBH (evolved numeric MF) = {:.2e}".format(f_pbh_Subaru_evolved[0]))
    print("f_PBH (unevolved numeric MF) = {:.2e}".format(f_pbh_Subaru_unevolved[0]))
    
    print("\nRecalculated")    
    # Calculate numeric MF constraint without calling constraint_Carr()
    psi_initial_values = mf_numeric(m_delta_Subaru, m_p, Delta, normalised=normalised, extrap_upper_const=extrap_upper_const)
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

#%%
    # Initial masses matching those used in constraint_Carr() when calculating constraints for evolved MFs.
    n_steps = 1000
    m_init_values = np.sort(np.concatenate((np.logspace(np.log10(min(m_delta_Subaru)), np.log10(m_star), n_steps), np.arange(m_star, m_star*(1+1e-11), 5e2), np.arange(m_star*(1+1e-11), m_star*(1+1e-6), 1e7), np.logspace(np.log10(m_star*(1+1e-4)), np.log10(max(m_delta_Subaru))+4, n_steps))))

    psi_initial_values = mf_numeric(m_init_values, m_p, Delta, normalised=normalised)
    print(psi_initial_values[0:50])
    m_evolved = mass_evolved(m_init_values, t=t_0)
    psi_evolved_values = psi_evolved_normalised(psi_initial_values, m_evolved, m_init_values)
    #psi_evolved_values_interpolated = 10**np.interp(np.log10(m_delta_Subaru), np.log10(m_evolved), np.log10(psi_evolved_values), left=-100, right=-100)
    psi_evolved_values_interpolated = 10**np.interp(np.log10(m_delta_Subaru), np.log10(m_evolved), np.log10(psi_evolved_values), left=-np.infty, right=-np.infty)   
    print(len(psi_evolved_values_interpolated[psi_evolved_values_interpolated>0]))
    f_pbh_Subaru_evolved = 1 / np.trapz(psi_evolved_values_interpolated / f_max_Subaru, m_delta_Subaru)

    print("f_PBH (evolved numeric MF following constraint_Carr) = {:.2e}".format(f_pbh_Subaru_evolved))

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
            
    #%% Illustrate issue of using trapezoidal rule to integrate non-continuous functions
    
    x_values = np.arange(0, 10)
    y_values = np.concatenate((np.zeros(3), np.ones(4), np.zeros(3)))
    print(np.trapz(y_values, x_values))
    print(np.trapz(y_values[y_values > 0], x_values[y_values > 0]))
    
    x_values_2 = [0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9]
    y_values_2 = np.concatenate((np.zeros(4), np.ones(4), np.zeros(4)))

    print(np.trapz(y_values_2, x_values_2))

    plt.plot(x_values_2, y_values_2)
    plt.plot(x_values, y_values)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()