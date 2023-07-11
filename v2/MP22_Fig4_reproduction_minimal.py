#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 14:34:47 2023

@author: ppxmg2
"""

"""A minimal version of the code required to reproduce Fig. 4 of Mosbech & Picker (2022), 
for an evolved mass function, to aid with understanding."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from preliminaries import load_data, LN

# Specify the plot style
mpl.rcParams.update({'font.size': 16,'font.family':'serif'})
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

plt.style.use('tableau-colorblind10')
filepath = './Extracted_files/'

m_Pl = 2.176e-5    # Planck mass, in grams
t_Pl = 5.391e-44    # Planck time, in seconds
t_0 = 13.8e9 * 365.25 * 86400    # Age of Universe, in seconds

#%%

def alpha_eff_extracted(M0_values):
    """
    Result for alpha_eff, extracted from Fig. 1 of Mosbech & Picker (2022).

    Parameters
    ----------
    M0_values : Array-like
        PBH formation masses, in grams.

    Returns
    -------
    Array-like.
        Value of alpha_eff.

    """
    M0_extracted, alpha_eff_extracted_data = load_data("2203.05743/2203.05743_Fig1.csv")
    # Value assigned at large masses equals that of the fitting function at M >~ 1e18g in Eq. 10 of Mosbech & Picker (2022), in turn from Page (1976).
    alpha_eff_values = np.interp(M0_values, M0_extracted, alpha_eff_extracted_data, left=max(alpha_eff_extracted_data), right=2.011e-4)
    return alpha_eff_values


def m_pbh_evolved_MP22(M0_values, t):
    """
    Find the PBH mass at time t, evolved from initial masses M0_values.

    Parameters
    ----------
    M0_values : Array-like
        Initial PBH masses.
    t : Float
        Time (after Big Bang) at which to evaluate PBH masses.

    Returns
    -------
    Array-like
        PBH mass at time t.

    """
    # Find the PBH mass at time t, evolved from initial masses M0_values
    M_values = []
    
    for M_0 in M0_values:
        if M_0**3 - 3 * alpha_eff_extracted(np.array([M_0])) * m_Pl**3 * (t / t_Pl) <= 0:
            M_values.append(0)
        else:
            # By default, alpha_eff_mixed() takes array-like quantities as arguments.
            # Choose the 'zeroth' entry to append a scalar to the list M_values.
            M_values.append(np.power(M_0**3 - 3 * alpha_eff_extracted(np.array([M_0])) * m_Pl**3 * (t / t_Pl), 1/3)[0])
    
    return np.array(M_values)


def psi_LN_number_density(m, m_c, sigma, log_m_factor=3, n_steps=1000):
    """
    Distribution function for PBH energy density, when the number density follows a log-normal in the mass.

    Parameters
    ----------
    m : Array-like
        PBH mass, in grams.
    m_c : Float
        Characteristic mass of the initial log-normal distribution in the number density.
    sigma : Float
        Standard deviation of the initial log-normal distribution in the number density.
    log_m_factor : Float, optional
        Number of multiples of sigma (in log-space) of masses around m_c to consider when estimating the maximum. The default is 3.
    n_steps : Integer, optional
        Number of masses at which to evaluate the evolved mass function. The default is 1000.

    Returns
    -------
    Array-like
        Evolved PBH mass density distribution, evaluated at time t and masses m.

    """
    log_m_min = np.log10(m_c) - log_m_factor*sigma
    log_m_max = np.log10(m_c) + log_m_factor*sigma

    m_pbh_values = np.logspace(log_m_min, log_m_max, n_steps)
    normalisation = 1 / np.trapz(LN(m_pbh_values, m_c, sigma) * m_pbh_values, m_pbh_values)
    return LN(m, m_c, sigma) * m * normalisation


def psi_evolved_direct(psi_formation, M_values, M0_values):
    """
    PBH mass function (in terms of the mass density) at time t, evolved form 
    the initial MF phi_formation using Eq. 11 of Mosbech & Picker (2022).   

    Parameters
    ----------
    psi_formation : Array-like
        Initial PBH mass distribution (in mass density).
    M_values : Array-like
        PBH masses at time t.
    M0_values : Array-like
        Initial PBH masses.

    Returns
    -------
    Array-like
        Evolved values of the PBH mass density distribution function (not normalised to unity).

    """
    return psi_formation * (M_values / M0_values)**3


def psi_evolved_direct_normalised(psi_formation, M_values, M0_values):
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
    M0_values : Array-like
        Initial PBH masses.

    Returns
    -------
    Array-like
        Evolved values of the PBH mass density distribution function (normalised to unity).

    """
    return psi_evolved_direct(psi_formation, M_values, M0_values) / np.trapz(psi_evolved_direct(psi_formation, M_values, M0_values), M_values)


#%% Plot constraints for extended MF (reproducing Fig. 4 of Mosbech & Picker (2022)), using direct calculation of psi

if "__main__" == __name__:
    # Constraints data for each energy bin of each instrument (extended MF)
    
    constraints_mono_file_lower = np.transpose(np.genfromtxt("./Data/fPBH_GC_full_all_bins_Fermi-LAT_1512.01846_lower_monochromatic_wide.txt"))
    constraints_mono_file_upper = np.transpose(np.genfromtxt("./Data/fPBH_GC_full_all_bins_Fermi-LAT_1512.01846_upper_monochromatic_wide.txt"))
        
    M_values_eval = np.logspace(10, 18, 100)   # masses at which the constraint is evaluated for a delta-function MF
    mc_values_evolved = np.logspace(13, 17, 50)[5:]
    
    M0_values_input = np.concatenate((np.arange(7.473420349255e+14, 7.4734203494e+14, 5e2), np.arange(7.4734203494e+14, 7.47344e+14, 1e7), np.logspace(np.log10(7.474e14), 18, 500)))
    M_values_input = m_pbh_evolved_MP22(M0_values_input, t_0)

    # Constraint from each energy bin
    f_PBH_energy_bin_lower = []
    f_PBH_energy_bin_upper = []
    
    normalised_unity = False    
    sigma = 0.1
    
    params_LN = [sigma]
    
    # Evolved mass function
    constraint_lower_evolved = []
    constraint_upper_evolved = []
        
    for m_c in mc_values_evolved:
        
        psi_initial = psi_LN_number_density(M0_values_input, m_c, sigma)
        
        # Evolved mass function
        if normalised_unity:
            psi_evolved = psi_evolved_direct_normalised(psi_initial, M_values_input, M0_values_input)
        else:
            psi_evolved = psi_evolved_direct(psi_initial, M_values_input, M0_values_input)
            
        # Interpolate evolved mass function at the evolved masses at which the delta-function MF constraint is calculated
        M_values_input_nozeros = M_values_input[psi_evolved > 0]
        psi_evolved_nozeros = psi_evolved[psi_evolved > 0]
        mf_evolved_interp = 10**np.interp(np.log10(M_values_eval), np.log10(M_values_input_nozeros), np.log10(psi_evolved_nozeros), left=-100, right=-100)
        
        # Constraint from each energy bin
        f_PBH_energy_bin_lower = []
        for k in range(len(constraints_mono_file_lower)):
    
            # Constraint from a particular energy bin (delta function MF)
            constraint_energy_bin = constraints_mono_file_lower[k]
            
            integrand = mf_evolved_interp / constraint_energy_bin
            integral = np.trapz(np.nan_to_num(integrand), M_values_eval)

            if integral == 0 or np.isnan(integral):
                f_PBH_energy_bin_lower.append(10)
            else:
                f_PBH_energy_bin_lower.append(1/integral)

        constraint_lower_evolved.append(min(f_PBH_energy_bin_lower))
        
        
        f_PBH_energy_bin_upper = []
        for k in range(len(constraints_mono_file_upper)):
    
            # Constraint from a particular energy bin (delta function MF)
            constraint_energy_bin = constraints_mono_file_upper[k]
            
            integrand = mf_evolved_interp / constraint_energy_bin
            integral = np.trapz(np.nan_to_num(integrand), M_values_eval)
            
            if integral == 0 or np.isnan(integral):
                f_PBH_energy_bin_upper.append(10)
            else:
                f_PBH_energy_bin_upper.append(1/integral)
        constraint_upper_evolved.append(min(f_PBH_energy_bin_upper))