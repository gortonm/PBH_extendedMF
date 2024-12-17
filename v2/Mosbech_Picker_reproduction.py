#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 11:55:35 2023

@author: ppxmg2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from preliminaries import load_data, LN
from extended_MF_checks import envelope, constraint_Carr, load_results_Isatis
import os

# Script for reproducing results from Mosbech & Picker (2022) [arXiv:2203.05743v2]

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
m_star = 7.473420349255e+14    # Formation mass of a PBH with a lifetimt equal to the age of the Universe, in grams.

#%%
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


# Calculate alpha_eff directly from BlackHawk
# Put this here because it is used by later methods (more efficient than recalculating each time other methods are called)
alpha_eff_values_BlackHawk = alpha_eff(np.array(pbh_lifetimes), m_pbh_values_formation_BlackHawk)


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
            #M_values.append(np.power(M_0**3 - 3 * alpha_eff_mixed(np.array([M_0])) * m_Pl**3 * (t / t_Pl), 1/3)[0])
            M_values.append(np.power(M_0**3 - 3 * alpha_eff_extracted(np.array([M_0])) * m_Pl**3 * (t / t_Pl), 1/3)[0])
    
    return np.array(M_values)


def m_pbh_formation_MP22(M_values, t):
    """
    Find formation mass in terms of the masses M_values at time t.

    Parameters
    ----------
    M_values : Array-like
        PBH masses at time t.
    t : Float
        Time (after Big Bang) at which PBH masses in M_values are evaluated.

    Returns
    -------
    M0_values : Array-like
        Initial PBH masses.

    """
    # Smallest initial PBH mass to consider corresponds to the initial mass of a PBH with a lifetime equal to the age of the Universe (BlackHawk defaults, using the default energy range of primary particles when using PYTHIA hadronisation tables).
    M0_min = 7.56e14
    M0_test_values = np.logspace(np.log10(M0_min), 18, 1000)
    M_evolved_test_values = m_pbh_evolved_MP22(M0_test_values, t)

    # Logarithmically interpolate to estimate the formation mass M0_values (y-axis) corresponding to present mass M_values (x-axis)
    M0_values = np.interp(x=M_values, xp=M_evolved_test_values, fp=M0_test_values)
    return M0_values


def phi_LN(m, m_c, sigma):
    """
    Log-normal number density distribution of PBHs.

    Parameters
    ----------
    m : Array-like
        PBH mass, in grams.
    m_c : Float
        Characteristic PBH mass, in grams.
    sigma : Float
        Standard deviation of the distribution.

    Returns
    -------
    Array-like
        Values of the PBH number density distribution function.

    """
    return LN(m, m_c, sigma)


def phi_evolved(phi_formation, M_values, t):
    """
    PBH mass function at time t, evolved form the initial MF phi_formation 
    using Eq. 11 of Mosbech & Picker (2022).

    Parameters
    ----------
    phi_formation : Array-like
        Initial PBH mass distribution (in number density).
    M_values : Array-like
        PBH masses at time t.
    t : Float
        Time (after Big Bang) at which PBH masses in M_values are evaluated.

    Returns
    -------
    Array-like
        Evolved values of the PBH number density distribution function.

    """
    M0_values = m_pbh_formation_MP22(M_values, t)
    return phi_formation * M_values**2 * np.power(M_values**3 + 3 * alpha_eff_extracted(M0_values) * m_Pl**3 * (t / t_Pl), -2/3)


def phi_evolved_v2(phi_formation, M_values, M0_values, t):
    """
    PBH mass function at time t, evolved form the initial MF phi_formation 
    using Eq. 11 of Mosbech & Picker (2022).   

    Parameters
    ----------
    phi_formation : Array-like
        Initial PBH mass distribution (in number density).
    M_values : Array-like
        PBH masses at time t.
    M0_values : Array-like
        Initial PBH masses.
    t : Float
        Time (after Big Bang) at which PBH masses in M_values are evaluated.

    Returns
    -------
    Array-like
        Evolved values of the PBH number density distribution function.

    """
    # PBH mass function at time t, evolved form the initial MF phi_formation using Eq. 11 
    # In terms of the initial masses M
    return phi_formation * M_values**2 * np.power(M_values**3 + 3 * alpha_eff_extracted(M0_values) * m_Pl**3 * (t / t_Pl), -2/3)


def psi_evolved_LN_number_density(m_c, sigma, t, log_m_factor=3, n_steps=1000, log_output=True):
    """
    PBH mass distribution at time t, for a log-normal initial number density distribution
    with characteristic mass m_c and width sigma.

    Parameters
    ----------
    m_c : Float
        Characteristic mass of the initial log-normal distribution in the number density.
    sigma : Float
        Standard deviation of the initial log-normal distribution in the number density.
    t : Float
        Time (after Big Bang) at which to evaluate the distribution.
    log_m_factor : Float, optional
        Number of multiples of sigma (in log-space) of masses around m_c to consider when estimating the maximum. The default is 3.
    n_steps : Integer, optional
        Number of masses at which to evaluate the evolved mass function. The default is 1000.
    log_output : Boolean, optional
        If True, return the logarithm of the evolved masses and mass density distribution (helpful for interpolation). The default is True.

    Returns
    -------
    Array-like
        Evolved PBH masses, evaluated at time t in grams. If log_output=True, returns the logarithm of this quantity.
    Array-like
        Evolved PBH mass density distribution, evaluated at time t. If log_output=True, returns the logarithm of this quantity.

    """
    # Distribution function for PBH energy density, when the number density follows distribution phi_evolved, obtained
    # from an initially log-normal distribution

    log_m_min = np.log10(m_c) - log_m_factor*sigma
    log_m_max = np.log10(m_c) + log_m_factor*sigma
    
    # To accurately evaluate the evolved mass function at small PBH masses, need to include a dense sampling of initial masses close to M_* = 7.5e14g
    M0_test_values = np.sort(np.concatenate((np.arange(7.473420349255e+14, 7.4734203494e+14, 5e2), np.arange(7.4734203494e+14, 7.47344e+14, 1e7), np.logspace(log_m_min, log_m_max, n_steps))))
    M_test_values = m_pbh_evolved_MP22(M0_test_values, t)
    
    phi_formation = LN(M0_test_values, m_c, sigma)
    phi_evolved_values = phi_evolved_v2(phi_formation, M_test_values, M0_test_values, t)
       
    # Find mass * evolved values of phi (proportional to the mass distribution psi)
    psi_unnormalised = phi_evolved_values * M_test_values
    
    # Estimate the normalisation of psi
    psi_normalisation = 1 / np.trapz(phi_formation*M0_test_values, M0_test_values)
    
    if log_output:
        return np.log10(M_test_values), np.log10(psi_unnormalised * psi_normalisation)
    
    else:
        return M_test_values, psi_unnormalised * psi_normalisation


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


def Delta(l_min, l_max, b_min, b_max):
    """
    Calculate solid angle observed by a telescope.

    Parameters
    ----------
    l_min : Float
        Minimum galactic longitude, in radians.
    l_max : Float
        Maximum galactic longitude, in radians.
    b_min : Float
        Minimum galactic latitude, in radians.
    b_max : Float
        Maximum galactic latitude, in radians.

    Returns
    -------
    Delta : Float
        Solid angle observed, in steradians.

    """
    nb_angles = 100

    b, l = [], []
    for i in range(0, nb_angles):
        l.append(l_min + i*(l_max - l_min)/(nb_angles - 1))
        b.append(b_min + i*(b_max - b_min)/(nb_angles - 1))

    Delta = 0
    for i in range(0, nb_angles-1):
        for j in range(0, nb_angles-1):
            Delta += abs(np.cos(b[i])) * (l[i+1] - l[i]) * (b[j+1] - b[j])
    return Delta


#%% Reproduce Fig. 1 of Mosbech & Picker (2022), using different forms of alpha_eff.


if "__main__" == __name__:

    alpha_eff_extracted_values = alpha_eff_extracted(m_pbh_values_formation_wide)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(m_pbh_values_formation_BlackHawk, alpha_eff_values_BlackHawk, label="Calculated using BlackHawk")
    ax.plot(m_pbh_values_formation_wide, alpha_eff_extracted_values, linestyle="None", marker="x", label="Extracted (Fig. 1 MP '22)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Formation mass $M_0$~[g]")
    ax.set_ylabel(r"$\alpha_\mathrm{eff}$")
    ax.legend(fontsize="small")
    fig.tight_layout()
    
    # Plot derivative of alpha_eff (calculated approximately)
    alpha_eff_extracted_values = alpha_eff_extracted(m_pbh_values_formation_wide)
    derivative_approx = np.diff(alpha_eff_extracted_values) / np.diff(m_pbh_values_formation_wide)
    
    # = ħ * c^4 / G^2 in units of g^3 / s
    prefactor = 1.912e29
    t_0 = 13.7e9 * 365.25 * 86400  # age of Universe, in s
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(m_pbh_values_formation_wide[:-1], prefactor*t_0*np.abs(derivative_approx)/np.power(m_pbh_values_formation_wide[:-1], 2), linestyle="None", marker="x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Formation mass ${M_0} \, [{\rm g}]$")
    ax.set_ylabel(r"$\frac{\hbar c^4}{G^2}|\frac{{\rm d}\alpha_\mathrm{eff}}{\rm{d}M_0}| t_0/ {M_0}^2$")
    ax.legend(fontsize="small")
    fig.tight_layout()

    
    # Plot the present mass against formation mass
    fig, ax = plt.subplots(figsize=(6, 6))
    m_pbh_values_formation_plot = np.logspace(np.log10(5e14), 16, 500)
    ax.plot(m_pbh_values_formation_plot, m_pbh_values_formation_plot, linestyle="dotted", color="k", label="Formation mass = Present mass")
    ax.plot(m_pbh_values_formation_plot, m_pbh_evolved_MP22(m_pbh_values_formation_plot, t=t_0), marker="x", linestyle="None", label="Eq. 7 (MP '22)")
    
    # Test: plot formation mass against present mass
    m_evolved_test = m_pbh_evolved_MP22(m_pbh_values_formation_plot, t=t_0)
    m_formation_test = m_pbh_formation_MP22(m_evolved_test, t=t_0)
    ax.plot(m_formation_test, m_evolved_test, marker="+", linestyle="None", label="Inverting Eq. 7 (MP '22)")
    ax.set_xlabel("Formation mass $M_0$ [g]")
    ax.set_ylabel("Present mass $M$ [g]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(min(m_pbh_values_formation_plot), 1e16)
    ax.set_ylim(1e-1 * min(m_pbh_values_formation_plot), max(m_pbh_values_formation_plot))
    ax.legend()
    fig.tight_layout()
    
    # Plot the ratio of the present mass to the formation mass
    fig, ax = plt.subplots(figsize=(6, 6))
    m_pbh_values_formation_plot = np.logspace(np.log10(5e14), 16, 500)
    ax.plot(m_pbh_values_formation_plot, m_pbh_evolved_MP22(m_pbh_values_formation_plot, t=t_0) / m_pbh_values_formation_plot, marker="x", linestyle="None")
    ax.hlines(0.99, xmin=min(m_pbh_values_formation_plot), xmax=max(m_pbh_values_formation_plot), color="k", linestyle="dotted")
    ax.hlines(0.9, xmin=min(m_pbh_values_formation_plot), xmax=max(m_pbh_values_formation_plot), color="k", linestyle="dashed", alpha=0.75)
    ax.hlines(0.5, xmin=min(m_pbh_values_formation_plot), xmax=max(m_pbh_values_formation_plot), color="k", alpha=0.75)
    ax.set_xlabel("Formation mass $M_0$~[g]")
    ax.set_ylabel(r"$M / M_0$")
 
#%% Reproduce Fig. 2 of Mosbech & Picker (2022)

if "__main__" == __name__:
    
    m_pbh_values_formation = np.logspace(11, 17, 500)
    # Initial PBH mass values includes values close to the initial mass of a PBH with lifetime equal to the age of the Universe,
    # corresponding to evolved masses at t=t_0 as low as a few times 10^11 g.
    #m_pbh_values_formation_to_evolve = np.concatenate((np.arange(7.473420349255e+14, 7.4734203494e+14, 5e2), np.arange(7.4734203494e+14, 7.47344e+14, 1e7), np.logspace(np.log10(7.474e14), 17, 500)))
    m_pbh_values_formation_to_evolve = np.concatenate((np.arange(m_star, m_star*(1+1e-11), 5e2), np.arange(m_star*(1+1e-11), m_star*(1+1e-6), 1e7), np.logspace(np.log10(m_star*(1+1e-4)), 17, 500)))
    m_pbh_values_evolved = m_pbh_evolved_MP22(m_pbh_values_formation_to_evolve, t_0)
    m_c = 1e15
    
    for sigma in [0.1, 0.5, 1, 1.5]:
        
        if sigma < 1:
           m_extracted_evolved, phi_extracted_evolved = load_data("2203.05743/2203.05743_Fig2_sigma0p{:.0f}.csv".format(10*sigma))
        else:
           m_extracted_evolved, phi_extracted_evolved = load_data("2203.05743/2203.05743_Fig2_sigma1p{:.0f}.csv".format(10*(sigma-1)))

        
        phi_initial = phi_LN(m_pbh_values_formation, m_c, sigma)
        phi_initial_to_evolve = phi_LN(m_pbh_values_formation_to_evolve, m_c, sigma)
                
        phi_present = phi_evolved_v2(phi_initial_to_evolve, m_pbh_values_evolved, m_pbh_values_formation_to_evolve, t_0)
        # test that the "evolved" mass function at t=0 matches the initial mass function.
        phi_test = phi_evolved_v2(phi_initial_to_evolve, m_pbh_values_formation_to_evolve, m_pbh_values_formation_to_evolve, 0)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(m_pbh_values_formation, phi_initial, label="$t=0$")
        ax.plot(m_pbh_values_evolved, phi_present, label="$t=t_0$", marker="x")
        ax.plot(m_pbh_values_formation_to_evolve, phi_test, label="$t=0$ (test)")
        ax.plot(m_extracted_evolved, phi_extracted_evolved, linestyle="None", marker="+", label="$t=0$ (extracted)")

        ax.set_xlabel("$M~[\mathrm{g}]$")
        ax.set_ylabel("$\phi(M)~[\mathrm{g}]^{-1}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(title="$\sigma={:.1f}$".format(sigma), fontsize="small")
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


#%% Plot the mass density distribution, for an initial distribution following a log-normal in the number density
# This cell includes a test of the method psi_evolved_LN_number_density()
# Includes comparisons of results obtained using different numbers of evaluations, and compares logarithmic and linear interpolation of the results

if "__main__" == __name__:
    
    m_pbh_values_formation = np.logspace(11, 17, 100)
    m_pbh_values_formation_to_evolve = np.concatenate((np.arange(7.473420349255e+14, 7.4734203494e+14, 5e2), np.logspace(np.log10(7.47e14), 17, 100)))
    m_pbh_values_evolved = m_pbh_evolved_MP22(m_pbh_values_formation_to_evolve, t_0)
    m_c = 1e15
    
    for sigma in [0.1, 0.5, 1, 1.5]:
        psi_initial = psi_LN_number_density(m_pbh_values_formation, m_c, sigma)
        log_m_pbh_v2_100, log_psi_present_v2_100 = psi_evolved_LN_number_density(m_c, sigma, t_0, n_steps=100)
        log_m_pbh_v2_1000, log_psi_present_v2_1000 = psi_evolved_LN_number_density(m_c, sigma, t_0, n_steps=1000)
        log_m_pbh_v2_10000, log_psi_present_v2_10000 = psi_evolved_LN_number_density(m_c, sigma, t_0, n_steps=10000)
        m_pbh_v2_10000, psi_present_v2_10000 = psi_evolved_LN_number_density(m_c, sigma, t_0, n_steps=10000, log_output=False)
        
        phi_initial_to_evolve = phi_LN(m_pbh_values_formation_to_evolve, m_c, sigma)
        phi_present = phi_evolved_v2(phi_initial_to_evolve, m_pbh_values_evolved, m_pbh_values_formation_to_evolve, t_0)

        # Plot initial mass distribution psi and compare to the evolved mass distributions estimated using linear and logarithmic interpolation
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(m_pbh_values_formation, psi_initial, label="$t=0$")
        ax.plot(m_pbh_values_formation, 10**np.interp(np.log10(m_pbh_values_formation), log_m_pbh_v2_10000, log_psi_present_v2_10000, left=-100, right=-100), label="$t=t_0$ (method v2, 10000 evaluations) \n (log interpolation)" , marker="+")
        ax.plot(m_pbh_values_formation, np.interp(m_pbh_values_formation, m_pbh_v2_10000, psi_present_v2_10000, left=0, right=0), linestyle="None", label="$t=t_0$ (method v2, 10000 evaluations) \n (linear interpolation)" , marker="x")
        ax.plot(m_pbh_values_evolved, phi_present*m_pbh_values_evolved / np.trapz(phi_present*m_pbh_values_evolved, m_pbh_values_evolved), marker="+", linestyle="None", label=r"$\propto M\phi(M)$ " + "\n (normalised to 1) ($t=t_0$)")

        ax.set_xlabel("$M~[\mathrm{g}]$")
        ax.set_ylabel("$\psi(M)~[\mathrm{g}]^{-1}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(title="$\sigma={:.1f}$".format(sigma), fontsize="small")
        ax.set_xlim(1e11, max(m_pbh_values_formation))
        fig.tight_layout()
        
        # Plot the evolved mass distribution psi when evaluated at a different number of masses
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(m_pbh_values_formation, 10**np.interp(np.log10(m_pbh_values_formation), log_m_pbh_v2_100, log_psi_present_v2_100, left=-100, right=-100), label="100", marker="x", markersize=5)
        ax.plot(m_pbh_values_formation, 10**np.interp(np.log10(m_pbh_values_formation), log_m_pbh_v2_1000, log_psi_present_v2_1000, left=-100, right=-100), linestyle="None", label="1000 ", marker="x", markersize=3)
        ax.plot(m_pbh_values_formation, 10**np.interp(np.log10(m_pbh_values_formation), log_m_pbh_v2_10000, log_psi_present_v2_10000, left=-100, right=-100), linestyle="None", label="10000" , marker="+")
        ax.set_xlabel("$M~[\mathrm{g}]$")
        ax.set_ylabel("$\psi(M)~[\mathrm{g}]^{-1}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(title="Number of evaluations of $\psi$", fontsize="small")
        ax.set_title("Logarithmic interpolation of $\psi$, method v2, $\sigma={:.1f}$".format(sigma), fontsize="small")
        ax.set_xlim(1e11, max(m_pbh_values_formation))
        fig.tight_layout()    
        
        # Plot the evolved mass distribution psi when evaluated at a different number of masses
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(m_pbh_values_formation, 10**np.interp(np.log10(m_pbh_values_formation), log_m_pbh_v2_100, log_psi_present_v2_100, left=-100, right=-100) / 10**np.interp(np.log10(m_pbh_values_formation), log_m_pbh_v2_10000, log_psi_present_v2_10000, left=-100, right=-100), linestyle="None", label="100", marker="x", markersize=5)
        ax.plot(m_pbh_values_formation, 10**np.interp(np.log10(m_pbh_values_formation), log_m_pbh_v2_1000, log_psi_present_v2_1000, left=-100, right=-100) / 10**np.interp(np.log10(m_pbh_values_formation), log_m_pbh_v2_10000, log_psi_present_v2_10000, left=-100, right=-100), linestyle="None", label="1000 ", marker="x", markersize=3)
        ax.set_xlabel("$M~[\mathrm{g}]$")
        ax.set_ylabel("$\psi(N_\mathrm{eval}) / \psi(N_\mathrm{eval}=10000)$")
        ax.set_xscale("log")
        ax.legend(title="Number of evaluations \n of $\psi$, $N_\mathrm{eval}$", fontsize="small")
        ax.set_title("Logarithmic interpolation of $\psi$, method v2, $\sigma={:.1f}$".format(sigma), fontsize="small")
        ax.set_xlim(1e11, max(m_pbh_values_formation))
        fig.tight_layout()


#%% Plot the mass density distribution, for an initial distribution following a log-normal in the number density
# This cell includes a test of the normalisation of psi(M), by comparing the initial and evolved psi(M) in a mass 
# range where the effects of evaporation are negligible (M>~1e15g).

if "__main__" == __name__:
    
    m_pbh_values_formation = np.logspace(11, 17, 100)
    
    for sigma in [0.1]:
        
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        
        for m_c in np.logspace(15, 11, 5):
        
            psi_initial = psi_LN_number_density(m_pbh_values_formation, m_c, sigma)
            
            if sigma < 0.5:
                log_m_pbh_v2_10000, log_psi_present_v2_10000 = psi_evolved_LN_number_density(m_c, sigma, log_m_factor=50, t=t_0, n_steps=50000)
            else:
                log_m_pbh_v2_10000, log_psi_present_v2_10000 = psi_evolved_LN_number_density(m_c, sigma, log_m_factor=8, t=t_0, n_steps=50000)
                
            # Plot the evolved mass distribution psi when evaluated at a different number of masses
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(m_pbh_values_formation, psi_initial, label="$t=0$")
            ax.plot(m_pbh_values_formation, 10**np.interp(np.log10(m_pbh_values_formation), log_m_pbh_v2_10000, log_psi_present_v2_10000, left=-100, right=-100), label="$t=t_0$")
            ax.set_xlabel("$M~[\mathrm{g}]$")
            ax.set_ylabel("$\psi(M)~[\mathrm{g}]^{-1}$")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend(fontsize="small")
            ax.set_title("$M_c={:.1e}".format(m_c) + "~\mathrm{g},~" + "\sigma={:.1f}$".format(sigma), fontsize="small")
            ax.set_xlim(1e11, max(m_pbh_values_formation))
            fig.tight_layout()
            
            # Plot the ratio of evolved to initial MF values
            ax1.plot(m_pbh_values_formation, 10**np.interp(np.log10(m_pbh_values_formation), log_m_pbh_v2_10000, log_psi_present_v2_10000, left=-100, right=-100) / psi_initial, label="{:.1e}".format(m_c), marker="x", markersize=5)

        # Plot the evolved mass distribution psi when evaluated at a different number of masses
        ax1.set_xlabel("$M~[\mathrm{g}]$")
        ax1.set_ylabel("$\psi(t=t_0)/ \psi(t=0)$")
        ax1.set_xscale("log")
        ax1.legend(title="$M_c~[\mathrm{g}]$", fontsize="small")
        ax1.set_title("$\sigma={:.1f}$".format(sigma), fontsize="small")
        ax1.set_xlim(1e15, max(m_pbh_values_formation))
        ax1.set_ylim(0, 1.5)
        fig1.tight_layout()


#%% Test how much the normalisation factor used to convert between the number density and the mass density (n_0 / rho_0, equal to the mean of phi(M)), for a log-normal initial mass function, depends on the range of masses and number of masses:

if "__main__" == __name__:
    
    m_c = 1e12
    sigma = 0.1
    t = t_0
    
    n_steps_values = np.logspace(4, 1, 4)
    log_m_factors = np.linspace(10, 1, 10)
    
    for m_c in np.logspace(11, 15, 5):
        
        for sigma in [0.1, 0.5, 1]:
            fig, ax = plt.subplots(figsize=(5, 5))

            for n_steps in n_steps_values:    
                
                K_values_per_step = []
                
                for log_m_factor in log_m_factors:
                    
                    n_steps = int(n_steps)
                    
                    # Distribution function for PBH energy density, when the number density follows distribution phi_evolved, obtained
                    # from an initially log-normal distribution
                
                    log_m_min = np.log10(m_c) - log_m_factor*sigma
                    log_m_max = np.log10(m_c) + log_m_factor*sigma
                    
                    # To accurately evaluate the evolved mass function at small PBH masses, need to include a dense sampling of initial masses close to M_* = 7.5e14g
                    M0_test_values = np.sort(np.concatenate((np.arange(7.473420349255e+14, 7.4734203494e+14, 5e2),  np.logspace(log_m_min, log_m_max, n_steps))))
                    M_test_values = m_pbh_evolved_MP22(M0_test_values, t)
                    
                    phi_formation = LN(M0_test_values, m_c, sigma)
                    
                    # Estimate the normalisation of psi (such that the evolved MF is normalised to 1)
                    psi_normalisation = 1 / np.trapz(phi_formation*M0_test_values, M0_test_values)
                    
                    if log_m_factor == max(log_m_factors) and n_steps == max(n_steps_values):
                        psi_normalisation_most_acc = psi_normalisation
                        K_values_per_step.append(0)
        
                    else:
                        K_values_per_step.append(psi_normalisation / psi_normalisation_most_acc - 1)
                    
                ax.plot(log_m_factors, K_values_per_step, marker="x", linestyle="None", label="{:.0f}".format(np.log10(n_steps)))
                    
            ax.set_title("$M_c={:.0e}$".format(m_c) + "g , $\sigma={:.1f}$".format(sigma))
            ax.legend(title="$\log_{10}(N_\mathrm{steps})$", fontsize="small")
            ax.set_ylabel("$K / K_\mathrm{most~acc.} - 1$")
            ax.set_xlabel("log_m_factor")
            ax.set_ylim(-1, 1)
            fig.tight_layout()
            if sigma < 1:
                fig.savefig("./MP22_reproduction_pictures/K_comparison_Mc=1e{:.0f}g".format(np.log10(m_c)) + "_sigma=0p{:.0f}.pdf".format(10*sigma))
                fig.savefig("./MP22_reproduction_pictures/K_comparison_Mc=1e{:.0f}g".format(np.log10(m_c)) + "_sigma=0p{:.0f}.png".format(10*sigma))
            else:
                fig.savefig("./MP22_reproduction_pictures/K_comparison_Mc=1e{:.0f}g".format(np.log10(m_c)) + "_sigma={:.0f}.pdf".format(sigma))
                fig.savefig("./MP22_reproduction_pictures/K_comparison_Mc=1e{:.0f}g".format(np.log10(m_c)) + "_sigma={:.0f}.png".format(sigma))
            plt.close()
            

#%% Reproduce Fig. 4 of Mosbech & Picker (2022) 

if "__main__" == __name__:
    b_max, l_max = np.radians(3.5), np.radians(3.5)
    delta_Omega = Delta(-l_max, l_max, -b_max, b_max)

    # Load data from HESS (Abramowski et al. 2016, 1603.07730)
    E_lower_y_HESS, flux_lower_y_HESS = load_data("1603.07730/1603.07730_lower_y.csv")
    E_upper_y_HESS, flux_upper_y_HESS = load_data("1603.07730/1603.07730_upper_y.csv")
    E_lower_HESS, flux_mid_HESS = load_data("1603.07730/1603.07730_x_bins.csv")
    
    # Divide HESS data by a factor of 10 to account for the shift in Fig. 3 of Abramowski et al. (2016)
    flux_mid_HESS /= 10
    flux_lower_y_HESS /= 10
    flux_upper_y_HESS /= 10
    
    # Widths of energy bins
    E_minus_HESS = E_upper_y_HESS - E_lower_HESS[:-1]
    E_plus_HESS = E_lower_HESS[1:] - E_upper_y_HESS
    
    # Upper and lower error bars on flux values
    flux_plus_HESS = flux_upper_y_HESS - flux_mid_HESS[:-1]
    flux_minus_HESS = flux_mid_HESS[:-1] - flux_lower_y_HESS
    
    # Load data from FermiLAT (Lacroix et al. 2016, 1512.01846)
    E_lower_y_FermiLAT, flux_lower_y_FermiLAT_sys = load_data("1512.01846/1512.01846_lower_y_sys.csv")
    E_lower_y_FermiLAT, flux_lower_y_FermiLAT_stat = load_data("1512.01846/1512.01846_lower_y_stat.csv")
    E_upper_y_FermiLAT, flux_upper_y_FermiLAT_sys = load_data("1512.01846/1512.01846_upper_y_sys.csv")
    E_upper_y_FermiLAT, flux_upper_y_FermiLAT_stat = load_data("1512.01846/1512.01846_upper_y_stat.csv")
    E_lower_FermiLAT, flux_mid_FermiLAT = load_data("1512.01846/1512.01846_x_bins.csv")
    
    # Widths of energy bins
    E_minus_FermiLAT = E_upper_y_FermiLAT - E_lower_FermiLAT[:-1]
    E_plus_FermiLAT = E_lower_FermiLAT[1:] - E_upper_y_FermiLAT
    
    # Upper and lower error bars on flux values
    flux_plus_FermiLAT_stat = flux_upper_y_FermiLAT_stat - flux_mid_FermiLAT[:-1]
    flux_minus_FermiLAT_stat = flux_mid_FermiLAT[:-1] - flux_lower_y_FermiLAT_stat
    
    flux_plus_FermiLAT_sys = flux_upper_y_FermiLAT_sys - flux_mid_FermiLAT[:-1]
    flux_minus_FermiLAT_sys = flux_mid_FermiLAT[:-1] - flux_lower_y_FermiLAT_stat
    
    # Check the data is plotted in the correct position, matching Fig. 3 of Mosbech & Picker (2022)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.errorbar(E_lower_y_HESS, flux_mid_HESS[:-1], yerr=(flux_minus_HESS, flux_plus_HESS), xerr=(E_minus_HESS, E_plus_HESS), linestyle="None", label="HESS")
    ax.errorbar(E_lower_y_FermiLAT, flux_mid_FermiLAT[:-1], yerr=(flux_minus_FermiLAT_stat, flux_plus_FermiLAT_stat), xerr=(E_minus_FermiLAT, E_plus_FermiLAT), marker="x", linestyle="None", label="Fermi-LAT")
    ax.errorbar(E_lower_y_FermiLAT, flux_mid_FermiLAT[:-1], yerr=(flux_minus_FermiLAT_sys, flux_plus_FermiLAT_sys), xerr=(E_minus_FermiLAT, E_plus_FermiLAT), linestyle="None")
    ax.set_xlabel("Particle energy $E$ [GeV]")
    ax.set_ylabel("$\gamma$ flux: $E^2 \mathrm{d}^2 N / \mathrm{d}E\mathrm{d}t~[\mathrm{GeV}~\mathrm{s}^{-1}~\mathrm{cm}^{-2}]$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-6, 2e5)
    ax.set_ylim(1e-19, 1e-3)
    ax.legend()
    fig.tight_layout()
    
    flux_plus_FermiLAT = flux_plus_FermiLAT_stat + flux_plus_FermiLAT_sys
    flux_minus_FermiLAT = flux_minus_FermiLAT_stat + flux_minus_FermiLAT_sys
    
    # Print fluxes per steradian
    print(flux_mid_FermiLAT[:-1] / (delta_Omega * E_lower_y_FermiLAT**2))
    print(flux_minus_FermiLAT / (delta_Omega * E_lower_y_FermiLAT**2))
    print(flux_plus_FermiLAT / (delta_Omega * E_lower_y_FermiLAT**2))

    print("\nValues used in Isatis and isatis_reproduction.py")
    file_path_data = "./../../Downloads/version_finale/scripts/Isatis/constraints/photons/"    
    append = "Fermi-LAT_1512.01846"
    energies, energies_minus, energies_plus, flux, flux_minus, flux_plus = np.genfromtxt("%sflux_%s.txt"%(file_path_data, append), skip_header = 6).transpose()[0:6]
    print(flux)
    print(flux_minus)
    print(flux_plus)
    
    print("\nOlder values:")
    energies_old, energies_minus_old, energies_plus_old, flux_old, flux_minus_old, flux_plus_old = np.genfromtxt("%sflux_%s_old.txt"%(file_path_data, append), skip_header = 6).transpose()[0:6]
    print(flux_old)
    print(flux_minus_old)
    print(flux_plus_old)

    print("\nMiddle fluxes (corrected / old)")
    print(flux / flux_old)
    

#%% Plot the Fermi-LAT constraints (monochromatic MF)

if "__main__" == __name__:
    
    delta_Omega = Delta(-l_max, l_max, -b_max, b_max)
    
    m_pbh_mono = np.logspace(10, 18, 100)
        
    # Constraints data at each PBH mass, calculated using Isatis
    constraints_names_lower, constraints_Isatis_file_lower = load_results_Isatis(mf_string="results_MP22_lower_v2")
    constraints_names_upper, constraints_Isatis_file_upper = load_results_Isatis(mf_string="results_MP22_upper_v2")
    
    f_PBH_Isatis_lower = np.array(constraints_Isatis_file_lower[-1])
    f_PBH_Isatis_upper = np.array(constraints_Isatis_file_upper[-1])
   
    # Constraints data for each energy bin of each instrument, calculated using isatis_reproduction.py   
    constraints_Isatis_reproduction_file_lower = np.transpose(np.genfromtxt("./Data/fPBH_GC_full_all_bins_Fermi-LAT_1512.01846_lower_monochromatic_wide.txt"))
    constraints_Isatis_reproduction_file_upper = np.transpose(np.genfromtxt("./Data/fPBH_GC_full_all_bins_Fermi-LAT_1512.01846_upper_monochromatic_wide.txt"))
     
    f_PBH_Isatis_reproduction_lower = envelope(constraints_Isatis_reproduction_file_lower)
    f_PBH_Isatis_reproduction_upper = envelope(constraints_Isatis_reproduction_file_upper)
       
    # Plot the delta function MF constraint
    fig, ax = plt.subplots(figsize=(6,6))
    ax.fill_between(m_pbh_mono, f_PBH_Isatis_lower, f_PBH_Isatis_upper)
    ax.fill_between(m_pbh_mono, f_PBH_Isatis_reproduction_lower, f_PBH_Isatis_reproduction_upper, alpha=0.5)
    ax.plot(m_pbh_mono, f_PBH_Isatis_lower, marker="x", linestyle="None")
    ax.set_xlim(1e10, 1e18)
    ax.set_ylim(10**(-10), 1)
    ax.set_xlabel("$M_\mathrm{PBH}~[\mathrm{g}]$")
    ax.set_ylabel("$f_\mathrm{PBH}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.tight_layout()
    

#%% Plot constraints for extended MF (reproducing Fig. 4 of Mosbech & Picker (2022)).

if "__main__" == __name__:
    # Constraints data for each energy bin of each instrument (extended MF)
    
    constraints_mono_file_lower = np.transpose(np.genfromtxt("./Data/fPBH_GC_full_all_bins_Fermi-LAT_1512.01846_lower_monochromatic_wide.txt"))
    constraints_mono_file_upper = np.transpose(np.genfromtxt("./Data/fPBH_GC_full_all_bins_Fermi-LAT_1512.01846_upper_monochromatic_wide.txt"))
        
    M_values_eval = np.logspace(10, 18, 100)   # masses at which the constraint is evaluated for a delta-function MF
    mc_values_unevolved = np.logspace(14, 17, 50)
    mc_values_evolved = np.logspace(13, 17, 50)

    # Final constraint
    constraint_lower = []
    constraint_upper = []
    
    # Constraint from each energy bin
    energy_bin_constraints_lower = []
    energy_bin_constraints_upper = []
    
    sigma = 0.1
    
    params_LN = [sigma]
    
    # Unevolved mass function
    for k in range(len(constraints_mono_file_lower)):

        # Constraint from a particular energy bin
        constraint_energy_bin = constraints_mono_file_lower[k]

        # Calculate constraint on f_PBH from each bin
        f_PBH_k = constraint_Carr(mc_values_unevolved, m_mono=M_values_eval, f_max=constraint_energy_bin, mf=psi_LN_number_density, params=params_LN)
        energy_bin_constraints_lower.append(f_PBH_k)

    for k in range(len(constraints_mono_file_upper)):

        # Constraint from a particular energy bin
        constraint_energy_bin = constraints_mono_file_upper[k]

        # Calculate constraint on f_PBH from each bin
        f_PBH_k = constraint_Carr(mc_values_unevolved, m_mono=M_values_eval, f_max=constraint_energy_bin, mf=psi_LN_number_density, params=params_LN)
        energy_bin_constraints_upper.append(f_PBH_k)
        
    constraint_lower.append(envelope(energy_bin_constraints_lower))
    constraint_upper.append(envelope(energy_bin_constraints_upper))
    

    # Evolved mass function
    constraint_lower_evolved = []
    constraint_upper_evolved = []
        
    for m_c in mc_values_evolved:
                
        # Evolved mass function
        log_m_evolved, log_psi_evolved = psi_evolved_LN_number_density(m_c, sigma, t_0, log_m_factor=3, n_steps=1000)   # evolved PBH distribution, evaluated at present masses corresponding to the formation masses in M0_values
        # Interpolate evolved mass function at the evolved masses at which the delta-function MF constraint is calculated
        mf_evolved_interp = 10**np.interp(np.log10(M_values_eval), log_m_evolved, log_psi_evolved)
        
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
        
    
    # Load data from Fig. 4 of Mosbech & Picker (2022)
    m_LN_lower, f_LN_lower = load_data("2203.05743/MP22_sigma_{:.1f}_LN_lower.csv".format(sigma))
    m_LN_upper, f_LN_upper = load_data("2203.05743/MP22_sigma_{:.1f}_LN_upper.csv".format(sigma))
    m_evolved_lower, f_evolved_lower = load_data("2203.05743/MP22_sigma_{:.1f}_evolved_lower.csv".format(sigma))
    m_evolved_upper, f_evolved_upper = load_data("2203.05743/MP22_sigma_{:.1f}_evolved_upper.csv".format(sigma))
   
    fig, ax = plt.subplots(figsize=(6.5,6.5))
    ax.fill_between(mc_values_unevolved, constraint_lower[0], constraint_upper[0], color="tab:green", alpha=0.5, label="Log-normal \n (unevolved)")
    ax.plot(m_LN_lower, f_LN_lower, color="tab:green", linestyle="dotted")   
    ax.plot(m_LN_upper, f_LN_upper, color="tab:green", linestyle="dotted")

    ax.fill_between(mc_values_evolved, constraint_lower_evolved, constraint_upper_evolved, color="tab:purple", alpha=0.5, label="Evolved")
    ax.plot(m_evolved_lower, f_evolved_lower, color="tab:purple", linestyle="dotted")   
    ax.plot(m_evolved_upper, f_evolved_upper, color="tab:purple", linestyle="dotted")
    ax.plot(0,0, linestyle="dotted", color="k", label="Extracted from \n Mosbech \& \n Picker (2022)")
    ax.plot(0,0, color="k", label="Reproduced")
    
    ax.set_xlim(1e10, 1e18)
    ax.set_ylim(10**(-12), 1)
    ax.set_xlabel("$M_c~[\mathrm{g}]$")
    ax.set_ylabel("$f_\mathrm{PBH}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("$\sigma={:.1f}$".format(sigma))
    ax.legend(fontsize="small")
    plt.tight_layout()
 
    print(f_evolved_lower)
    print(f_evolved_upper)
    
#%% Additional methods, for further checks and calculations

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


#%% Compare the 'direct' calculations of the evolved and unevolved MFs with older implementations of the methods
if "__main__" == __name__:
    
    fig, ax = plt.subplots(figsize=(6.5,6.5))
    m_pbh_values_formation = np.concatenate((np.arange(7.473420349255e+14, 7.4734203494e+14, 5e2), np.arange(7.4734203494e+14, 7.47344e+14, 1e7), np.logspace(np.log10(7.474e14), 17, 500)))
    m_pbh_values_evolved = m_pbh_evolved_MP22(m_pbh_values_formation, t_0)
    m_c = 1e15
    
    for sigma in [0.1, 0.5, 1, 1.5]:
        
        log_m_indirect_values, log_psi_evolved_indirect_values = psi_evolved_LN_number_density(m_c, sigma, t_0)
        m_pbh_indirect_values, psi_evolved_indirect_values = 10**log_m_indirect_values, 10**log_psi_evolved_indirect_values
        
        psi_initial = psi_LN_number_density(m_pbh_values_formation, m_c, sigma)       
        psi_evolved_direct_values = psi_evolved_direct(psi_initial, m_pbh_values_evolved, m_pbh_values_formation)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(m_pbh_indirect_values, psi_evolved_indirect_values, label="Indirect")
        ax.plot(m_pbh_values_evolved, psi_evolved_direct_values, label="Direct", linestyle="None", marker="x")
        ax.set_xlabel("$M~[\mathrm{g}]$")
        ax.set_ylabel("$\psi(M, t_0)~[\mathrm{g}]^{-1}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(title="$\sigma={:.1f}$".format(sigma), fontsize="small")
        ax.set_xlim(1e11, max(m_pbh_values_formation))
        fig.tight_layout()

        fig, ax = plt.subplots(figsize=(6, 6))
        frac_diff = 10**np.interp(np.log10(m_pbh_values_evolved), log_m_indirect_values, log_psi_evolved_indirect_values)/psi_evolved_direct_values - 1
        ax.plot(m_pbh_values_evolved, abs(frac_diff), marker="x", linestyle="None")
        ax.set_xlabel("$M~[\mathrm{g}]$")
        ax.set_ylabel("$\psi(M, t)$ $|$(indirect / direct - 1)$|$")
        ax.set_xscale("log")
        ax.set_title("$\sigma={:.1f}$".format(sigma))
        # Choose x-axis range to ensure only the range in which both mass functions are defined is included (to avoid complications due to interpolation outside the range defined).
        ax.set_xlim(max([min(m_pbh_values_evolved), min(m_pbh_indirect_values)]), min([max(m_pbh_values_evolved), max(m_pbh_indirect_values)]))
        ax.set_ylim(1e-6, 1)
        ax.set_yscale("log")
        fig.tight_layout()



#%% Plot the mass density distribution, for an initial distribution following a log-normal in the number density
# Compare the direct calculation in psi_evolved_direct() to the indirect calculation using psi_evolved_LN_number_density()

if "__main__" == __name__:
        
    log_m_factor = 3
    n_steps = 1000
    m_c = 1e17
    
    normalised_unity = False
        
    for sigma in [0.1, 0.5, 1, 1.5]:
        
        log_m_min = np.log10(m_c) - log_m_factor*sigma
        log_m_max = np.log10(m_c) + log_m_factor*sigma
        m_pbh_values_formation = np.sort(np.concatenate((np.arange(7.473420349255e+14, 7.4734203494e+14, 5e2), np.arange(7.4734203494e+14, 7.47344e+14, 1e7), np.logspace(log_m_min, log_m_max, n_steps))))
        m_pbh_values_evolved = m_pbh_evolved_MP22(m_pbh_values_formation, t_0)
        psi_initial = psi_LN_number_density(m_pbh_values_formation, m_c, sigma)
        
        log_m_pbh_v2, log_psi_present_v2 = psi_evolved_LN_number_density(m_c, sigma, t_0)
        psi_present_direct = psi_evolved_direct(psi_initial, m_pbh_values_evolved, m_pbh_values_formation, t_0)
        phi_initial = phi_LN(m_pbh_values_formation, m_c, sigma)
        phi_present = phi_evolved_v2(phi_initial, m_pbh_values_evolved, m_pbh_values_formation, t_0)

        # Plot initial mass distribution psi and compare to the evolved mass distribution psi calculated directly and through calculating phi first.
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(m_pbh_values_formation, psi_initial, label="$t=0$")
        ax.plot(m_pbh_values_evolved, 10**np.interp(np.log10(m_pbh_values_evolved), log_m_pbh_v2, log_psi_present_v2, left=-100, right=-100), label="Calculate $\phi$ first" , linestyle="None", marker="+")
        ax.plot(m_pbh_values_evolved, psi_present_direct, linestyle="None", label="Calculate $\psi$ directly" , marker="x")
        ax.set_xlabel("$M~[\mathrm{g}]$")
        ax.set_ylabel("$\psi(M)~[\mathrm{g}]^{-1}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(title="$\sigma={:.1f}$".format(sigma), fontsize="small")
        ax.set_xlim(1e11, max(m_pbh_values_formation))
        fig.tight_layout()
        
        # Plot fractional difference between the evolved mass distribution psi calculated directly and through calculating phi first.
        frac_diff = 10**np.interp(np.log10(m_pbh_values_evolved), log_m_pbh_v2, log_psi_present_v2, left=-100, right=-100) / psi_present_direct - 1
        print(frac_diff)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(m_pbh_values_evolved, np.abs(frac_diff), linestyle="None", marker="x")
        ax.set_xlabel("$M~[\mathrm{g}]$")
        ax.set_ylabel("$| \psi(M, t)$ (through $\phi$ first / direct - 1) $|$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(title="$\sigma={:.1f}$".format(sigma), fontsize="small")
        ax.set_xlim(1e11, max(m_pbh_values_formation))
        ax.set_ylim(1e-10, 0.5)
        fig.tight_layout()
        
        
#%% Plot constraints for extended MF (reproducing Fig. 4 of Mosbech & Picker (2022)), using direct calculation of psi

if "__main__" == __name__:
    # Constraints data for each energy bin of each instrument (extended MF)
    
    constraints_mono_file_lower = np.transpose(np.genfromtxt("./Data/fPBH_GC_full_all_bins_Fermi-LAT_1512.01846_lower_monochromatic_wide.txt"))
    constraints_mono_file_upper = np.transpose(np.genfromtxt("./Data/fPBH_GC_full_all_bins_Fermi-LAT_1512.01846_upper_monochromatic_wide.txt"))
        
    M_values_eval = np.logspace(10, 18, 100)   # masses at which the constraint is evaluated for a delta-function MF
    mc_values_unevolved = np.logspace(14, 17, 50)
    mc_values_evolved = np.logspace(13, 17, 50)[5:]
    
    M0_values_input = np.concatenate((np.arange(7.473420349255e+14, 7.4734203494e+14, 5e2), np.arange(7.4734203494e+14, 7.47344e+14, 1e7), np.logspace(np.log10(7.474e14), 18, 500)))
    M_values_input = m_pbh_evolved_MP22(M0_values_input, t_0)

    # Final constraint
    constraint_lower = []
    constraint_upper = []
    
    # Constraint from each energy bin
    f_PBH_energy_bin_lower = []
    f_PBH_energy_bin_upper = []
    
    normalised_unity = False    
    sigma = 0.1
    
    params_LN = [sigma]
    
    for m_c in mc_values_unevolved:
        
        psi_unevolved = psi_LN_number_density(M_values_eval, m_c, sigma)
                                
        # Constraint from each energy bin
        f_PBH_energy_bin_lower = []
        for k in range(len(constraints_mono_file_lower)):
    
            # Constraint from a particular energy bin (delta function MF)
            constraint_energy_bin = constraints_mono_file_lower[k]
            
            integrand = psi_unevolved / constraint_energy_bin
            integral = np.trapz(np.nan_to_num(integrand), M_values_eval)

            if integral == 0 or np.isnan(integral):
                f_PBH_energy_bin_lower.append(10)
            else:
                f_PBH_energy_bin_lower.append(1/integral)

        constraint_lower.append(min(f_PBH_energy_bin_lower))
        
        # Constraint from each energy bin        
        f_PBH_energy_bin_upper = []
        for k in range(len(constraints_mono_file_upper)):
    
            # Constraint from a particular energy bin (delta function MF)
            constraint_energy_bin = constraints_mono_file_upper[k]
            
            integrand = psi_unevolved / constraint_energy_bin
            integral = np.trapz(np.nan_to_num(integrand), M_values_eval)
            
            if integral == 0 or np.isnan(integral):
                f_PBH_energy_bin_upper.append(10)
            else:
                f_PBH_energy_bin_upper.append(1/integral)
        constraint_upper.append(min(f_PBH_energy_bin_upper))
        

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
        
    
    # Load data from Fig. 4 of Mosbech & Picker (2022)
    m_LN_lower, f_LN_lower = load_data("2203.05743/MP22_sigma_{:.1f}_LN_lower.csv".format(sigma))
    m_LN_upper, f_LN_upper = load_data("2203.05743/MP22_sigma_{:.1f}_LN_upper.csv".format(sigma))
    m_evolved_lower, f_evolved_lower = load_data("2203.05743/MP22_sigma_{:.1f}_evolved_lower.csv".format(sigma))
    m_evolved_upper, f_evolved_upper = load_data("2203.05743/MP22_sigma_{:.1f}_evolved_upper.csv".format(sigma))
   
    fig, ax = plt.subplots(figsize=(6.5,6.5))
    ax.fill_between(mc_values_unevolved, constraint_lower, constraint_upper, color="tab:green", alpha=0.5, label="Log-normal \n (unevolved)")
    ax.plot(m_LN_lower, f_LN_lower, color="tab:green", linestyle="dotted")   
    ax.plot(m_LN_upper, f_LN_upper, color="tab:green", linestyle="dotted")

    ax.fill_between(mc_values_evolved, constraint_lower_evolved, constraint_upper_evolved, color="tab:purple", alpha=0.5, label="Evolved")
    ax.plot(m_evolved_lower, f_evolved_lower, color="tab:purple", linestyle="dotted")   
    ax.plot(m_evolved_upper, f_evolved_upper, color="tab:purple", linestyle="dotted")
    ax.plot(0,0, linestyle="dotted", color="k", label="Extracted from \n Mosbech \& \n Picker (2022)")
    ax.plot(0,0, color="k", label="Reproduced")
    
    ax.set_xlim(1e10, 1e18)
    ax.set_ylim(10**(-12), 1)
    ax.set_xlabel("$M_c~[\mathrm{g}]$")
    ax.set_ylabel("$f_\mathrm{PBH}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("$\sigma={:.1f}$".format(sigma))
    ax.legend(fontsize="small")
    plt.tight_layout()
    
    
#%% Plot constraints for extended MF (reproducing Fig. 4 of Mosbech & Picker (2022)), using direct calculation of psi,
# comparing results obtained from the evolved unnormalised mass function with the evolved normalised one.

if "__main__" == __name__:
    # Constraints data for each energy bin of each instrument (extended MF)
    
    constraints_mono_file_lower = np.transpose(np.genfromtxt("./Data/fPBH_GC_full_all_bins_Fermi-LAT_1512.01846_lower_monochromatic_wide.txt"))
    constraints_mono_file_upper = np.transpose(np.genfromtxt("./Data/fPBH_GC_full_all_bins_Fermi-LAT_1512.01846_upper_monochromatic_wide.txt"))
        
    M_values_eval = np.logspace(10, 18, 100)   # masses at which the constraint is evaluated for a delta-function MF
    mc_values_unevolved = np.logspace(14, 17, 50)
   
    M0_values_input = np.concatenate((np.arange(7.473420349255e+14, 7.4734203494e+14, 5e2), np.arange(7.4734203494e+14, 7.47344e+14, 1e7), np.logspace(np.log10(7.474e14), 25, 1000)))
    M_values_input = m_pbh_evolved_MP22(M0_values_input, t_0)
    
    # Constraint from each energy bin
    f_PBH_energy_bin_lower = []
    f_PBH_energy_bin_upper = []
    
    sigmas = [1.5, 1.0, 0.5, 0.1]
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11,11))
    axes_plotting = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]
    
    fig1, axes1 = plt.subplots(nrows=2, ncols=2, figsize=(11,11))
    axes_plotting1 = [axes1[0][0], axes1[0][1], axes1[1][0], axes1[1][1]]

    
    for i in range(len(sigmas)):
        
        sigma = sigmas[i]
        params_LN = [sigma]        
        
        if sigma >= 1:
            mc_values_evolved = np.logspace(10, 18, 100)
        else:
            mc_values_evolved = np.logspace(13, 18, 50)[5:]
    
        # Evolved mass function
        constraint_lower_evolved = []
        constraint_upper_evolved = []
            
        for m_c in mc_values_evolved:
            
            psi_initial = psi_LN_number_density(M0_values_input, m_c, sigma)
            
            # Evolved mass function (normalised to unity at the formation time only)
            psi_evolved = psi_evolved_direct(psi_initial, M_values_input, M0_values_input)
            print("Integral of evolved MF over M = ", np.trapz(psi_evolved, M_values_input))
                
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
    
    
        # Evolved mass function (normalised to unity)
        constraint_lower_evolved_normalised_unity = []
        constraint_upper_evolved_normalised_unity = []
            
        for m_c in mc_values_evolved:
            
            psi_initial = psi_LN_number_density(M0_values_input, m_c, sigma)
            
            # Evolved mass function (normalised to unity at all times)
            psi_evolved_normalised_unity = psi_evolved_direct_normalised(psi_initial, M_values_input, M0_values_input)
            print("Integral of evolved normalised over M = ", np.trapz(psi_evolved_normalised_unity, M_values_input))
                
            # Interpolate evolved mass function at the evolved masses at which the delta-function MF constraint is calculated
            M_values_input_nozeros = M_values_input[psi_evolved_normalised_unity > 0]
            psi_evolved_normalised_unity_nozeros = psi_evolved_normalised_unity[psi_evolved_normalised_unity > 0]
            mf_evolved_normalised_unity_interp = 10**np.interp(np.log10(M_values_eval), np.log10(M_values_input_nozeros), np.log10(psi_evolved_normalised_unity_nozeros), left=-100, right=-100)
            
            # Constraint from each energy bin
            f_PBH_energy_bin_lower = []
            for k in range(len(constraints_mono_file_lower)):
        
                # Constraint from a particular energy bin (delta function MF)
                constraint_energy_bin = constraints_mono_file_lower[k]
                
                integrand = mf_evolved_normalised_unity_interp / constraint_energy_bin
                integral = np.trapz(np.nan_to_num(integrand), M_values_eval)
    
                if integral == 0 or np.isnan(integral):
                    f_PBH_energy_bin_lower.append(10)
                else:
                    f_PBH_energy_bin_lower.append(1/integral)
    
            constraint_lower_evolved_normalised_unity.append(min(f_PBH_energy_bin_lower))
            
            
            f_PBH_energy_bin_upper = []
            for k in range(len(constraints_mono_file_upper)):
        
                # Constraint from a particular energy bin (delta function MF)
                constraint_energy_bin = constraints_mono_file_upper[k]
                
                integrand = mf_evolved_normalised_unity_interp / constraint_energy_bin
                integral = np.trapz(np.nan_to_num(integrand), M_values_eval)
                
                if integral == 0 or np.isnan(integral):
                    f_PBH_energy_bin_upper.append(10)
                else:
                    f_PBH_energy_bin_upper.append(1/integral)
            constraint_upper_evolved_normalised_unity.append(min(f_PBH_energy_bin_upper))
    
        ax = axes_plotting[i]
        ax.fill_between(mc_values_evolved, constraint_lower_evolved_normalised_unity, constraint_upper_evolved_normalised_unity, color="tab:cyan", alpha=0.5, label="$\psi_\mathrm{N}$")
        ax.fill_between(mc_values_evolved, constraint_lower_evolved, constraint_upper_evolved, color="tab:purple", alpha=0.5, label="$\psi$")

        print(np.array(constraint_lower_evolved_normalised_unity) / np.array(constraint_lower_evolved) - 1)

        if sigma <= 1.5:
            # Load data from Fig. 4 of Mosbech & Picker (2022)
            m_evolved_lower, f_evolved_lower = load_data("2203.05743/MP22_sigma_{:.1f}_evolved_lower.csv".format(sigma))
            m_evolved_upper, f_evolved_upper = load_data("2203.05743/MP22_sigma_{:.1f}_evolved_upper.csv".format(sigma))
                    
            ax.plot(m_evolved_lower, f_evolved_lower, color="tab:purple", linestyle="dotted", label="MP (2022)")   
            ax.plot(m_evolved_upper, f_evolved_upper, color="tab:purple", linestyle="dotted")
        
        ax.set_xlim(1e10, 1e18)
        #ax.set_xlim(1e13, 1e16)
        ax.set_ylim(10**(-12), 1)
        ax.set_xlabel("$M_c~[\mathrm{g}]$")
        ax.set_ylabel("$f_\mathrm{PBH}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("$\sigma={:.1f}$".format(sigma))
        
        
        ax1 = axes_plotting1[i]
        ax1.plot(mc_values_evolved, abs(np.array(constraint_lower_evolved_normalised_unity) / np.array(constraint_lower_evolved) - 1), color="tab:blue", marker="x", linestyle="dotted")
        ax1.plot(mc_values_evolved, abs(np.array(constraint_upper_evolved_normalised_unity) / np.array(constraint_upper_evolved) - 1), color="tab:blue", marker="x", linestyle="dotted")
        ax1.set_xlabel("$M_c~[\mathrm{g}]$")
        ax1.set_ylabel("$|f_\mathrm{PBH}(\psi_N) / f_\mathrm{PBH}(\psi) - 1|$")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_ylim(1e-6, 2)
        ax1.set_xlim(1e14, 1e17)
        ax1.set_title("$\sigma={:.1f}$".format(sigma))
       
        if i == 0:
            ax.legend()
    plt.tight_layout()


#%% Plot constraints for unevolved MF (reproducing Fig. 4 of Mosbech & Picker (2022)),
# where one of the unevolved MFs follows a log-normal in psi and the other follows a log-normal in phi.

if "__main__" == __name__:
    # Constraints data for each energy bin of each instrument (extended MF)
    
    constraints_mono_file_lower = np.transpose(np.genfromtxt("./Data/fPBH_GC_full_all_bins_Fermi-LAT_1512.01846_lower_monochromatic_wide.txt"))
    constraints_mono_file_upper = np.transpose(np.genfromtxt("./Data/fPBH_GC_full_all_bins_Fermi-LAT_1512.01846_upper_monochromatic_wide.txt"))
        
    M_values_eval = np.logspace(10, 18, 100)   # masses at which the constraint is evaluated for a delta-function MF
    mc_values_unevolved = np.logspace(14, 17, 50)
       
    # Constraint from each energy bin
    f_PBH_energy_bin_lower = []
    f_PBH_energy_bin_upper = []
    
    sigmas = [1.5, 1.0, 0.5, 0.1]
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11,11))
    axes_plotting = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]
    
    fig1, axes1 = plt.subplots(nrows=2, ncols=2, figsize=(11,11))
    axes_plotting1 = [axes1[0][0], axes1[0][1], axes1[1][0], axes1[1][1]]
    
    fracdiff_ylims = [(5e2, 1e4), (1e1, 1e3), (5e-1, 5e3), (1e-4, 5)]

    
    for i in range(len(sigmas)):
        
        sigma = sigmas[i]
        params_LN = [sigma]        
        
        mc_values_unevolved = np.logspace(14, 18, 50)
    
        # unevolved mass function
        constraint_lower_unevolved = []
        constraint_upper_unevolved = []
            
        for m_c in mc_values_unevolved:
                        
            # unevolved mass function (log-normal in phi)
            psi_unevolved_LN_phi = psi_LN_number_density(M_values_eval, m_c, sigma)
                            
            # Constraint from each energy bin
            f_PBH_energy_bin_lower = []
            for k in range(len(constraints_mono_file_lower)):
        
                # Constraint from a particular energy bin (delta function MF)
                constraint_energy_bin = constraints_mono_file_lower[k]
                
                integrand = psi_unevolved_LN_phi / constraint_energy_bin
                integral = np.trapz(np.nan_to_num(integrand), M_values_eval)
    
                if integral == 0 or np.isnan(integral):
                    f_PBH_energy_bin_lower.append(10)
                else:
                    f_PBH_energy_bin_lower.append(1/integral)
    
            constraint_lower_unevolved.append(min(f_PBH_energy_bin_lower))
            
            
            f_PBH_energy_bin_upper = []
            for k in range(len(constraints_mono_file_upper)):
        
                # Constraint from a particular energy bin (delta function MF)
                constraint_energy_bin = constraints_mono_file_upper[k]
                
                integrand = psi_unevolved_LN_phi / constraint_energy_bin
                integral = np.trapz(np.nan_to_num(integrand), M_values_eval)
                
                if integral == 0 or np.isnan(integral):
                    f_PBH_energy_bin_upper.append(10)
                else:
                    f_PBH_energy_bin_upper.append(1/integral)
                    
            constraint_upper_unevolved.append(min(f_PBH_energy_bin_upper))
    
    
        # unevolved mass function (normalised to unity)
        constraint_lower_unevolved_LN_psi = []
        constraint_upper_unevolved_LN_psi = []
            
        for m_c in mc_values_unevolved:
                        
            # unevolved mass function (log-normal in psi)
            psi_unevolved_LN_psi = LN(M_values_eval, m_c, sigma)
                            
            # Constraint from each energy bin
            f_PBH_energy_bin_lower = []
            for k in range(len(constraints_mono_file_lower)):
        
                # Constraint from a particular energy bin (delta function MF)
                constraint_energy_bin = constraints_mono_file_lower[k]
                
                integrand = psi_unevolved_LN_psi / constraint_energy_bin
                integral = np.trapz(np.nan_to_num(integrand), M_values_eval)
    
                if integral == 0 or np.isnan(integral):
                    f_PBH_energy_bin_lower.append(10)
                else:
                    f_PBH_energy_bin_lower.append(1/integral)
    
            constraint_lower_unevolved_LN_psi.append(min(f_PBH_energy_bin_lower))
            
            
            f_PBH_energy_bin_upper = []
            for k in range(len(constraints_mono_file_upper)):
        
                # Constraint from a particular energy bin (delta function MF)
                constraint_energy_bin = constraints_mono_file_upper[k]
                
                integrand = psi_unevolved_LN_psi / constraint_energy_bin
                integral = np.trapz(np.nan_to_num(integrand), M_values_eval)
                
                if integral == 0 or np.isnan(integral):
                    f_PBH_energy_bin_upper.append(10)
                else:
                    f_PBH_energy_bin_upper.append(1/integral)
            constraint_upper_unevolved_LN_psi.append(min(f_PBH_energy_bin_upper))
    
        ax = axes_plotting[i]
        ax.fill_between(mc_values_unevolved, constraint_lower_unevolved_LN_psi, constraint_upper_unevolved_LN_psi, color="y", alpha=0.5, label="$\psi$ lognormal")
        ax.fill_between(mc_values_unevolved, constraint_lower_unevolved, constraint_upper_unevolved, color="tab:green", alpha=0.5, label="$\phi$ lognormal")

        print(np.array(constraint_lower_unevolved_LN_psi) / np.array(constraint_lower_unevolved))

        if sigma <= 1.5:
            # Load data from Fig. 4 of Mosbech & Picker (2022)
            m_LN_lower, f_LN_lower = load_data("2203.05743/MP22_sigma_{:.1f}_LN_lower.csv".format(sigma))
            m_LN_upper, f_LN_upper = load_data("2203.05743/MP22_sigma_{:.1f}_LN_upper.csv".format(sigma))      
              
            ax.plot(m_LN_lower, f_LN_lower, color="tab:green", linestyle="dotted", label="MP (2022)")   
            ax.plot(m_LN_upper, f_LN_upper, color="tab:green", linestyle="dotted")
        
        ax.set_xlim(1e10, 1e18)
        ax.set_ylim(10**(-12), 1)
        ax.set_xlabel("$M_c~[\mathrm{g}]$")
        ax.set_ylabel("$f_\mathrm{PBH}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("$\sigma={:.1f}$".format(sigma))
        
        ax1 = axes_plotting1[i]
        ax1.fill_between(mc_values_unevolved, abs(np.array(constraint_lower_unevolved) / np.array(constraint_lower_unevolved_LN_psi) - 1), abs(np.array(constraint_upper_unevolved)/np.array(constraint_upper_unevolved_LN_psi) -1), color="tab:blue")
        ax1.set_xlabel("$M_c~[\mathrm{g}]$")
        ax1.set_ylabel("$|\Delta f_\mathrm{PBH} / f_\mathrm{PBH}|$")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_ylim(fracdiff_ylims[i])
        ax1.set_xlim(1e14, 1e17)
        ax1.set_title("$\sigma={:.1f}$".format(sigma))
        
        if i == 0:
            ax.legend()
            
    fig.tight_layout()
    
    fig1.suptitle("Fractional difference (LN in $\phi$ / LN in $\psi$ - 1)")
    fig1.tight_layout()
