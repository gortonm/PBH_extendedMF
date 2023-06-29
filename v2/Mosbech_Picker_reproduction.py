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
from isatis_reproduction import read_blackhawk_spectra
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

#%%
# Reproduce Fig. 1 of Mosbech & Picker (2022), using different forms of
# alpha_eff.
m_pbh_values_formation_BlackHawk = np.logspace(np.log10(4e14), 16, 50)
m_pbh_values_formation_wide = np.logspace(8, 18, 100)
pbh_lifetimes = []


for j in range(len(m_pbh_values_formation_BlackHawk)):
    
    destination_folder = "mass_evolution_v2" + "_{:.0f}".format(j+1)
    filename = os.path.expanduser('~') + "/Downloads/version_finale/results/" + destination_folder + "/life_evolutions.txt"
    data = np.genfromtxt(filename, delimiter="    ", skip_header=4, unpack=True, dtype='str')
    times = data[0]
    tau = float(times[-1])

    pbh_lifetimes.append(tau)   # add the last time value at which BlackHawk calculates the PBH mass


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


alpha_eff_values_BlackHawk = alpha_eff(np.array(pbh_lifetimes), m_pbh_values_formation_BlackHawk)

def alpha_eff_approx(M0_values):
    """
    Fitting formula used for alpha_eff, given in Eq. 10 of Mosbech & Picker (2022).

    Parameters
    ----------
    M0_values : Array-like
        PBH formation masses, in grams.

    Returns
    -------
    Array-like.
        Approximate value of alpha_eff.

    """
    c_1 = -0.3015
    c_2 = 0.3113
    p = -0.0008
    
    alpha_eff_values = []
    for M_0 in M0_values:
        if M_0 < 1e18:
            alpha_eff_values.append(c_1 + c_2 * M_0**p)
        else:
            alpha_eff_values.append(2.011e-4)
    return alpha_eff_values


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
    M0_extracted, alpha_eff_extracted_data = load_data("2203.05743/2203.05743_Fig2.csv")
    
    alpha_eff_values = np.interp(M0_values, M0_extracted, alpha_eff_extracted_data, left=max(alpha_eff_extracted_data), right=2.011e-4)
    return alpha_eff_values


def alpha_eff_mixed(M0_values):
    """
    Calculate alpha_eff, using the BlackHawk result in the mass range
    in which that is calculated, and the values extracted from Fig. 1 of Mosbech
    & Picker (2022) outside of that mass range.

    Parameters
    ----------
    M0_values : Array-like
        PBH formation masses, in grams.

    Returns
    -------
    Array-like.
        Value of alpha_eff.

    """    
    M0_min_BH, M0_max_BH = min(m_pbh_values_formation_BlackHawk), max(m_pbh_values_formation_BlackHawk)
    
    alpha_eff_values = []
    
    for M_0 in M0_values:
        if M0_min_BH < M_0 < M0_max_BH:
            alpha_eff_values.append(np.interp(M_0, m_pbh_values_formation_BlackHawk, alpha_eff_values_BlackHawk))
        else:
            alpha_eff_values.append(alpha_eff_extracted(M_0))
            
    return np.array(alpha_eff_values)


def m_pbh_evolved_MP23(M0_values, t):
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
        if M_0**3 - 3 * alpha_eff_mixed(np.array([M_0])) * m_Pl**3 * (t / t_Pl) <= 0:
            M_values.append(0)
        else:
            M_values.append(np.power(M_0**3 - 3 * alpha_eff_mixed(np.array([M_0])) * m_Pl**3 * (t / t_Pl), 1/3)[0])
    
    return np.array(M_values)


def m_pbh_formation_MP23(M_values, t):
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
    M_min = 7.56e14
    M0_test_values = np.logspace(np.log10(M_min), 18, 1000)
    M_evolved_test_values = m_pbh_evolved_MP23(M0_test_values, t)

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
    # PBH mass function at time t, evolved form the initial MF phi_formation using Eq. 11 
    M0_values = m_pbh_formation_MP23(M_values, t)
    return phi_formation * M_values**2 * np.power(M_values**3 + 3 * alpha_eff_mixed(M0_values) * m_Pl**3 * (t / t_Pl), -2/3)


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
    return phi_formation * M_values**2 * np.power(M_values**3 + 3 * alpha_eff_mixed(M0_values) * m_Pl**3 * (t / t_Pl), -2/3)


def psi_evolved_LN_number_density(m_c, sigma, t, log_m_factor=5, n_steps=10000, log_output=True):
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
        Number of multiples of sigma (in log-space) of masses around m_c to consider when estimating the maximum. The default is 5.
    n_steps : Integer, optional
        Number of masses at which to evaluate the evolved mass function. The default is 10000.
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
    #M0_test_values = np.sort(np.concatenate((np.arange(7.4687715114e14, 7.4687715115e14, 5e2), np.arange(7.4687715115e14, 7.47e14, 5e7), np.logspace(log_m_min, log_m_max, n_steps))))
    M0_test_values = np.sort(np.concatenate((np.arange(7.4687715114e14, 7.4687715115e14, 5e2),  np.logspace(log_m_min, log_m_max, n_steps))))
    M_test_values = m_pbh_evolved_MP23(M0_test_values, t)
    
    phi_formation = LN(M0_test_values, m_c, sigma)
    phi_evolved_values = phi_evolved_v2(phi_formation, M_test_values, M0_test_values, t)
       
    # Find mass * evolved values of phi (proportional to the mass distribution psi)
    psi_unnormalised = phi_evolved_values * M_test_values
    
    # Estimate the normalisation of psi (such that the evolved MF is normalised to 1)
    psi_normalisation = 1 / np.trapz(phi_formation*M0_test_values, M0_test_values)
    
    if log_output:
        return np.log10(M_test_values), np.log10(psi_unnormalised * psi_normalisation)
    
    else:
        return M_test_values, psi_unnormalised * psi_normalisation



def psi_LN_number_density(m, m_c, sigma, log_m_factor=5, n_steps=100000):
    # Distribution function for PBH energy density, when the number density follows a log-normal in the mass 
    
    log_m_min = np.log10(m_c) - log_m_factor*sigma
    log_m_max = np.log10(m_c) + log_m_factor*sigma

    m_pbh_values = np.logspace(log_m_min, log_m_max, n_steps)
    normalisation = 1 / np.trapz(LN(m_pbh_values, m_c, sigma) * m_pbh_values, m_pbh_values)
    return LN(m, m_c, sigma) * m * normalisation


if "__main__" == __name__:

    alpha_eff_approx_values = alpha_eff_approx(m_pbh_values_formation_BlackHawk)
    alpha_eff_extracted_values = alpha_eff_extracted(m_pbh_values_formation_wide)
    alpha_eff_mixed_values = alpha_eff_mixed(m_pbh_values_formation_wide)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(m_pbh_values_formation_BlackHawk, alpha_eff_values_BlackHawk, label="Calculated using BlackHawk")
    ax.plot(m_pbh_values_formation_BlackHawk, alpha_eff_approx_values, linestyle="dashed", label="Fitting formula (Eq. 10 MP '22)")
    ax.plot(m_pbh_values_formation_wide, alpha_eff_extracted_values, linestyle="None", marker="x", label="Extracted (Fig. 1 MP '22)")
    ax.plot(m_pbh_values_formation_wide, alpha_eff_mixed_values, linestyle="None", marker="+", label="Mixed (extracted and BlackHawk)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Formation mass $M_0$~[g]")
    ax.set_ylabel(r"$\alpha_\mathrm{eff}$")
    ax.legend(fontsize="small")
    fig.tight_layout()
    
    # Plot the present mass against formation mass
    fig, ax = plt.subplots(figsize=(6, 6))
    m_pbh_values_formation_plot = np.logspace(np.log10(5e14), 16, 500)
    ax.plot(m_pbh_values_formation_plot, m_pbh_values_formation_plot, linestyle="dotted", color="k", label="Formation mass = Present mass")
    ax.plot(m_pbh_values_formation_plot, m_pbh_evolved_MP23(m_pbh_values_formation_plot, t=t_0), marker="x", linestyle="None", label="Eq. 7 (MP '22)")
    
    # Test: plot formation mass against present mass
    m_evolved_test = m_pbh_evolved_MP23(m_pbh_values_formation_plot, t=t_0)
    m_formation_test = m_pbh_formation_MP23(m_evolved_test, t=t_0)
    ax.plot(m_formation_test, m_evolved_test, marker="+", linestyle="None", label="Inverting Eq. 7 (MP '22)")
    
    ax.set_xlabel("Formation mass $M_0$ [g]")
    ax.set_ylabel("Present mass $M$ [g]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(min(m_pbh_values_formation_plot), 1e16)
    ax.set_ylim(1e-1 * min(m_pbh_values_formation_plot), max(m_pbh_values_formation_plot))
    ax.legend()
    fig.tight_layout()
    
    
#%% Reproduce Fig. 2 of Mosbech & Picker (2022)
if "__main__" == __name__:
    
    m_pbh_values_formation = np.logspace(11, 17, 500)
    m_pbh_values_formation_to_evolve = np.concatenate((np.arange(7.4687715114e14, 7.4687715115e14, 5e2), np.arange(7.4687715115e14, 7.47e14, 5e7), np.logspace(np.log10(7.47e14), 17, 500)))
    m_pbh_values_evolved = m_pbh_evolved_MP23(m_pbh_values_formation_to_evolve, t_0)
    m_c = 1e15
    
    for sigma in [0.1, 0.5, 1, 1.5]:
        phi_initial = phi_LN(m_pbh_values_formation, m_c, sigma)
        phi_initial_to_evolve = phi_LN(m_pbh_values_formation_to_evolve, m_c, sigma)
                
        phi_present = phi_evolved_v2(phi_initial_to_evolve, m_pbh_values_evolved, m_pbh_values_formation_to_evolve, t_0)
        # test that the "evolved" mass function at t=0 matches the initial mass function.
        phi_test = phi_evolved_v2(phi_initial_to_evolve, m_pbh_values_formation_to_evolve, m_pbh_values_formation_to_evolve, 0)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(m_pbh_values_formation, phi_initial, label="$t=0$")
        ax.plot(m_pbh_values_evolved, phi_present, label="$t=t_0$", marker="x")
        ax.plot(m_pbh_values_formation_to_evolve, phi_test, label="$t=0$ (test)")

        ax.set_xlabel("$M~[\mathrm{g}]$")
        ax.set_ylabel("$\phi(M)~[\mathrm{g}]^{-1}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(title="$\sigma={:.1f}$".format(sigma), fontsize="small")
        ax.set_xlim(1e11, max(m_pbh_values_formation))
        ax.set_ylim(1e-21, 1e-12)
        fig.tight_layout()


#%% Plot the mass density distribution, for an initial distribution following a log-normal in the number density
# This cell includes a test of the method psi_evolved_LN_number_density()

if "__main__" == __name__:
    
    m_pbh_values_formation = np.logspace(11, 17, 100)
    m_pbh_values_formation_to_evolve = np.concatenate((np.arange(7.4687715114e14, 7.4687715116e14, 5e2), np.logspace(np.log10(7.47e14), 17, 100)))
    m_pbh_values_evolved = m_pbh_evolved_MP23(m_pbh_values_formation_to_evolve, t_0)
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
# This cell includes a test of the effect of not evolving the total PBH mass density with time on the resulting mass function psi(M),
# by comparing the initial and evolved psi(M) in a mass range where the effects of evaporation are negligible (M>~1e15g).

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
    
    n_steps_values = np.logspace(5, 1, 5)
    log_m_factors = np.linspace(10, 1, 10)
    
    for m_c in np.linspace(11, 15, 5):
        
        for sigma in [0.1, 0.5, 1.0]:
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
                    #M0_test_values = np.sort(np.concatenate((np.arange(7.4687715114e14, 7.4687715115e14, 5e2), np.arange(7.4687715115e14, 7.47e14, 5e7), np.logspace(log_m_min, log_m_max, n_steps))))
                    M0_test_values = np.sort(np.concatenate((np.arange(7.4687715114e14, 7.4687715115e14, 5e2),  np.logspace(log_m_min, log_m_max, n_steps))))
                    M_test_values = m_pbh_evolved_MP23(M0_test_values, t)
                    
                    phi_formation = LN(M0_test_values, m_c, sigma)
                    
                    # Estimate the normalisation of psi (such that the evolved MF is normalised to 1)
                    psi_normalisation = 1 / np.trapz(phi_formation*M0_test_values, M0_test_values)
                    
                    if log_m_factor == max(np.linspace(1, 10, 10)) and n_steps == max(np.logspace(1, 5, 5)):
                        psi_normalisation_most_acc = psi_normalisation
                        K_values_per_step.append(1)
        
                    else:
                        K_values_per_step.append(psi_normalisation / psi_normalisation_most_acc)
                    
                ax.plot(log_m_factors, K_values_per_step, marker="x", linestyle="None", label="{:.0f}".format(np.log10(n_steps)))
                    
            ax.set_title("$M_c={:.0e}$".format(m_c) + "g , $\sigma={:.1f}$".format(sigma))
            ax.legend(title="$\log_{10}(N_\mathrm{steps})$", fontsize="small")
            ax.set_ylabel=("$K / K_\mathrm{most~acc.}$")
            ax.set_xlabel("log_m_factor")
            ax.set_ylim(0, 10)
            fig.tight_layout()
            fig.savefig("./MP22_reproduction_pictures/K_comparison_Mc=1e{:.0f}g".format(np.log10(m_c)) + "_sigma=0p{:.0f}.pdf".format(10*sigma))
                

#%% Reproduce Fig. 3 of Mosbech & Picker (2022)

if "__main__" == __name__:
        
    # Create and save file for PBH mass and spin distribution
    BlackHawk_path = "./../../Downloads/version_finale/"
    
    # Initial line of each PBH mass spectrum file.
    spec_file_initial_line = "mass/spin \t 0.00000e+00"
    
    mc_values = [5e14, 1e15, 5e15, 1e16]
    sigma = 1.5
    m_pbh_values_formation_to_evolve = np.concatenate((np.arange(7.4687715115e14, 7.47e14, 5e7), np.logspace(np.log10(7.47e14), 17, 500)))
    m_pbh_values_evolved = m_pbh_evolved_MP23(m_pbh_values_formation_to_evolve, t_0)
    dlog10m = (np.log10(max(m_pbh_values_evolved)) - np.log10(min(m_pbh_values_evolved))) / (len(m_pbh_values_evolved) - 1)
    
    colors = ["lime", "limegreen", "green", "darkgreen"]
    
    
    b_max, l_max = np.radians(7/2), np.radians(7/2)    
    from isatis_reproduction import J_D
    D_factor = J_D(-l_max, l_max, -b_max, b_max)
    prefactors = []
    
    for i, m_c in enumerate(mc_values):
        
        PBH_mass_mean = m_c * np.exp(sigma**2 / 2)
        prefactor = D_factor / PBH_mass_mean
        prefactors.append(prefactor)
        
        spec_file = []
        spec_file.append(spec_file_initial_line)
    
        filename_BH_spec = BlackHawk_path + "/src/tables/users_spectra/" + "MP22_test_evolved_{:.0f}.txt".format(i)
        
        phi_formation = phi_LN(m_pbh_values_formation_to_evolve, m_c, sigma)
        phi_present = phi_evolved(phi_formation, m_pbh_values_evolved, t_0)
        spec_values = phi_present * m_pbh_values_evolved * dlog10m * np.log(10)
        print(spec_values[5:10])
        
        for j in range(len(m_pbh_values_evolved)):
            spec_file.append("{:.5e}\t{:.5e}".format(m_pbh_values_evolved[j], spec_values[j]))
            
        np.savetxt(filename_BH_spec, spec_file, fmt="%s", delimiter = " = ")            

    f_pbh = 1e-8
    
    # Load gamma ray spectrum calculated from BlackHawk
    file_path_BlackHawk_data = BlackHawk_path + "/results/"
    
    E_5e14, spectrum_5e14 = read_blackhawk_spectra(file_path_BlackHawk_data + "MP22_test_5e14/instantaneous_secondary_spectra.txt")
    E_1e15, spectrum_1e15 = read_blackhawk_spectra(file_path_BlackHawk_data + "MP22_test_1e15/instantaneous_secondary_spectra.txt")
    E_5e15, spectrum_5e15 = read_blackhawk_spectra(file_path_BlackHawk_data + "MP22_test_5e15/instantaneous_secondary_spectra.txt")
    E_1e16, spectrum_1e16 = read_blackhawk_spectra(file_path_BlackHawk_data + "MP22_test_1e16/instantaneous_secondary_spectra.txt")

    E_5e14_evolved, spectrum_5e14_evolved = read_blackhawk_spectra(file_path_BlackHawk_data + "MP22_test_5e14_evolved/instantaneous_secondary_spectra.txt")
    E_1e15_evolved, spectrum_1e15_evolved = read_blackhawk_spectra(file_path_BlackHawk_data + "MP22_test_1e15_evolved/instantaneous_secondary_spectra.txt")
    E_5e15_evolved, spectrum_5e15_evolved = read_blackhawk_spectra(file_path_BlackHawk_data + "MP22_test_5e15_evolved/instantaneous_secondary_spectra.txt")
    E_1e16_evolved, spectrum_1e16_evolved = read_blackhawk_spectra(file_path_BlackHawk_data + "MP22_test_1e16_evolved/instantaneous_secondary_spectra.txt")


    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(E_5e14, prefactors[0] * f_pbh * E_5e14**2 * spectrum_5e14, label=r"$5\times 10^{14}$", linestyle="dotted", color=colors[0])
    ax.plot(E_5e14_evolved, prefactors[0] * f_pbh * E_5e14_evolved**2 * spectrum_5e14_evolved, color=colors[0])
    
    ax.plot(E_1e15, prefactors[1] * f_pbh * E_1e15**2 * spectrum_1e15, label=r"$1\times 10^{15}$", linestyle="dotted", color=colors[1])
    ax.plot(E_1e15_evolved, prefactors[1] * f_pbh * E_1e15_evolved**2 * spectrum_1e15_evolved, color=colors[1])
    
    ax.plot(E_5e15, prefactors[2] * f_pbh * E_5e15**2 * spectrum_5e15, label=r"$5\times 10^{15}$", linestyle="dotted", color=colors[2])
    ax.plot(E_5e15_evolved, prefactors[2] * f_pbh * E_5e15_evolved**2 * spectrum_5e15_evolved, color=colors[2])
    
    ax.plot(E_1e16, prefactors[3] * f_pbh * E_1e16**2 * spectrum_1e16, label=r"$1\times 10^{16}$", linestyle="dotted", color=colors[3])
    ax.plot(E_1e16_evolved, prefactors[3] * f_pbh * E_1e16_evolved**2 * spectrum_1e16_evolved, color=colors[3])
    
    ax.plot(0, 0, color="k", label="\nEvolved")    
    ax.plot(0, 0, linestyle="dotted", color="k", label="Lognormal")

    ax.legend(title="$M_*~[\mathrm{g}]$")
    ax.set_xlabel("Particle energy $E$ [GeV]")
    ax.set_ylabel("$\gamma$ flux: $E^2 \mathrm{d}^2 N / \mathrm{d}E\mathrm{d}t~[\mathrm{GeV}~\mathrm{s}^{-1}~\mathrm{cm}^{-2}]$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-6, 2e5)
    ax.set_ylim(1e-19, 1e-3)
    ax.set_title("$\sigma={:.1f}$".format(sigma) + " (PYTHIA)")
    fig.tight_layout()
    

#%% Reproduce Fig. 4 of Mosbech & Picker (2022) 

if "__main__" == __name__:

    # Load data from HESS (Abramowski et al. 2016, 1603.07730)
    E_lower_y_HESS, flux_lower_y_HESS = load_data("1603.07730/1603.07730_lower_y.csv")
    E_upper_y_HESS, flux_upper_y_HESS = load_data("1603.07730/1603.07730_upper_y.csv")
    E_lower_HESS, flux_mid_HESS = load_data("1603.07730/1603.07730_x_bins.csv")
    
    # Divide HESS data by a factor of 10 to account for the shift in Fig. 3 of Abramowski et al. (2016)
    flux_mid_HESS /= 10
    flux_lower_y_HESS /= 10
    flux_upper_y_HESS /= 10
    
    # widths of energy bins
    E_minus_HESS = E_upper_y_HESS - E_lower_HESS[:-1]
    E_plus_HESS = E_lower_HESS[1:] - E_upper_y_HESS
    
    # upper and lower error bars on flux values
    flux_plus_HESS = flux_upper_y_HESS - flux_mid_HESS[:-1]
    flux_minus_HESS = flux_mid_HESS[:-1] - flux_lower_y_HESS
    
    
    # Load data from FermiLAT (Abramowski et al. 2016, 1512.01846)
    E_lower_y_FermiLAT, flux_lower_y_FermiLAT_sys = load_data("1512.01846/1512.01846_lower_y_sys.csv")
    E_lower_y_FermiLAT, flux_lower_y_FermiLAT_stat = load_data("1512.01846/1512.01846_lower_y_stat.csv")
    E_upper_y_FermiLAT, flux_upper_y_FermiLAT_sys = load_data("1512.01846/1512.01846_upper_y_sys.csv")
    E_upper_y_FermiLAT, flux_upper_y_FermiLAT_stat = load_data("1512.01846/1512.01846_upper_y_stat.csv")
    E_lower_FermiLAT, flux_mid_FermiLAT = load_data("1512.01846/1512.01846_x_bins.csv")
    
    # widths of energy bins
    E_minus_FermiLAT = E_upper_y_FermiLAT - E_lower_FermiLAT[:-1]
    E_plus_FermiLAT = E_lower_FermiLAT[1:] - E_upper_y_FermiLAT
    
    # upper and lower error bars on flux values
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
    
    # Print fluxes (rather than E^2 * fluxes)
    print(flux_mid_FermiLAT[:-1] / E_lower_y_FermiLAT**2)
    print(flux_minus_FermiLAT / E_lower_y_FermiLAT**2)
    print(flux_plus_FermiLAT / E_lower_y_FermiLAT**2)
    

#%% Plot the photon spectrum for different PBH masses

if "__main__" == __name__:
    M_values_eval = np.logspace(10, 18, 50)
    fig, ax = plt.subplots(figsize=(6,6))
    
    for i in range(len(M_values_eval)):
        if (i+1) % 4 == 0 and i < 20:
            filepath = os.path.expanduser('~') + "/Downloads/version_finale/results/GC_mono_PYTHIA_v2_{:.0f}/".format(i+1)
            energies, spectrum = read_blackhawk_spectra(filepath + "instantaneous_secondary_spectra.txt")
            ax.plot(energies[200:500], spectrum[200:500], label="{:.2e} g".format(M_values_eval[i]))
            
    ax.set_xlabel("$E~[\mathrm{GeV}]$")
    ax.set_ylabel("$\mathrm{d}^2 N / \mathrm{d}E\mathrm{d}t~[\mathrm{GeV}^{-1}~\mathrm{sr}^{-1}~\mathrm{s}^{-1}~\mathrm{cm}^{-2}]$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e19, 1e30)
    ax.legend(fontsize="small")
    fig.tight_layout()


#%% Plot the Fermi-LAT constraints (monochromatic MF)
from extended_MF_checks import envelope, constraint_Carr

b_max, l_max = np.radians(3.5), np.radians(3.5)

def Delta(l_min, l_max, b_min, b_max):
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


from extended_MF_checks import load_results_Isatis

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
       
    
    # Plot the monochromatic MF constraint
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

#%% Calculate constraints for an extended MF (initial lognormal and evolved lognormal)

if "__main__" == __name__:
    M0_values = np.logspace(9, 18, 50)
    
    sigmas = [0.1, 0.5, 1., 1.5]
    mc_values = [3e14, 3e13, 1e12, 2e10]
    
    for i in range(len(sigmas)):
        fig, ax = plt.subplots(figsize=(6,6))
        initial_MF = psi_LN_number_density(M0_values, mc_values[i], sigmas[i])
        
        M_values = m_pbh_evolved_MP23(M0_values, t=t_0)
        evolved_MF = phi_evolved(initial_MF, M_values, t=t_0)
        
        ax.plot(M0_values, initial_MF, linestyle="dotted")
        ax.plot(M_values, evolved_MF)
        ax.set_xlabel("$M~[\mathrm{g}$]")
        ax.set_ylabel("$\psi(M)$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.vlines(5e10, ymin=min(initial_MF), ymax=max(initial_MF), color="k", linestyle="dotted", label=r"$M=5\times 10^{10}~\mathrm{g}$")
        ax.hlines(1e-4 * max(initial_MF), xmin=min(M_values), xmax=max(M_values), color="k", linestyle="dashed", label="$\psi = \psi_\mathrm{max} / 10^4$")
        ax.legend()
        ax.set_title("$M_c={:.0e}$g, $\sigma={:.1f}$".format(mc_values[i], sigmas[i]))
        ax.set_xlim(min(M0_values), max(M0_values))
        fig.tight_layout()
        
        # Check if the mass function is normalised to a value close to 1
        print(np.trapz(initial_MF, M0_values))
        print(np.trapz(evolved_MF, M_values))


#%% Plot constraints for extended MF (reproducing Fig. 4 of Mosbech & Picker (2022)).

if "__main__" == __name__:
    # Constraints data for each energy bin of each instrument (extended MF)
    
    constraints_mono_file_lower = np.transpose(np.genfromtxt("./Data/fPBH_GC_full_all_bins_Fermi-LAT_1512.01846_lower_monochromatic_wide.txt"))
    constraints_mono_file_upper = np.transpose(np.genfromtxt("./Data/fPBH_GC_full_all_bins_Fermi-LAT_1512.01846_upper_monochromatic_wide.txt"))
        
    M_values_eval = np.logspace(10, 18, 100)   # masses at which the constraint is evaluated for a delta-function MF
    mc_values = np.logspace(14, 17, 50)
 
    # Final constraint
    constraint_lower = []
    constraint_upper = []
    
    # Constraint from each energy bin
    energy_bin_constraints_lower = []
    energy_bin_constraints_upper = []
    
    sigma = 0.5
    
    params_LN = [sigma]
    
    # Unevolved mass function
    for k in range(len(constraints_mono_file_lower)):

        # Constraint from a particular energy bin
        constraint_energy_bin = constraints_mono_file_lower[k]

        # Calculate constraint on f_PBH from each bin
        f_PBH_k = constraint_Carr(mc_values, m_mono=M_values_eval, f_max=constraint_energy_bin, mf=psi_LN_number_density, params=params_LN)
        energy_bin_constraints_lower.append(f_PBH_k)

    for k in range(len(constraints_mono_file_upper)):

        # Constraint from a particular energy bin
        constraint_energy_bin = constraints_mono_file_upper[k]

        # Calculate constraint on f_PBH from each bin
        f_PBH_k = constraint_Carr(mc_values, m_mono=M_values_eval, f_max=constraint_energy_bin, mf=psi_LN_number_density, params=params_LN)
        energy_bin_constraints_upper.append(f_PBH_k)
        
    constraint_lower.append(envelope(energy_bin_constraints_lower))
    constraint_upper.append(envelope(energy_bin_constraints_upper))
    

    # Evolved mass function
    constraint_lower_evolved = []
    constraint_upper_evolved = []
    
    for m_c in mc_values:
                
        # Evolved mass function
        log_m_evolved, log_psi_evolved = psi_evolved_LN_number_density(m_c, sigma, t_0, n_steps=1000)   # evolved PBH distribution, evaluated at present masses corresponding to the formation masses in M0_values
        # Interpolate evolved mass function at the evolved masses at which the delta-function MF constraint is calculated
        mf_evolved_interp = 10**np.interp(np.log10(M_values_eval), log_m_evolved, log_psi_evolved)
        
        # Constraint from each energy bin
        f_PBH_energy_bin_lower = []
        for k in range(len(constraints_mono_file_lower)):
    
            # Constraint from a particular energy bin (delta function MF)
            constraint_energy_bin = constraints_mono_file_lower[k]
            
            integrand = mf_evolved_interp / constraint_energy_bin
            integral = np.trapz(np.nan_to_num(integrand), M_values_eval)
            print(integrand)
            print(M_values_eval)
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
    ax.fill_between(mc_values, constraint_lower[0], constraint_upper[0], color="tab:green", alpha=0.5, label="Log-normal \n (unevolved)")
    ax.plot(m_LN_lower, f_LN_lower, color="tab:green", linestyle="dotted")   
    ax.plot(m_LN_upper, f_LN_upper, color="tab:green", linestyle="dotted")

    ax.fill_between(mc_values, constraint_lower_evolved, constraint_upper_evolved, color="tab:purple", alpha=0.5, label="Evolved")
    ax.plot(m_evolved_lower, f_evolved_lower, color="tab:purple", linestyle="dotted")   
    ax.plot(m_evolved_upper, f_evolved_upper, color="tab:purple", linestyle="dotted")
    ax.plot(0,0, linestyle="dotted", color="k", label="Extracted from \n Mosbech \& \n Picker (2022)")
    ax.plot(0,0, color="k", label="Reproduced")
    
    ax.set_xlim(1e10, 1e18)
    ax.set_ylim(10**(-15), 1)
    ax.set_xlabel("$M_c~[\mathrm{g}]$")
    ax.set_ylabel("$f_\mathrm{PBH}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("$\sigma={:.1f}$ (Comparing flux per sr)".format(sigma))
    ax.legend(fontsize="small")
    plt.tight_layout()