#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 18:25:20 2022

@author: ppxmg2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Produce plots of the Galctic Centre photon constraints on PBHs, for 
# extended mass functions.

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


filepath = './Extracted_files/'

# Range of log-normal mass function central PBH masses.
"""
mc_min = 1e14
mc_max = 1e19
masses = 10**np.arange(np.log10(mc_min), np.log10(mc_max), 0.1)
"""
# Range of masses at which the mass function peaks.
m_min = 1e14
m_max = 1e19
masses = 10**np.arange(np.log10(m_min), np.log10(m_max), 0.1)

Isatis_path = "./../Downloads/version_finale/scripts/Isatis/"

mcs_SLN_Gow22 = np.exp(np.array([4.13, 4.13, 4.15, 4.21, 4.40, 4.88, 5.41]))
mps_SLN_Gow22 = np.array([40.9, 40.9, 40.9, 40.8, 40.8, 40.6, 32.9])


def load_data(filename):
    """
    Load data from a file located in the folder './Extracted_files/'.

    Parameters
    ----------
    fname : String
        File name.

    Returns
    -------
    Array-like.
        Contents of file.
    """
    return np.genfromtxt(filepath+filename, delimiter=',', unpack=True)


def LN_MF_density(m, m_c, sigma, A=1):
    """Log-normal distribution function (for PBH mass density).

    Parameters
    ----------
    m : Float
        PBH mass, in grams.
    m_c : Float
        Critical (median) PBH mass for log-normal mass function, in grams.
    sigma : Float
        Width of the log-normal distribution.
    A : Float, optional
        Amplitude of the distribution. The default is 1.

    Returns
    -------
    Array-like
        Log-normal distribution function (for PBH masses).

    """
    return A * np.exp(-np.log(m/m_c)**2 / (2*sigma**2)) / (np.sqrt(2*np.pi) * sigma * m**2)


def LN_MF_number_density(m, m_c, sigma, A=1):
    """Log-normal distribution function (for PBH number density).

    Parameters
    ----------
    m : Float
        PBH mass, in grams.
    m_c : Float
        Critical (median) PBH mass for log-normal mass function, in grams.
    sigma : Float
        Width of the log-normal distribution.
    A : Float, optional
        Amplitude of the distribution. The default is 1.

    Returns
    -------
    Array-like
        Log-normal distribution function (for PBH masses).

    """
    return A * np.exp(-np.log(m/m_c)**2 / (2*sigma**2)) / (np.sqrt(2*np.pi) * sigma * m)


def f_max(m, m_GC_mono, f_max_GC_mono):
    """Linearly interpolate the maximum fraction of dark matter in PBHs (monochromatic mass distribution).

    Parameters
    ----------
    m : Array-like
        PBH masses (in grams).
    m_GC_mono : Array-like
        PBH masses for the monochromatic MF, to use for interpolation.
    f_max_GC_mono : Array-like
        Constraint on abundance of PBHs (loaded data, monochromatic MF).

    Returns
    -------
    Array-like
        Maximum observationally allowed fraction of dark matter in PBHs for a
        monochromatic mass distribution.

    """
    return 10**np.interp(np.log10(m), np.log10(m_GC_mono), np.log10(f_max_GC_mono))


def integrand(A, m, m_c, sigma, m_GC_mono, f_max_GC_mono):
    """Compute integrand appearing in Eq. 12 of 1705.05567 for a log-normal mass function (for reproducing constraints with an extended mass function following 1705.05567).

    Parameters
    ----------
    A : Float.
        Amplitude of log-normal mass function.
    m : Array-like
        PBH masses (in grams).
    m_c : Float
        Critical (median) PBH mass for log-normal mass function (in grams).
    sigma : Float
        Width of the log-normal distribution.
    m_GC_mono : Array-like
        PBH masses for the monochromatic MF, to use for interpolation.
    f_max_GC_mono : Array-like
        Constraint on abundance of PBHs (loaded data, monochromatic MF).

    Returns
    -------
    Array-like
        Integrand appearing in Eq. 12 of 1705.05567.

    """
    return LN_MF_number_density(m, m_c, sigma, A) / f_max(m, m_GC_mono, f_max_GC_mono)


def integrand_general_mf(m, mf, m_c, params, m_GC_mono, f_max_GC_mono):
    """Compute integrand appearing in Eq. 12 of 1705.05567 for a general mass
    function.

    Parameters
    ----------
    m : Array-like
        PBH masses (in grams).
    mf : Function
        PBH mass function.
    m_c : Float
        Characteristic PBH mass.
    params : Array-like
        Other parameters of the PBH mass function.
    m_GC_mono : Array-like
        PBH masses for the monochromatic MF, to use for interpolation.
    f_max_GC_mono : Array-like
        Constraint on abundance of PBHs (loaded data, monochromatic MF).

    Returns
    -------
    Array-like
        Integrand appearing in Eq. 12 of 1705.05567.

    """
    return mf(m, m_c, *params) / f_max(m, m_GC_mono, f_max_GC_mono)


def isatis_constraints(sigma, lognormal_MF=True):
    """Output constraints on f_PBH for a log-normal MF, calculated directly using Isatis.

    Parameters
    ----------
    sigma : Float
        Width of the mass function.
    lognormal_MF : Boolean, optional
        If True, use a log-normal mass function. The default is True.

    Returns
    -------
    constraints_extended_plotting : Array-like
        Constraints on f_PBH.
    constraints_names : Array-like
        Names of the constraints used.

    """
    if lognormal_MF:
        filename_append = "_lognormal_sigma={:.1f}".format(sigma)

    # Load result from Isatis
    results_name = "results_photons_GC%s" % (filename_append)

    constraints_file = np.genfromtxt("%s%s.txt" % (Isatis_path,results_name), dtype="str")
    constraints_names_bis = constraints_file[0, 1:]
    constraints = np.zeros([len(constraints_file)-1, len(constraints_file[0])-1])
    for i in range(len(constraints)):
        for j in range(len(constraints[0])):
            constraints[i, j] = float(constraints_file[i+1, j+1])

    # Choose which constraints to plot, and create labels.
    constraints_names = []
    constraints_extended_plotting = []

    for i in range(len(constraints_names_bis)):
        # Only include labels for constraints that Isatis calculated.
        if not(np.all(constraints[:, i] == -1.) or np.all(constraints[:, i] == 0.)):
            temp = constraints_names_bis[i].split("_")
            temp2 = ""
            """
            for j in range(len(temp)-1):
                temp2 = "".join([temp2, temp[j], "\,\,"])
            temp2 = "".join([temp2, "\,\,[arXiv:",temp[-1], "]"])
            """
            for j in range(len(temp)-1):
                temp2 = "".join([temp2, temp[j]])
            temp2 = "".join([temp2, "[arXiv:",temp[-1], "]"])

            constraints_names.append(temp2)
            constraints_extended_plotting.append(constraints[:, i])

    return constraints_extended_plotting, constraints_names


def isatis_constraints_general(mf, Delta, lognormal_MF=True):
    """Output constraints for a general extended mass function, calculated directly using Isatis.

    Parameters
    ----------
    mf : Function
        PBH mass function.
    Delta : Float
        Power spectrum width generating the PBH MF.

    Returns
    -------
    constraints_extended_plotting : Array-like
        Constraints on f_PBH.
    constraints_names : Array-like
        Names of the constraints used.

    """
    if mf == skew_LN:
        filename_append = "_SLN_Delta={:.1f}".format(Delta)
    if mf == CC3:
        filename_append = "_CC3_Delta={:.1f}".format(Delta)
        
    # Load result from Isatis
    results_name = "results_photons_GC%s" % (filename_append)

    constraints_file = np.genfromtxt("%s%s.txt" % (Isatis_path,results_name), dtype="str")
    constraints_names_bis = constraints_file[0, 1:]
    constraints = np.zeros([len(constraints_file)-1, len(constraints_file[0])-1])
    for i in range(len(constraints)):
        for j in range(len(constraints[0])):
            constraints[i, j] = float(constraints_file[i+1, j+1])

    # Choose which constraints to plot, and create labels.
    constraints_names = []
    constraints_extended_plotting = []

    for i in range(len(constraints_names_bis)):
        # Only include labels for constraints that Isatis calculated.
        if not(np.all(constraints[:, i] == -1.) or np.all(constraints[:, i] == 0.)):
            temp = constraints_names_bis[i].split("_")
            temp2 = ""
            """
            for j in range(len(temp)-1):
                temp2 = "".join([temp2, temp[j], "\,\,"])
            temp2 = "".join([temp2, "\,\,[arXiv:",temp[-1], "]"])
            """
            for j in range(len(temp)-1):
                temp2 = "".join([temp2, temp[j]])
            temp2 = "".join([temp2, "[arXiv:",temp[-1], "]"])

            constraints_names.append(temp2)
            constraints_extended_plotting.append(constraints[:, i])

    return constraints_extended_plotting, constraints_names



def constraints_Carr(sigma, lognormal_MF=True):
    """Calculate constraints for a log-normal MF, using the method from 1705.05567.

    Parameters
    ----------
    sigma : Float
        Width of the mass function.
    lognormal_MF : Boolean, optional
        If True, use a log-normal mass function. The default is True.

    Returns
    -------
    constraints_extended_Carr : Array-like
        Constraints on f_PBH.

    """
    # Constraints for monochromatic MF, calculated using isatis_reproduction.py.
    masses_mono = 10**np.arange(11, 19.05, 0.1)
    constraints_mono_calculated = []

    constraints_names = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
    constraints_extended_Carr = []

    # Loop through instruments
    for i in range(len(constraints_names)):

        constraints_mono_file = np.transpose(np.genfromtxt("./Data/fPBH_GC_full_all_bins_%s_monochromatic.txt"%(constraints_names[i])))

        # Constraint from given instrument
        constraint_extended_Carr = []
        constraint_mono_Carr = []

        for l in range(len(masses_mono)):
            constraint_mass_m = []
            for k in range(len(constraints_mono_file)):   # cycle over bins
                constraint_mass_m.append(constraints_mono_file[k][l])

            constraint_mono_Carr.append(min(constraint_mass_m))

        constraints_mono_calculated.append(constraint_mono_Carr)
        
        # Loop through central PBH masses
        for m_c in masses:

            # Constraint from each energy bin
            constraints_over_bins_extended = []

            # Loop through energy bins
            for j in range(len(constraints_mono_file)):
                f_max_values = constraints_mono_file[j]

                # Only include positive values of f_max.
                # Exclude f_max > 100, since including these can cause 
                # overflow errors.
                masses_mono_truncated = masses_mono[f_max_values < 1e2]
                f_max_truncated = f_max_values[f_max_values < 1e2]

                masses_mono_truncated = masses_mono_truncated[f_max_truncated > 0]
                f_max_truncated = f_max_truncated[f_max_truncated > 0]

                masses_mono_truncated = masses_mono_truncated[f_max_truncated != float("inf")]
                f_max_truncated = f_max_truncated[f_max_truncated != float("inf")]

                # If all values of f_max are excluded, assign a non-
                # physical value f_PBH = 10 to the constraint at that mass.
                if len(f_max_truncated) == 0:
                    constraints_over_bins_extended.append(10.)
                else:
                    # Constraint from each bin
                    constraints_over_bins_extended.append(1/np.trapz(integrand(1, masses_mono_truncated, m_c, sigma, masses_mono_truncated, f_max_truncated), masses_mono_truncated))

            # Constraint from given instrument (extended MF)
            constraint_extended_Carr.append(min(constraints_over_bins_extended))

        constraints_extended_Carr.append(np.array(constraint_extended_Carr))

    return constraints_extended_Carr


def constraints_Carr_general(mf, params):
    """Calculate constraints for a general extended mass function, using the method from 1705.05567.

    Parameters
    ----------
    mf : Function
        PBH mass function.
    params : Array-like
        Other parameters of the PBH mass function.

    Returns
    -------
    constraints_extended_Carr : Array-like
        Constraints on f_PBH.

    """
    # Constraints for monochromatic MF, calculated using isatis_reproduction.py.
    masses_mono = 10**np.arange(11, 19.05, 0.1)
    constraints_mono_calculated = []

    constraints_names = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
    constraints_extended_Carr = []

    # Loop through instruments
    for i in range(len(constraints_names)):

        constraints_mono_file = np.transpose(np.genfromtxt("./Data/fPBH_GC_full_all_bins_%s_monochromatic.txt"%(constraints_names[i])))

        # Constraint from given instrument
        constraint_extended_Carr = []
        constraint_mono_Carr = []

        for l in range(len(masses_mono)):
            constraint_mass_m = []
            for k in range(len(constraints_mono_file)):   # cycle over bins
                constraint_mass_m.append(constraints_mono_file[k][l])

            constraint_mono_Carr.append(min(constraint_mass_m))

        constraints_mono_calculated.append(constraint_mono_Carr)

        # Loop through central PBH masses
        for m_c in masses:

            # Constraint from each energy bin
            constraints_over_bins_extended = []

            # Loop through energy bins
            for j in range(len(constraints_mono_file)):
                f_max_values = constraints_mono_file[j]

                # Only include positive values of f_max.
                # Exclude f_max > 100, since including these can cause 
                # overflow errors.
                masses_mono_truncated = masses_mono[f_max_values < 1e2]
                f_max_truncated = f_max_values[f_max_values < 1e2]

                masses_mono_truncated = masses_mono_truncated[f_max_truncated > 0]
                f_max_truncated = f_max_truncated[f_max_truncated > 0]

                masses_mono_truncated = masses_mono_truncated[f_max_truncated != float("inf")]
                f_max_truncated = f_max_truncated[f_max_truncated != float("inf")]

                # If all values of f_max are excluded, assign a non-
                # physical value f_PBH = 10 to the constraint at that mass.
                if len(f_max_truncated) == 0:
                    constraints_over_bins_extended.append(10.)
                else:
                    # Constraint from each bin
                    integral = np.trapz(integrand_general_mf(masses_mono_truncated, mf, m_c, params, masses_mono_truncated, f_max_truncated), masses_mono_truncated)
                    if integral == 0:
                        constraints_over_bins_extended.append(10)
                    else:
                        constraints_over_bins_extended.append(1/integral)

            # Constraint from given instrument (extended MF)
            constraint_extended_Carr.append(min(constraints_over_bins_extended))

        constraints_extended_Carr.append(np.array(constraint_extended_Carr))

    return constraints_extended_Carr
 

#%%

if "__main__" == __name__:
    
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    
    fig, ax = plt.subplots(figsize=(6,6))
    fig1, (ax1a, ax1b) = plt.subplots(nrows=2, ncols=1, figsize=(6.5, 10))
    fig2, ax2 = plt.subplots(figsize=(6,6))
    fig3, ax3 = plt.subplots(figsize=(6,6))
    #fig4, ax4 = plt.subplots(figsize=(6,6))
    
    sigma = 0.5
    constraints_extended_plotting, constraints_names = isatis_constraints(sigma)
        
    constraints_extended_Isatis_sigma05, constraints_names = isatis_constraints(sigma=0.5)
    constraints_extended_Isatis_sigma10, constraints_names = isatis_constraints(sigma=1.)

    constraints_extended_Carr_sigma05 = constraints_Carr(sigma=0.5)
    constraints_extended_Carr_sigma10 = constraints_Carr(sigma=1.)
    constraints_extended_Carr = constraints_Carr(sigma=sigma)
    
    for i in range(len(constraints_names)):
        ax.plot(masses, constraints_extended_plotting[i], marker='x', label=constraints_names[i])
        ax1a.plot(masses, constraints_extended_Carr_sigma05[i], color=colors[i], alpha=0.5)
        ax1b.plot(masses, constraints_extended_Carr_sigma10[i], color=colors[i], alpha=0.5)
        
    ax.set_xlabel("$M_c$ [g]")
    ax.set_ylabel("$f_\mathrm{PBH}$")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e-10, 1)
    ax.set_xlim(m_min, m_max)
    ax.legend(fontsize='small')
    ax.set_title("Log-normal ($\sigma = {:.1f}$) \n (Direct Isatis calculation)".format(sigma))
    fig.tight_layout()
    
    # Extended mass function constraints using the method from 1705.05567.
    # Choose which constraints to plot, and create labels
    constraints_labels = []
    constraints_names = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
    
    for i in range(len(constraints_names)):
        temp = constraints_names[i].split("_")
        temp2 = ""
        for j in range(len(temp)-1):
            temp2 = "".join([temp2, temp[j], "\,\,"])
        temp2 = "".join([temp2, "\,\,[arXiv:", temp[-1], "]"])
        constraints_labels.append(temp2)
        
    # Constraints for monochromatic MF, calculated using Isatis.
    results_name_mono = "results_photons_GC_mono"
    
    constraints_mono_file = np.genfromtxt("%s%s.txt"%(Isatis_path,results_name_mono),dtype = "str")
    constraints_names_bis = constraints_mono_file[0,1:]
    constraints_mono = np.zeros([len(constraints_mono_file)-1,len(constraints_mono_file[0])-1])
    
    for i in range(len(constraints_mono)):
        for j in range(len(constraints_mono[0])):
            constraints_mono[i,j] = float(constraints_mono_file[i+1,j+1])
    
    # Choose which constraints to plot, and create labels.
    constraints_mono_names = []
    constraints_mono_plotting = []
    
    for i in range(len(constraints_names_bis)):
        if np.all(constraints_mono[:, i] <= 0.):  # only include calculated constraints
            print("all = -1 or 0")
        else:
            temp = constraints_names_bis[i].split("_")
            temp2 = ""
            for j in range(len(temp)-1):
                temp2 = "".join([temp2,temp[j],"\,\,"])
            temp2 = "".join([temp2,"\,\,[arXiv:",temp[-1],"]"])
            constraints_mono_names.append(temp2)
            constraints_mono_plotting.append(constraints_mono[:, i])
    
    for i in range(len(constraints_extended_Carr)):
        ax2.plot(masses, constraints_extended_Carr[i], marker='x', label=constraints_labels[i])
        ax1a.plot(masses, constraints_extended_Carr_sigma05[i], marker='x', linestyle='None', color=colors[i], label=constraints_labels[i])
        ax1b.plot(masses, constraints_extended_Carr_sigma10[i], marker='x', linestyle='None', color=colors[i], label=constraints_labels[i])
        ax3.plot(masses, constraints_extended_plotting[i]/constraints_extended_Carr[i] - 1, marker='x', color=colors[i], label=constraints_labels[i])
       
    # Plot the log-normal mass function constraints, calculated using the method
    # from 1705.05567.
    ax2.set_xlabel("$M_c$ [g]")
    ax2.set_ylabel("$f_\mathrm{PBH}$")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-10, 1)
    ax2.set_xlim(m_min, m_max)
    ax2.legend(fontsize='small')
    ax2.set_title("Log-normal ($\sigma = {:.1f}$) \n (1705.05567 method)".format(sigma))
    fig2.tight_layout()
    
    # Comparison of log-normal mass function constraints (crosses), calculated 
    # using the method from 1705.05567, and direct Isatis calculation (translucent 
    # lines).
    for ax1 in (ax1a, ax1b):
        ax1.set_xlabel("$M_c$ [g]")
        ax1.set_ylabel("$f_\mathrm{PBH}$")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_ylim(1e-10, 1)
        ax1.set_xlim(m_min, m_max)
    ax1b.legend(fontsize='small')
    ax1a.set_title("$\sigma = 0.5$")
    ax1b.set_title("$\sigma = 1.0$")
    fig1.suptitle("Log-normal")
    fig1.tight_layout()
    
    # Comparison of log-normal mass function constraints (crosses), calculated 
    # using the method from 1705.05567, and direct Isatis calculation (translucent 
    # lines).
    ax3.set_xlabel("$M_c$ [g]")
    ax3.set_ylabel("$f_\mathrm{PBH, Isatis} / f_\mathrm{PBH, Carr} - 1$")
    ax3.set_xscale("log")
    ax3.set_xlim(m_min, m_max)
    ax3.set_ylim(0.007, 0.0115)
    ax3.legend(fontsize='small')
    ax3.set_title("Log-normal ($\sigma = {:.1f}$)".format(sigma))
    fig3.tight_layout()
    
#%%
from scipy.special import erf, loggamma


def skew_LN(m, m_c, sigma, alpha):
    """
    Skew-lognormal mass function, as defined in Eq. (8) of 2009.03204.

    Parameters
    ----------
    m : Array-like
        PBH masses.
    m_c : Float
        Characteristic PBH mass.
    sigma : Float
        Parameter controlling width of mass function (corresponds to the standard deviation when alpha=0).
    alpha : Float
        Parameter controlling skew of the distribution.

    Returns
    -------
    Array-like
        Values of the mass function, evaluated at m.

    """
    return np.exp(-np.log(m/m_c)**2 / (2*sigma**2)) * (1 + erf( alpha * np.log(m/m_c) / (np.sqrt(2) * sigma))) / (np.sqrt(2*np.pi) * sigma * m)


def skew_LN_peak(m, m_p, sigma, alpha, Delta_index):
    """
    Skew-lognormal mass function, as defined in Eq. (8) of 2009.03204. Expressed in terms of the peak mass instead of M_c.

    Parameters
    ----------
    m : Array-like
        PBH masses.
    m_p : Float
        PBH mass at which the mass function peaks.
    sigma : Float
        Parameter controlling width of mass function (corresponds to the standard deviation when alpha=0).
    alpha : Float
        Parameter controlling skew of the distribution.
    Delta_index : Integer
        Index corresponding to the value of the power spectrum width Delta adopted.

    Returns
    -------
    Array-like
        Values of the mass function, evaluated at m.

    """
    m_c = m_p * mcs_SLN_Gow22[Delta_index] / mps_SLN_Gow22[Delta_index]
    return np.exp(-np.log(m/m_c)**2 / (2*sigma**2)) * (1 + erf( alpha * np.log(m/m_c) / (np.sqrt(2) * sigma))) / (np.sqrt(2*np.pi) * sigma * m)



def loc_param_CC3(m_p, alpha, beta):
    """
    Location parameter for generalised critical collapse mass function, from Table I of 2009.03204 (denoted m_f in that paper).
    
    Parameters
    ----------
    m_p : Array-like
        Peak mass of generalised critical collapse mass function.
    alpha : Float
        Parameter controlling width of mass function.
    beta : Float
        Parameter controlling strength of tail at small masses.

    Returns
    -------
    Array-like
        Location parameter.

    """
    return m_p * np.power(beta/alpha, 1/beta)


def CC3(m, m_p, alpha, beta):
    """
    Generalised critical collapse mass function, as defined in Eq. (9) of 2009.03204.

    Parameters
    ----------
    m : Array-like
        PBH masses.
    m_p : Float
        Peak mass of generalised critical collapse mass function.
    alpha : Float
        Parameter controlling width of mass function.
    beta : Float
        Parameter controlling strength of tail at small masses.

    Returns
    -------
    Array-like
        Values of the mass function, evaluated at m.

    """
    m_f = loc_param_CC3(m_p, alpha, beta)
    log_psi = np.log(beta/m_f) - loggamma((alpha+1) / beta) + (alpha * np.log(m/m_f)) - np.power(m/m_f, beta)
    return np.exp(log_psi)

mcs_SLN_Gow22 = np.exp(np.array([4.13, 4.13, 4.15, 4.21, 4.40, 4.88, 5.41]))
mps_SLN_Gow22 = np.array([40.9, 40.9, 40.9, 40.8, 40.8, 40.6, 32.9])


if "__main__" == __name__:
    
    # Choose which constraints to plot, and create labels.
    constraints_names = []
    constraints_extended_plotting = []
    
    # Extended mass function constraints using the method from 1705.05567.
    # Choose which constraints to plot, and create labels
    constraints_labels = []
    constraints_names = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]

    # Mass function parameter values, from 2009.03204.
    Deltas = np.array([0., 0.1, 0.3, 0.5, 1.0, 2.0, 5.0])
    sigmas = np.array([0.55, 0.55, 0.57, 0.60, 0.71, 0.97, 2.77])
    alphas_SL = np.array([-2.27, -2.24, -2.07, -1.82, -1.31, -0.66, 1.39])

    alphas_CC = np.array([3.06, 3.09, 3.34, 3.82, 5.76, 18.9, 13.9])
    betas = np.array([2.12, 2.08, 1.72, 1.27, 0.51, 0.0669, 0.0206])
    
    mcs_SLN_Gow22 = np.exp(np.array([4.13, 4.13, 4.15, 4.21, 4.40, 4.88, 5.41]))
    mps_SLN_Gow22 = np.array([40.9, 40.9, 40.9, 40.8, 40.8, 40.6, 32.9])
    
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    
    for k in range(len(Deltas)):
        fig, ax = plt.subplots(figsize=(12,6))
        
        # Calculate constraints for extended MF from gamma-rays.
        
        f_pbh_skew_LN = []
        f_pbh_CC3 = []
        
        params_SLN_peak = [sigmas[k], alphas_SL[k], k]
        params_CC3 = [alphas_CC[k], betas[k]]

        constraints_extended_Carr_SLN = constraints_Carr_general(skew_LN_peak, params_SLN_peak)
        constraints_extended_Carr_CC3 = constraints_Carr_general(CC3, params_CC3)
    
        # envelope of constraints, with the tightest constraint
        envelope_SLN = []
        envelope_CC3 = []
        
        mc_values_SLN = masses
        mp_values_CC3 = masses
        mp_values_SLN = mc_values_SLN * (mps_SLN_Gow22[k] / mcs_SLN_Gow22[k])
        
        if Deltas[k] == 0:
            constraints_Isatis_SLN, constraints_names = isatis_constraints_general(skew_LN, Deltas[k])
            constraints_Isatis_CC3, constraints_names = isatis_constraints_general(CC3, Deltas[k])
                           
            for i in range(len(constraints_names)):
                ax.plot(mp_values_SLN, constraints_Isatis_SLN[i], linestyle="dotted", color=colors[i])
                ax.plot(mp_values_CC3, constraints_Isatis_CC3[i], linestyle="dashed", color=colors[i])
        
        for i in range(len(constraints_names)):
            ax.plot(mp_values_SLN, constraints_extended_Carr_SLN[i], marker='x', linestyle='None', label="SLN, " + str(constraints_names[i]), color=colors[i])
            ax.plot(mp_values_CC3, constraints_extended_Carr_CC3[i], marker='x', linestyle='None', label="CC3, " + str(constraints_names[i]), color=colors[i])
                        
        for j in range(len(masses)):
            constraints_SLN = []
            constraints_CC3 = []
            
            for l in range(len(constraints_names)):
                constraints_SLN.append(constraints_extended_Carr_SLN[l][j])
                constraints_CC3.append(constraints_extended_Carr_CC3[l][j])
                
            envelope_SLN.append(min(constraints_SLN))
            envelope_CC3.append(min(constraints_CC3))
        
        ax.plot(mp_values_SLN, envelope_SLN, linestyle="dotted", label="Envelope (SLN)", color="k")
        ax.plot(mp_values_CC3, envelope_CC3, linestyle="dashed", label="Envelope (CC3)", color="k")
        
        ax.set_xlabel(r"$M_p$ [g]")
        ax.set_ylabel(r"$f_\mathrm{PBH}$")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(1e-10, 1)
        ax.set_xlim(m_min, m_max)
        ax.legend(fontsize="small")
        ax.legend(title=r"$\Delta = {:.1f}$".format(Deltas[k]))
        fig.tight_layout()
        plt.savefig("./Figures/GC_constraints/Delta={:.1f}.png".format(Deltas[k]))
        
        # Save constraints data
        
        for i in range(len(constraints_names)):
            data_filename_SLN = "./Data_files/constraints_extended_MF/SLN_GC_" + str(constraints_names[i]) + "_Carr_Delta={:.1f}".format(Deltas[k])
            data_filename_CC3 = "./Data_files/constraints_extended_MF/CC3_GC_" + str(constraints_names[i]) + "_Carr_Delta={:.1f}".format(Deltas[k])
            np.savetxt(data_filename_SLN, [mp_values_SLN, constraints_extended_Carr_SLN[i]], delimiter="\t")
            np.savetxt(data_filename_CC3, [mp_values_CC3, constraints_extended_Carr_CC3[i]], delimiter="\t")
        
        data_filename_SLN = "./Data_files/constraints_extended_MF/SLN_GC_envelope_Carr_Delta={:.1f}".format(Deltas[k])
        data_filename_CC3 = "./Data_files/constraints_extended_MF/CC3_GC_envelope_Carr_Delta={:.1f}".format(Deltas[k])
        np.savetxt(data_filename_SLN, np.array([mp_values_SLN, envelope_SLN]), fmt="%s", delimiter="\t")
        np.savetxt(data_filename_CC3, np.array([mp_values_CC3, envelope_CC3]), fmt="%s", delimiter="\t")
       

#%% Constraints for a log-normal MF, using the values of sigma found as best-
# fit values from Table II of 2008.03289.
Deltas = np.array([0., 0.1, 0.3, 0.5, 1.0, 2.0])
sigmas_LN = np.array([0.374, 0.377, 0.395, 0.430, 0.553, 0.864])

if "__main__" == __name__:
    
    # Choose which constraints to plot, and create labels.
    constraints_names = []
    constraints_extended_plotting = []
    
    # Extended mass function constraints using the method from 1705.05567.
    # Choose which constraints to plot, and create labels
    constraints_labels = []
    constraints_names = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
    
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    
    for k in range(len(Deltas)):
        fig, ax = plt.subplots(figsize=(12,6))
        
        # Calculate constraints for extended MF from gamma-rays.
        
        f_pbh_N = []
        
        constraints_extended_Carr_LN = constraints_Carr(sigma=sigmas_LN[k])
    
        # envelope of constraints, with the tightest constraint
        envelope_LN = []
        
        mc_values_LN = masses
        mp_values_LN = mc_values_LN * np.exp(sigmas_LN[k]**2)
        
        print("m_c / m_p = ", np.exp(sigmas_LN[k]**2))
                
        for i in range(len(constraints_names)):
            ax.plot(mp_values_LN, constraints_extended_Carr_LN[i], marker='x', linestyle='None', label="LN, " + str(constraints_names[i]), color=colors[i])
                        
        for j in range(len(masses)):
            constraints_LN = []
            
            for l in range(len(constraints_names)):
                constraints_LN.append(constraints_extended_Carr_LN[l][j])
                
            envelope_LN.append(min(constraints_LN))
        
        ax.plot(mp_values_LN, envelope_LN, linestyle="dotted", label="Envelope (SLN)", color="k")
        
        ax.set_xlabel(r"$M_p$ [g]")
        ax.set_ylabel(r"$f_\mathrm{PBH}$")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(1e-10, 1)
        ax.set_xlim(m_min, m_max)
        ax.legend(fontsize="small")
        ax.legend(title=r"$\Delta = {:.1f}$".format(Deltas[k]))
        fig.tight_layout()
        plt.savefig("./Figures/GC_constraints/Delta={:.1f}.png".format(Deltas[k]))
        
        # Save constraints data
        
        for i in range(len(constraints_names)):
            data_filename_LN = "./Data_files/constraints_extended_MF/LN_GC_" + str(constraints_names[i]) + "_Carr_Delta={:.1f}".format(Deltas[k])
            np.savetxt(data_filename_LN, [mp_values_LN, constraints_extended_Carr_LN[i]], delimiter="\t")
        
        data_filename_LN = "./Data_files/constraints_extended_MF/LN_GC_envelope_Carr_Delta={:.1f}".format(Deltas[k])
        np.savetxt(data_filename_LN, np.array([mp_values_LN, envelope_LN]), fmt="%s", delimiter="\t")



#%% Test case: compare results obtained using a LN MF with sigma=0.5 to the
# input skew LN MF with alpha=0
if "__main__" == __name__:

    fig, ax = plt.subplots(figsize=(12,6))
    
    # Calculate constraints for extended MF from gamma-rays.
    
    f_pbh_skew_LN = []
    f_pbh_CC3 = []
    
    sigma = 0.5
    params_SLN = [sigma, 0.]
    
    constraints_extended_Carr_SLN = constraints_Carr_general(skew_LN, params_SLN)
    constraints_extended_Carr_LN = constraints_Carr(sigma)
    
    mc_values = masses
    constraints_names = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    
    for i in range(len(constraints_names)):
        ax.plot(mc_values, constraints_extended_Carr_SLN[i], label=(constraints_names[i]), color=colors[i])
        ax.plot(mc_values, constraints_extended_Carr_LN[i], marker='x', color=colors[i])
    
    ax.plot(0, 5, color='k', label=r"SLN ($\alpha=0, \sigma={:.1f}$)".format(sigma))
    ax.plot(0, 5, color='k', marker='x', linestyle="None", label="LN ($\sigma={:.1f})$".format(sigma))
    ax.set_xlabel(r"$M_c$ [g]")
    ax.set_ylabel(r"$f_\mathrm{PBH}$")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e-10, 1)
    ax.set_xlim(m_min, m_max)
    ax.legend(fontsize="small")
    fig.tight_layout()
    plt.savefig("./Figures/GC_constraints/test_LN_SLN.png")


#%% Check formula for the approximate relation between M_c in the SLN and
# M_p in the CC3 MFs.


if "__main__" == __name__:

    Deltas = np.array([0., 0.1, 0.3, 0.5, 1.0, 2.0, 5.0])
    sigmas = np.array([0.55, 0.55, 0.57, 0.60, 0.71, 0.97, 2.77])
    alphas_SL = np.array([-2.27, -2.24, -2.07, -1.82, -1.31, -0.66, 1.39])
    alphas_CC = np.array([3.06, 3.09, 3.34, 3.82, 5.76, 18.9, 13.9])
    betas = np.array([2.12, 2.08, 1.72, 1.27, 0.51, 0.0669, 0.0206])
    
    mcs_SLN_Gow22 = np.exp(np.array([4.13, 4.13, 4.15, 4.21, 4.40, 4.88, 5.41]))
    mps_SLN_Gow22 = np.array([40.9, 40.9, 40.9, 40.8, 40.8, 40.6, 32.9])
    
    mp_CC3 = 1e20
    m = np.logspace(17, 23, 1000)
    
    for i in range(len(Deltas)):
        fig, ax = plt.subplots(figsize=(6, 6))
        mc_SLN = mp_CC3 * mcs_SLN_Gow22[i] / mps_SLN_Gow22[i]
        
        psi_SLN = skew_LN(m, mc_SLN, sigmas[i], alphas_SL[i]) / max(skew_LN(m, mc_SLN, sigmas[i], alphas_SL[i]))
        psi_CC3 = CC3(m, mp_CC3, alphas_CC[i], betas[i]) / max(CC3(m, mp_CC3, alphas_CC[i], betas[i]))
        
        #xmin = m[min(min(np.where(psi_SLN > 0.1)))]
        #xmax = m[max(max(np.where(psi_SLN > 0.1)))]
        xmin = 1e18
        xmax = 1e22
    
        ax.plot(m, psi_SLN, label="SLN")
        ax.plot(m, psi_CC3, label="CC3", linestyle="dashed")
        ax.set_xlabel(r"$M_c$ [g]")
        ax.set_ylabel(r"$f_\mathrm{PBH}$")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r"$M$ [g]")
        ax.set_ylabel(r"$\psi(M) / \psi_\mathrm{max}$")
        ax.legend(title=r"$\Delta = {:.1f}$".format(Deltas[i]))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(0.1, 2)
        fig.tight_layout()
        plt.savefig("./Figures/Test_plots/test_comp_MP_MC_Delta={:.1f}.png".format(Deltas[i]))
