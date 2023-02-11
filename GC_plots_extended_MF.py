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
mc_min = 1e14
mc_max = 1e19
masses = 10**np.arange(np.log10(mc_min), np.log10(mc_max), 0.1)

Isatis_path = "./../Downloads/version_finale/scripts/Isatis/"



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
            for j in range(len(temp)-1):
                temp2 = "".join([temp2, temp[j], "\,\,"])
            temp2 = "".join([temp2, "\,\,[arXiv:",temp[-1], "]"])
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
    if mf == GCC:
        filename_append = "_GCC_Delta={:.1f}".format(Delta)
        
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
            for j in range(len(temp)-1):
                temp2 = "".join([temp2, temp[j], "\,\,"])
            temp2 = "".join([temp2, "\,\,[arXiv:",temp[-1], "]"])
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
    ax.set_xlim(mc_min, mc_max)
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
    ax2.set_xlim(mc_min, mc_max)
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
        ax1.set_xlim(mc_min, mc_max)
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
    ax3.set_xlim(mc_min, mc_max)
    ax3.set_ylim(0.007, 0.0115)
    ax3.legend(fontsize='small')
    ax3.set_title("Log-normal ($\sigma = {:.1f}$)".format(sigma))
    fig3.tight_layout()
    
#%%
from scipy.special import erf, loggamma


def skew_LN(m, m_c, sigma, alpha):
    # Skew-lognormal mass function, as defined in Eq. (8) of 2009.03204.
    return np.exp(-np.log(m/m_c)**2 / (2*sigma**2)) * (1 + erf( alpha * np.log(m/m_c) / (np.sqrt(2) * sigma))) / (np.sqrt(2*np.pi) * sigma * m)

def loc_param_GCC(m_p, alpha, beta):
    # Location parameter for critical collapse mass function, from Table I of 2009.03204.
    return m_p * np.power(beta/alpha, 1/beta)

def GCC(m, m_f, alpha, beta):
    log_psi = np.log(beta/m_f) - loggamma((alpha+1) / beta) + (alpha * np.log(m/m_f)) - np.power(m/m_f, beta)
    return np.exp(log_psi)


if "__main__" == __name__:
    
    # Choose which constraints to plot, and create labels.
    constraints_names = []
    constraints_extended_plotting = []
    
    # Extended mass function constraints using the method from 1705.05567.
    # Choose which constraints to plot, and create labels
    constraints_labels = []
    constraints_names = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]

    # Mass function parameter values, from 2009.03204.
    deltas = np.array([0., 0.1, 0.3, 0.5, 1.0, 2.0, 5.0])
    sigmas = np.array([0.55, 0.55, 0.57, 0.60, 0.71, 0.97, 2.77])
    alphas_SL = np.array([-2.27, -2.24, -2.07, -1.82, -1.31, -0.66, 1.39])

    alphas_CC = np.array([3.06, 3.09, 3.34, 3.82, 5.76, 18.9, 13.9])
    betas = np.array([2.12, 2.08, 1.72, 1.27, 0.51, 0.0669, 0.0206])
    
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    
    for k in range(len(deltas)):
        fig, ax = plt.subplots(figsize=(6,6))
        
        # Calculate constraints for extended MF from gamma-rays.
        f_pbh_skew_LN = []
        f_pbh_GCC = []
        
        params_SLN = [sigmas[k], alphas_SL[k]]
        params_GCC = [alphas_CC[k], betas[k]]

        constraints_extended_Carr_SLN = constraints_Carr_general(skew_LN, params_SLN)
        constraints_extended_Carr_GCC = constraints_Carr_general(GCC, params_GCC)
    
        # envelope of constraints, with the tightest constraint
        envelope_SLN = []
        envelope_GCC = []
        
        if deltas[k] == 0:
            constraints_Isatis_SLN, constraints_names= isatis_constraints_general(skew_LN, deltas[k])
            constraints_Isatis_GCC, constraints_names = isatis_constraints_general(GCC, deltas[k])
                           
            for i in range(len(constraints_names)):
                ax.plot(masses, constraints_Isatis_SLN[i], marker='x', linestyle='None', color=colors[i])
                ax.plot(masses, constraints_Isatis_GCC[i], marker='x', linestyle='None', color=colors[i])
        
        for i in range(len(constraints_names)):
            ax.plot(masses, constraints_extended_Carr_SLN[i], linestyle="dotted", label="SLN, " + str(constraints_names[i]), color=colors[i])
            ax.plot(masses, constraints_extended_Carr_GCC[i], linestyle="dashed", label="GCC, " + str(constraints_names[i]), color=colors[i])
                        
        for j in range(len(masses)):
            constraints_SLN = []
            constraints_GCC = []
            for l in range(len(constraints_names)):
                constraints_SLN.append(constraints_extended_Carr_SLN[l][j])
                constraints_GCC.append(constraints_extended_Carr_GCC[l][j])
                
            envelope_SLN.append(min(constraints_SLN))
            envelope_GCC.append(min(constraints_GCC))
        
        ax.plot(masses, envelope_SLN, linestyle="dotted", label="Envelope (SLN)", color="k")
        ax.plot(masses, envelope_GCC, linestyle="dashed", label="Envelope (GCC)", color="k")

        ax.set_xlabel(r"$M_c$ [g]")
        ax.set_ylabel(r"$f_\mathrm{PBH}$")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(1e-10, 1)
        ax.set_xlim(mc_min, mc_max)
        ax.legend(fontsize="small")
        ax.legend(title=r"$\Delta = {:.1f}$".format(deltas[k]))
        fig.tight_layout()
