#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 18:25:20 2022

@author: ppxmg2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import erf, loggamma

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

# Range of characteristic PBH masses
m_min = 1e14
m_max = 1e19
masses = 10**np.arange(np.log10(m_min), np.log10(m_max), 0.1)

# Path to Isatis
Isatis_path = "./../Downloads/version_finale/scripts/Isatis/"

# Values of the peak mass m_p and characteristic mass m_c from 2009.03204 Table
# II.
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


def lognormal_number_density(m, m_c, sigma):
    """Log-normal distribution function (for PBH number density).

    Parameters
    ----------
    m : Array-like
        PBH masses.
    m_c : Float
        Characteristic PBH mass.
    sigma : Float
        Standard deviation of the log-normal distribution.

    Returns
    -------
    Array-like
        Log-normal distribution function (for PBH masses).

    """
    return np.exp(-np.log(m/m_c)**2 / (2*sigma**2)) / (np.sqrt(2*np.pi) * sigma * m)


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
        Parameter controlling width of mass function (corresponds to the 
        standard deviation when alpha=0).
    alpha : Float
        Parameter controlling skew of the distribution.

    Returns
    -------
    Array-like
        Values of the mass function, evaluated at m.

    """
    return np.exp(-np.log(m/m_c)**2 / (2*sigma**2)) * (1 + erf( alpha * np.log(m/m_c) / (np.sqrt(2) * sigma))) / (np.sqrt(2*np.pi) * sigma * m)


def loc_param_CC3(m_p, alpha, beta):
    """
    Location parameter for generalised critical collapse 3 mass function, from 
    Table I of 2009.03204 (denoted m_f in that paper).

    Parameters
    ----------
    m_p : Array-like
        Peak mass of generalised critical collapse 3 mass function.
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
    Generalised critical collapse 3 mass function, as defined in Eq. (9) of 
    2009.03204.

    Parameters
    ----------
    m : Array-like
        PBH masses.
    m_p : Float
        Peak mass of generalised critical collapse 3 mass function.
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


def f_max(m, m_mono, f_max_mono):
    """Linearly interpolate the maximum fraction of dark matter in PBHs 
    (monochromatic mass distribution).

    Parameters
    ----------
    m : Array-like
        PBH masses (in grams).
    m_mono : Array-like
        PBH masses for the monochromatic MF, to use for interpolation.
    f_max_mono : Array-like
        Constraint on abundance of PBHs (loaded data, monochromatic MF).

    Returns
    -------
    Array-like
        Maximum observationally allowed fraction of dark matter in PBHs for a
        monochromatic mass distribution.

    """
    return 10**np.interp(np.log10(m), np.log10(m_mono), np.log10(f_max_mono))


def integrand_general_mf(m, mf, m_c, params, m_mono, f_max_mono):
    """Compute integrand appearing in Eq. 12 of 1705.05567 for a general mass
    function.

    Parameters
    ----------
    m : Array-like
        PBH masses (in grams).
    mf : Function
        PBH mass function.
    m_c : Float
        Characteristic PBH mass (m_c for a (skew-)lognormal, m_p for CC3).
    params : Array-like
        Other parameters of the PBH mass function.
    m_mono : Array-like
        PBH masses for the monochromatic MF, to use for interpolation.
    f_max_mono : Array-like
        Constraint on abundance of PBHs (loaded data, monochromatic MF).

    Returns
    -------
    Array-like
        Integrand appearing in Eq. 12 of 1705.05567.

    """
    return mf(m, m_c, *params) / f_max(m, m_mono, f_max_mono)


def isatis_constraints_general(mf, Delta_index=0, sigma=0.5):
    """Output constraints for a general extended mass function, calculated 
    directly using Isatis.

    Parameters
    ----------
    mf : Function
        PBH mass function.
    Delta_index : Integer
        Index corresponding to the value of the power spectrum width Delta.
        The default is 0.
    sigma : Float
        Standard deviation of the lognormal distribution. The default is 0.5.

    Returns
    -------
    constraints_extended_plotting : Array-like
        Constraints on f_PBH.
    """
    if mf == lognormal_number_density:
        filename_append = "_lognormal_sigma={:.3f}".format(sigma)
    elif mf == skew_LN:
        filename_append = "_SLN_Delta={:.1f}".format(Deltas[Delta_index])
    elif mf == CC3:
        filename_append = "_CC3_Delta={:.1f}".format(Deltas[Delta_index])
        
    # Load result from Isatis
    results_name = "results_photons_GC%s" % (filename_append)

    constraints_file = np.genfromtxt("%s%s.txt" % (Isatis_path,results_name), dtype="str")
    constraints_names_bis = constraints_file[0, 1:]
    constraints = np.zeros([len(constraints_file)-1, len(constraints_file[0])-1])
    for i in range(len(constraints)):
        for j in range(len(constraints[0])):
            constraints[i, j] = float(constraints_file[i+1, j+1])
            
    # Choose which constraints to plot.
    constraints_extended_plotting = []
            
    for i in range(len(constraints_names_bis)):
        if not(np.all(constraints[:, i] == -1.) or np.all(constraints[:, i] == 0.)):
            constraints_extended_plotting.append(constraints[:, i])

    return constraints_extended_plotting


def constraints_Carr_general(mf, params):
    """Calculate constraints from Galactic centre gamma rays for a general 
    extended mass function, using the method from 1705.05567.

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
    masses_mono = 10**np.arange(11, 21.05, 0.1)

    constraints_names = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
    constraints_extended_Carr = []

    # Loop through instruments
    for i in range(len(constraints_names)):

        # Monochromatic MF constraints
        constraints_mono_file = np.transpose(np.genfromtxt("./Data/fPBH_GC_full_all_bins_%s_monochromatic.txt"%(constraints_names[i])))

        # Constraint from given instrument
        constraint_extended_Carr = []

        # Extended MF constraints calculation
        # Loop through characteristic PBH masses
        for m_c in masses:

            # Constraint from each energy bin
            constraints_over_bins_extended = []

            # Loop through energy bins (j = 0, 1,..., n_bins - 1)
            for j in range(len(constraints_mono_file)):
                f_max_values = constraints_mono_file[j]

                # Only include finite, positive values of f_max.
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
            
        # Array contains constraints from each instrument
        constraints_extended_Carr.append(np.array(constraint_extended_Carr))

    return constraints_extended_Carr


#%% Test case: compare Isatis output (for monochromatic MF) [unmodified version]
# with the constraint shown in Fig. 3 of 2201.01265

m_pbh_values = 10**np.arange(11, 21.05, 0.1)
Mmin = min(m_pbh_values)  # minimum PBH mass in your runs
Mmax = max(m_pbh_values)  # maximum PBH mass in your runs

# Load monochromatic MF results from evaporation, and calculate the envelope of constraints.
envelope_evap_mono = []

# Load result from Isatis
Isatis_path = './../Downloads/version_finale_unmodified/scripts/Isatis/'
results_name = "results_photons_GC_mono"

constraints_file = np.genfromtxt("%s%s.txt" % (Isatis_path,results_name), dtype="str")
constraints_names_bis = constraints_file[0, 1:]
constraints = np.zeros([len(constraints_file)-1, len(constraints_file[0])-1])
for i in range(len(constraints)):
    for j in range(len(constraints[0])):
        constraints[i, j] = float(constraints_file[i+1, j+1])

# Choose which constraints to plot, and create labels.
constraints_names = []
constraints_extended_plotting = []

fig, ax = plt.subplots(figsize=(6,6))

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


for j in range(len(m_pbh_values)):
    constraints_mono = []
    for l in range(len(constraints_names)):
        constraints_mono.append(constraints_extended_plotting[l][j])
        
    # use absolute value sign to not include unphysical cases with
    # f_PBH = -1 in the envelope
    envelope_evap_mono.append(min(abs(np.array(constraints_mono))))

# Plot evaporation constraints loaded from Isatis, to check the envelope
# works correctly
colors_evap = ["tab:blue", "tab:orange", "tab:red", "tab:green"]

for i in range(len(constraints_names)):
    ax.plot(m_pbh_values, constraints_extended_plotting[i], label=constraints_names[i], color=colors_evap[i])
ax.plot(m_pbh_values, envelope_evap_mono, label="Envelope", color="k", linestyle="dashed")
ax.set_xlim(1e14, 1e18)
ax.set_ylim(10**(-10), 1)
ax.set_xlabel("$M_\mathrm{PBH}~[\mathrm{g}]$")
ax.set_ylabel("$f_\mathrm{PBH}$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(fontsize="small")
plt.tight_layout()
#plt.savefig("./Figures/Combined_constraints/test_envelope_GC_excl_highestE_bin.png")
plt.title("Excluding highest-energy bin")
plt.tight_layout()


#%% Test: Plot Isatis constraints on Galactic centre photons for a monochromatic MF.

m_pbh_values = 10**np.arange(11, 21.05, 0.1)
Mmin = min(m_pbh_values)  # minimum PBH mass in your runs
Mmax = max(m_pbh_values)  # maximum PBH mass in your runs

# Load monochromatic MF results from evaporation, and calculate the envelope of constraints.
envelope_evap_mono = []

# Load result from Isatis
Isatis_path = './../Downloads/version_finale/scripts/Isatis/'
results_name = "results_photons_GC_mono"

constraints_file = np.genfromtxt("%s%s.txt" % (Isatis_path,results_name), dtype="str")
constraints_names_bis = constraints_file[0, 1:]
constraints = np.zeros([len(constraints_file)-1, len(constraints_file[0])-1])
for i in range(len(constraints)):
    for j in range(len(constraints[0])):
        constraints[i, j] = float(constraints_file[i+1, j+1])

# Choose which constraints to plot, and create labels.
constraints_names = []
constraints_extended_plotting = []

fig, ax = plt.subplots(figsize=(6,6))

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


for j in range(len(m_pbh_values)):
    constraints_mono = []
    for l in range(len(constraints_names)):
        constraints_mono.append(constraints_extended_plotting[l][j])
        
    # use absolute value sign to not include unphysical cases with
    # f_PBH = -1 in the envelope
    envelope_evap_mono.append(min(abs(np.array(constraints_mono))))
    
    
# compare to constraints calculated using isatis_reproduction.py
constraints_names_comp = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
constraints_mono_calculated = []

# Loop through instruments
for i in range(len(constraints_names_comp)):

    constraints_mono_file = np.transpose(np.genfromtxt("./Data/fPBH_GC_full_all_bins_%s_monochromatic.txt"%(constraints_names_comp[i])))

    # Constraint from given instrument
    constraint_extended_Carr = []
    constraint_mono_Carr = []

    for l in range(len(m_pbh_values)):
        constraint_mass_m = []
        for k in range(len(constraints_mono_file)):   # cycle over bins
            constraint_mass_m.append(constraints_mono_file[k][l])

        constraint_mono_Carr.append(min(constraint_mass_m))
    
    constraints_mono_calculated.append(constraint_mono_Carr)


# Plot evaporation constraints loaded from Isatis, to check the envelope
# works correctly
colors_evap = ["tab:blue", "tab:orange", "tab:red", "tab:green"]

for i in range(len(constraints_names)):
    ax.plot(m_pbh_values, constraints_extended_plotting[i], label=constraints_names[i], color=colors_evap[i])
    ax.plot(m_pbh_values, constraints_mono_calculated[i], color=colors_evap[i], linestyle="None", marker="x")
ax.plot(m_pbh_values, envelope_evap_mono, label="Envelope", color="k", linestyle="dashed")
ax.plot(0, 0, color="grey", label="Isatis")
ax.plot(0, 0, marker="x", linestyle="None", color="grey", label="isatis_reproduction.py")
ax.set_xlim(1e14, 1e18)
ax.set_ylim(10**(-10), 1)
ax.set_xlabel("$M_\mathrm{PBH}~[\mathrm{g}]$")
ax.set_ylabel("$f_\mathrm{PBH}$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(fontsize="small")
plt.tight_layout()
#plt.savefig("./Figures/Combined_constraints/test_envelope_GC.png")


#%% Test: compare results obtained using a LN MF using the method from 
# 1705.05567 to the result obtained directly using Isatis.

Isatis_path = './../Downloads/version_finale/scripts/Isatis/'

if "__main__" == __name__:
    sigmas_LN = np.array([0.374, 0.377, 0.395, 0.430, 0.553, 0.864, 0.01])

    for sigma in sigmas_LN:

        fig, ax = plt.subplots(figsize=(12,6))
                
        constraints_extended_Carr_LN = constraints_Carr_general(lognormal_number_density, [sigma])
        constraints_extended_Isatis_LN = isatis_constraints_general(lognormal_number_density, sigma=sigma)
        
        mc_values = masses
        constraints_names = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        
        for i in range(len(constraints_names)):
            ax.plot(mc_values, constraints_extended_Isatis_LN[i], label=(constraints_names[i]), color=colors[i])
            ax.plot(mc_values, constraints_extended_Carr_LN[i], linestyle="None", marker='x', color=colors[i])
        
        ax.plot(0, 5, color='k', label=r"Direct Isatis calculation")
        ax.plot(0, 5, color='k', marker='x', linestyle="None", label="Carr et al. method")
        ax.set_xlabel(r"$M_c$ [g]")
        ax.set_ylabel(r"$f_\mathrm{PBH}$")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(1e-10, 1)
        ax.set_xlim(m_min, m_max)
        ax.legend(fontsize="small", title="LN ($\sigma={:.3f})$".format(sigma))
        fig.tight_layout()
        #plt.savefig("./Figures/GC_constraints/test_LN_Carr_Isatis.png")
    

#%% Test: compare results obtained using a SLN MF using the method from 
# 1705.05567 to the result obtained directly using Isatis.

# Mass function parameter values, from 2009.03204.
Deltas = np.array([0., 0.1, 0.3, 0.5, 1.0, 2.0, 5.0])
sigmas_SLN = np.array([0.55, 0.55, 0.57, 0.60, 0.71, 0.97, 2.77])
alphas_SL = np.array([-2.27, -2.24, -2.07, -1.82, -1.31, -0.66, 1.39])

Isatis_path = './../Downloads/version_finale/scripts/Isatis/'

if "__main__" == __name__:
    
    for k in range(len(Deltas)):

        fig, ax = plt.subplots(figsize=(12,6))
        params_SLN = [sigmas_SLN[k], alphas_SL[k]]
            
        constraints_extended_Carr_SLN = constraints_Carr_general(skew_LN, params_SLN)
        constraints_extended_Isatis_SLN = isatis_constraints_general(skew_LN, Delta_index=k)
        
        mc_values = masses
        constraints_names = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        
        for i in range(len(constraints_names)):
            ax.plot(mc_values, constraints_extended_Isatis_SLN[i], label=(constraints_names[i]), color=colors[i])
            ax.plot(mc_values, constraints_extended_Carr_SLN[i], linestyle="None", marker='x', color=colors[i])
        
        ax.plot(0, 5, color='k', label=r"Direct Isatis calculation")
        ax.plot(0, 5, color='k', marker='x', linestyle="None", label="Carr et al. method")
        ax.set_xlabel(r"$M_c$ [g]")
        ax.set_ylabel(r"$f_\mathrm{PBH}$")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(1e-10, 1)
        ax.set_xlim(m_min, m_max)
        ax.legend(fontsize="small", title="SLN ($\Delta={:.1f})$".format(Deltas[k]))
        fig.tight_layout()
        #plt.savefig("./Figures/GC_constraints/test_SLN_Carr_Isatis.png")
    
    
#%% Test: compare results obtained using the CC3 MF using the method from 
# 1705.05567 to the result obtained directly using Isatis.

# Mass function parameter values, from 2009.03204.
Deltas = np.array([0., 0.1, 0.3, 0.5, 1.0, 2.0, 5.0])
alphas_CC = np.array([3.06, 3.09, 3.34, 3.82, 5.76, 18.9, 13.9])
betas = np.array([2.12, 2.08, 1.72, 1.27, 0.51, 0.0669, 0.0206])

Isatis_path = './../Downloads/version_finale/scripts/Isatis/'

if "__main__" == __name__:
    
    for k in range(len(Deltas)):
        
        fig, ax = plt.subplots(figsize=(12,6))
        params_CC3 = [alphas_CC[k], betas[k]]
            
        constraints_extended_Carr_CC3 = constraints_Carr_general(CC3, params_CC3)
        constraints_extended_Isatis_CC3 = isatis_constraints_general(CC3, Delta_index=k)
        
        mc_values = masses
        constraints_names = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        
        for i in range(len(constraints_names)):
            ax.plot(mc_values, constraints_extended_Isatis_CC3[i], label=(constraints_names[i]), color=colors[i])
            ax.plot(mc_values, constraints_extended_Carr_CC3[i], linestyle="None", marker='x', color=colors[i])
        
        ax.plot(0, 5, color='k', label=r"Direct Isatis calculation")
        ax.plot(0, 5, color='k', marker='x', linestyle="None", label="Carr et al. method")
        ax.set_xlabel(r"$M_c$ [g]")
        ax.set_ylabel(r"$f_\mathrm{PBH}$")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(1e-10, 1)
        ax.set_xlim(m_min, m_max)
        ax.legend(fontsize="small", title="CC3 ($\Delta={:.1f})$".format(Deltas[k]))
        fig.tight_layout()
        #plt.savefig("./Figures/GC_constraints/test_CC3_Carr_Isatis.png")


#%% Test: compare the MFs to those used in BlackHawk.
if "__main__" == __name__:
    mf = skew_LN
    
    check_index = 45   # corresponds to the value of m_c loaded
    Delta_index = 4    # corresponds to the value of Delta
    sigma = 0.864    # standard deviation of lognormal mass function
    masses_mono = 10**np.arange(11, 21.05, 0.1)
    
    if mf == lognormal_number_density:
        filename_append = "_LN_sigma={:.3f}".format(sigma)
        mf_values = lognormal_number_density(masses_mono, mc_values[check_index], sigma)
    elif mf == skew_LN:
        filename_append = "_SLN_Delta={:.1f}".format(Deltas[Delta_index])
        mf_values = skew_LN(masses_mono, mc_values[check_index], sigmas_SLN[Delta_index], alphas_SL[Delta_index])
    elif mf == CC3:
        filename_append = "_CC3_Delta={:.1f}".format(Deltas[Delta_index])
        mf_values = CC3(masses_mono, mc_values[check_index], alphas_CC[Delta_index], betas[Delta_index])
        
    # Load mass function saved by BlackHawk:
    filepath = "../Downloads/version_finale/results/GC%s_%i/BH_spectrum.txt" % (filename_append, check_index)
    data = np.genfromtxt(filepath, skip_header=3).transpose()
    m_Isatis, mf_Isatis = data[0], data[1]
            
    fig, ax = plt.subplots()
    ax.plot(m_Isatis, mf_Isatis, label="BlackHawk MF")
    ax.plot(masses_mono, mf_values, label="Exact MF")
    ax.set_xlabel(r"$M_\mathrm{PBH}$ [g]")
    ax.set_ylabel(r"$\psi(M_\mathrm{PBH})$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.legend(fontsize="small")
    plt.tight_layout()
    #plt.savefig("./Figures/Test_plots/test_MF%s_index=%i.png" % (filename_append, check_index))

#%% Test case: compare results obtained using a LN MF with sigma=0.5 to the
# input skew LN MF with alpha=0

if "__main__" == __name__:
    fig, ax = plt.subplots(figsize=(12,6))
    
    sigma = 0.5
    params_SLN = [sigma, 0.]
    
    constraints_extended_Carr_SLN = constraints_Carr_general(skew_LN, params_SLN)
    constraints_extended_Carr_LN = constraints_Carr_general(lognormal_number_density, [sigma])
    
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
    #plt.savefig("./Figures/GC_constraints/test_LN_SLN.png")


#%% 
# Compute constraints for the fitting functions from 2009.03204.

if "__main__" == __name__:

    # Extended mass function constraints using the method from 1705.05567.
    # Choose which constraints to plot, and create labels
    constraints_labels = []
    constraints_names = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]

    # Mass function parameter values, from 2009.03204.
    Deltas = np.array([0., 0.1, 0.3, 0.5, 1.0, 2.0, 5.0])
    sigmas_SLN = np.array([0.55, 0.55, 0.57, 0.60, 0.71, 0.97, 2.77])
    alphas_SL = np.array([-2.27, -2.24, -2.07, -1.82, -1.31, -0.66, 1.39])

    alphas_CC = np.array([3.06, 3.09, 3.34, 3.82, 5.76, 18.9, 13.9])
    betas = np.array([2.12, 2.08, 1.72, 1.27, 0.51, 0.0669, 0.0206])
    
    # Log-normal parameter values, from 2008.02389
    sigmas_LN = np.array([0.374, 0.377, 0.395, 0.430, 0.553, 0.864, 0.01])

    # Colours corresponding to each instrument
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for k in range(len(Deltas)):
        # Calculate constraints for extended MF from Galactic gamma-rays.
        
        params_SLN = [sigmas_SLN[k], alphas_SL[k]]
        params_CC3 = [alphas_CC[k], betas[k]]
        params_LN = [sigmas_LN[k]]

        constraints_extended_Carr_SLN = constraints_Carr_general(skew_LN, params_SLN)
        constraints_extended_Carr_CC3 = constraints_Carr_general(CC3, params_CC3)
        constraints_extended_Carr_LN = constraints_Carr_general(lognormal_number_density, params_LN)

        
        """
        # For now, include if statement since the Delta-function power spectrum
        # peak is the only one that I have calculated constraints from Isatis
        # for so far.
        if Deltas[k] == 0:
            constraints_Isatis_SLN, constraints_names = isatis_constraints_general(skew_LN, k)
            constraints_Isatis_CC3, constraints_names = isatis_constraints_general(CC3, k)
        """
        
        # Envelope of constraints, with the tightest constraint
        envelope_SLN = []
        envelope_CC3 = []
        envelope_LN = []

        for j in range(len(masses)):
            constraints_SLN = []
            constraints_CC3 = []
            constraints_LN = []
            
            for l in range(len(constraints_names)):
                constraints_SLN.append(constraints_extended_Carr_SLN[l][j])
                constraints_CC3.append(constraints_extended_Carr_CC3[l][j])
                constraints_LN.append(constraints_extended_Carr_LN[l][j])
                
            envelope_SLN.append(min(constraints_SLN))
            envelope_CC3.append(min(constraints_CC3))
            envelope_LN.append(min(constraints_LN))
            
        # Save constraints
        for i in range(len(constraints_names)):
            data_filename_SLN = "./Data_files/constraints_extended_MF/SLN_GC_" + str(constraints_names[i]) + "_Carr_Delta={:.1f}".format(Deltas[k])
            data_filename_CC3 = "./Data_files/constraints_extended_MF/CC3_GC_" + str(constraints_names[i]) + "_Carr_Delta={:.1f}".format(Deltas[k])
            data_filename_LN = "./Data_files/constraints_extended_MF/LN_GC_" + str(constraints_names[i]) + "_Carr_Delta={:.1f}".format(Deltas[k])
            
            np.savetxt(data_filename_SLN, [masses, constraints_extended_Carr_SLN[i]], delimiter="\t")
            np.savetxt(data_filename_CC3, [masses, constraints_extended_Carr_CC3[i]], delimiter="\t")
            np.savetxt(data_filename_LN, [masses, constraints_extended_Carr_LN[i]], delimiter="\t")
        
        data_filename_SLN = "./Data_files/constraints_extended_MF/SLN_GC_envelope_Carr_Delta={:.1f}".format(Deltas[k])
        data_filename_CC3 = "./Data_files/constraints_extended_MF/CC3_GC_envelope_Carr_Delta={:.1f}".format(Deltas[k])
        data_filename_LN = "./Data_files/constraints_extended_MF/LN_GC_envelope_Carr_Delta={:.1f}".format(Deltas[k])
        
        np.savetxt(data_filename_SLN, np.array([masses, envelope_SLN]), fmt="%s", delimiter="\t")
        np.savetxt(data_filename_CC3, np.array([masses, envelope_CC3]), fmt="%s", delimiter="\t")       
        np.savetxt(data_filename_LN, np.array([masses, envelope_LN]), fmt="%s", delimiter="\t")       
