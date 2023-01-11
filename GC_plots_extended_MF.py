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


lognormal_MF = True
sigma = 0.5

if lognormal_MF:
    filename_append = "_lognormal_sigma={:.1f}".format(sigma)



#%% Plot extended mass function results obtained using Isatis.

# Load result from Isatis
Isatis_path = "./../Downloads/version_finale/scripts/Isatis/"
results_name = "results_photons_GC%s"%(filename_append)

constraints_file = np.genfromtxt("%s%s.txt"%(Isatis_path,results_name),dtype = "str")
constraints_names_bis = constraints_file[0,1:]
constraints = np.zeros([len(constraints_file)-1,len(constraints_file[0])-1])
for i in range(len(constraints)):
    for j in range(len(constraints[0])):
        constraints[i,j] = float(constraints_file[i+1,j+1])

mu_min = 1e14
mu_max = 1e19
masses = 10**np.arange(np.log10(mu_min), np.log10(mu_max), 1)

# choose which constraints to plot
# create labels
constraints_names = []
constraints_plotting = []
for i in range(len(constraints_names_bis)):
    print(constraints[:, i])

    if np.all(constraints[:, i] == -1.):  # only include calculated constraints
        print("all = -1")
    elif np.all(constraints[:, i] == 0.):  # only include calculated constraints
        print("all = 0")
    else:
        temp = constraints_names_bis[i].split("_")
        temp2 = ""
        for j in range(len(temp)-1):
            temp2 = "".join([temp2,temp[j],'\,\,'])
        temp2 = "".join([temp2,'\,\,[arXiv:',temp[-1],']'])
        constraints_names.append(temp2)
        constraints_plotting.append(constraints[:, i])

plt.figure(figsize=(6,6))
ax = plt.gca()
for i in range(len(constraints_names)):
    ax.plot(masses, constraints_plotting[i], marker='x', label=constraints_names[i])
    
ax.set_xlabel("$M_c$ [g]")
ax.set_ylabel("$f_\mathrm{PBH}$")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(1e-10, 1)
ax.set_xlim(mu_min, mu_max)
ax.legend(fontsize='small')
plt.title("Log-normal ($\sigma = {:.1f}$) \n (Direct Isatis calculation)".format(sigma))
plt.tight_layout()

#%% Extended mass function results using the method from 1705.05567.

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


def integrand(A, m, m_c, m_GC_mono, f_max_GC_mono):
    """Compute integrand appearing in Eq. 12 of 1705.05567 (for reproducing constraints with an extended mass function following 1705.05567).

    Parameters
    ----------
    A : Float.
        Amplitude of log-normal mass function.
    m : Array-like
        PBH masses (in grams).
    m_c : Float
        Critical (median) PBH mass for log-normal mass function (in grams).
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


# calculate constraints for extended MF from evaporation
f_pbh_GC_Carr_COMPTEL = []
f_pbh_GC_Carr_EGRET = []
f_pbh_GC_Carr_INTEGRAL = []
f_pbh_GC_Carr_FermiLAT = []

m_GC_COMPTEL, f_max_GC_COMPTEL = np.genfromtxt("./Data/fPBH_Auffinger22_all_bins_COMPTEL_1107.0200.txt", delimiter="\t", unpack=True)
m_GC_EGRET, f_max_GC_EGRET = np.genfromtxt("./Data/fPBH_Auffinger22_all_bins_EGRET_9811211.txt", delimiter="\t", unpack=True)
m_GC_INTEGRAL, f_max_GC_INTEGRAL = np.genfromtxt("./Data/fPBH_Auffinger22_all_bins_INTEGRAL_1107.0200.txt", delimiter="\t", unpack=True)
m_GC_FermiLAT, f_max_GC_FermiLAT = np.genfromtxt("./Data/fPBH_Auffinger22_all_bins_Fermi-LAT_1101.1381.txt", delimiter="\t", unpack=True)

for m_c in masses:
    f_pbh_GC_Carr_COMPTEL.append(1/np.trapz(integrand(1, m_GC_COMPTEL, m_c, m_GC_COMPTEL, f_max_GC_COMPTEL), m_GC_COMPTEL))
    f_pbh_GC_Carr_EGRET.append(1/np.trapz(integrand(1, m_GC_EGRET, m_c, m_GC_EGRET, f_max_GC_EGRET), m_GC_EGRET))
    f_pbh_GC_Carr_INTEGRAL.append(1/np.trapz(integrand(1, m_GC_INTEGRAL, m_c, m_GC_INTEGRAL, f_max_GC_INTEGRAL), m_GC_INTEGRAL))
    f_pbh_GC_Carr_FermiLAT.append(1/np.trapz(integrand(1, m_GC_FermiLAT, m_c, m_GC_FermiLAT, f_max_GC_FermiLAT), m_GC_FermiLAT))

plt.figure(figsize=(6,6))
ax = plt.gca()
ax.plot(masses, f_pbh_GC_Carr_COMPTEL, marker='x', label = "COMPTEL")
ax.plot(masses, f_pbh_GC_Carr_EGRET, marker='x', label = "EGRET")
ax.plot(masses, f_pbh_GC_Carr_FermiLAT, marker='x', label = "Fermi-LAT")
ax.plot(masses, f_pbh_GC_Carr_INTEGRAL, marker='x', label = "INTEGRAL")
ax.set_xlabel("$M_c$ [g]")
ax.set_ylabel("$f_\mathrm{PBH}$")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(1e-10, 1)
ax.set_xlim(mu_min, mu_max)
ax.legend(fontsize='small')
plt.title("Log-normal ($\sigma = {:.1f}$) \n (1705.05567 method)".format(sigma))
plt.tight_layout()


#%%

masses_mono = 10**np.arange(11, 19.05, 0.1)
results_name_mono = "results_photons_GC_mono"

constraints_mono_file = np.genfromtxt("%s%s.txt"%(Isatis_path,results_name_mono),dtype = "str")
constraints_mono_names_bis = constraints_mono_file[0,1:]
constraints_mono = np.zeros([len(constraints_mono_file)-1,len(constraints_mono_file[0])-1])
for i in range(len(constraints_mono)):
    for j in range(len(constraints_mono[0])):
        constraints_mono[i,j] = float(constraints_mono_file[i+1,j+1])

# choose which constraints to plot
# create labels
constraints_mono_names = []
constraints_extended_Carr = []
constraints_mono_plotting = []
for i in range(len(constraints_names_bis)):
    if np.all(constraints_mono[:, i] <= 0.):  # only include calculated constraints
        print("all = -1 or 0")
    else:
        temp = constraints_names_bis[i].split("_")
        temp2 = ""
        for j in range(len(temp)-1):
            temp2 = "".join([temp2,temp[j],'\,\,'])
        temp2 = "".join([temp2,'\,\,[arXiv:',temp[-1],']'])
        constraints_mono_names.append(temp2)
        constraints_mono_plotting.append(constraints_mono[:, i])
        
        #print(constraints_mono[:, i])
        
        
        # restrict range to f_max < 100 (to avoid overflow errors in the mass function calculation)
        # remove unphysical of f_max = 0, -1, or inf
        constraint_extended_Carr = []
        f_max_values = constraints_mono[:, i]
        
        f_max_truncated = f_max_values[f_max_values<1e2]
        masses_mono_truncated = masses_mono[f_max_values<1e2]
        
        masses_mono_truncated = masses_mono_truncated[f_max_truncated>0]
        f_max_truncated = f_max_truncated[f_max_truncated>0]
        
        masses_mono_truncated = masses_mono_truncated[f_max_truncated != float('inf')]
        f_max_truncated = f_max_truncated[f_max_truncated != float('inf')]
        
        for m_c in masses:
            constraint_extended_Carr.append(1/np.trapz(integrand(1, masses_mono_truncated, m_c, masses_mono_truncated, f_max_truncated), masses_mono_truncated))
        
        constraints_extended_Carr.append(constraint_extended_Carr)
        
        
# Plot the monochromatic MF constraints, as a check
plt.figure(figsize=(6,6))
ax = plt.gca()
for i in range(len(constraints_mono_names)):
    ax.plot(masses_mono, constraints_mono_plotting[i], marker='x', label=constraints_mono_names[i])
    print(constraints_mono_plotting[i])
ax.set_xlabel("$M_\mathrm{PBH}$ [g]")
ax.set_ylabel("$f_\mathrm{PBH}$")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(1e-10, 1)
ax.set_xlim(1e14, 1e18)
ax.legend(fontsize='small')
plt.title("Monochromatic MF (Isatis calculation)")
plt.tight_layout()

# Plot the log-normal mass function constraints, calculated using the method
# from 1705.05567.
plt.figure(figsize=(6,6))
ax = plt.gca()
for i in range(len(constraints_extended_Carr)):
    ax.plot(masses, constraints_extended_Carr[i], marker='x', label=constraints_mono_names[i])
ax.set_xlabel("$M_c$ [g]")
ax.set_ylabel("$f_\mathrm{PBH}$")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(1e-10, 1)
ax.set_xlim(1e14, 1e19)
ax.legend(fontsize='small')
plt.title("Log-normal ($\sigma = {:.1f}$) \n (1705.05567 method)".format(sigma))
plt.tight_layout()
