#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 11:55:35 2023

@author: ppxmg2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from preliminaries import LN, SLN, CC3
from isatis_reproduction import read_blackhawk_spectra

# Produce plots of the Subaru-HSC microlensing constraints on PBHs, for
# extended mass functions, using the method from 1705.05567.

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

#%% Understand at which mass range secondary emission of photons and electrons/positrons from PBHs becomes significant.

m_pbh_values = np.logspace(11, 21, 1000)

# Path to BlackHawk data files
file_path_BlackHawk_data = "./../../Downloads/version_finale/results/"

# Plot the primary and total photon spectrum at different BH masses.

# Choose indices to plot
indices = [401, 501, 601, 701]
colors = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200']

fig, ax = plt.subplots(figsize=(6.5, 5))

for j, i in enumerate(indices):
    m_pbh = m_pbh_values[i]
    energies_primary, spectrum_primary = read_blackhawk_spectra(file_path_BlackHawk_data + "GC_mono_{:.0f}/".format(i) + "instantaneous_primary_spectra.txt", col=1)
    energies_tot, spectrum_tot = read_blackhawk_spectra(file_path_BlackHawk_data + "GC_mono_{:.0f}/".format(i) + "instantaneous_secondary_spectra.txt", col=1)
    
    ax.plot(energies_primary, spectrum_primary, linestyle="dotted", color=colors[j])
    ax.plot(energies_tot, spectrum_tot, color=colors[j], label="{:.0e}".format(m_pbh))
    
ax.legend(title="$m~[\mathrm{g}]$", fontsize="small")
ax.set_xlabel("$E~[\mathrm{GeV}]$")
ax.set_ylabel(r"$\tilde{Q}_\gamma(E)~[\mathrm{GeV^{-1} \cdot \mathrm{cm}^{-2} \cdot \mathrm{s}^{-1} \cdot \mathrm{sr}^{-1}}]$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e-5, 5)
ax.set_ylim(1e10, 1e25)


# Plot the primary and total electron/positron spectrum at different BH masses.

# Choose indices to plot
indices = [301, 371, 401, 501]
colors = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200']

fig, ax = plt.subplots(figsize=(6.5, 5))

for j, i in enumerate(indices):
    m_pbh = m_pbh_values[i]
    energies_primary, spectrum_primary = read_blackhawk_spectra(file_path_BlackHawk_data + "GC_mono_{:.0f}/".format(i) + "instantaneous_primary_spectra.txt", col=7)
    energies_tot, spectrum_tot = read_blackhawk_spectra(file_path_BlackHawk_data + "GC_mono_{:.0f}/".format(i) + "instantaneous_secondary_spectra.txt", col=2)
    
    ax.plot(energies_primary, spectrum_primary, linestyle="dotted", color=colors[j])
    ax.plot(energies_tot, spectrum_tot, color=colors[j], label="{:.0e}".format(m_pbh))
    
ax.legend(title="$m~[\mathrm{g}]$", fontsize="small")
ax.set_xlabel("$E~[\mathrm{GeV}]$")
ax.set_ylabel(r"$\tilde{Q}_e(E)~[\mathrm{GeV^{-1} \cdot \mathrm{cm}^{-2} \cdot \mathrm{s}^{-1} \cdot \mathrm{sr}^{-1}}]$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(5.11e-4, 10)
ax.set_ylim(1e16, 1e24)


#%%
# Plot the integral of primary and total photon spectrum over energy.
integral_primary = []
integral_secondary = []

for i in range(len(m_pbh_values)):
    energies_primary, spectrum_primary = read_blackhawk_spectra(file_path_BlackHawk_data + "GC_mono_{:.0f}/".format(i+1) + "instantaneous_primary_spectra.txt", col=1)
    energies_tot, spectrum_tot = read_blackhawk_spectra(file_path_BlackHawk_data + "GC_mono_{:.0f}/".format(i+1) + "instantaneous_secondary_spectra.txt", col=1)
            
    integral_primary.append(np.trapz(spectrum_primary, energies_primary))
    integral_secondary.append(np.trapz(spectrum_tot, energies_tot))

fit_m_square = integral_primary[500] * np.power(m_pbh_values/m_pbh_values[500], -1)

fig, ax = plt.subplots(figsize=(6.5, 5))
ax.plot(m_pbh_values, integral_secondary, color=colors[0], label="Total")
ax.plot(m_pbh_values, integral_primary, linestyle="dashed", color=colors[0], label="Primary emission only")
ax.plot(m_pbh_values, fit_m_square, color=colors[1], linestyle="dotted", label="$m^{-1}$ fit")
ax.set_xlabel("$m~[\mathrm{g}]$")
ax.set_ylabel("$\mathrm{d} N_\gamma/\mathrm{d}t~[\mathrm{s}^{-1}]$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(fontsize="small")
ax.set_xlim(1e16, 1e19)
ax.set_ylim(2e15, 1e19)


# Plot the integral of energy * primary and total photon spectrum over energy.
integral_primary = []
integral_secondary = []

for i in range(len(m_pbh_values)):
    energies_primary, spectrum_primary = read_blackhawk_spectra(file_path_BlackHawk_data + "GC_mono_{:.0f}/".format(i+1) + "instantaneous_primary_spectra.txt", col=1)
    energies_tot, spectrum_tot = read_blackhawk_spectra(file_path_BlackHawk_data + "GC_mono_{:.0f}/".format(i+1) + "instantaneous_secondary_spectra.txt", col=1)
            
    integral_primary.append(np.trapz(spectrum_primary*energies_primary, energies_primary))
    integral_secondary.append(np.trapz(spectrum_tot*energies_tot, energies_tot))

fit_m_square = integral_primary[500] * np.power(m_pbh_values/m_pbh_values[500], -2)

fig, ax = plt.subplots(figsize=(6.5, 5))
ax.plot(m_pbh_values, integral_secondary, color=colors[0], label="Total")
ax.plot(m_pbh_values, integral_primary, linestyle="dashed", color=colors[0], label="Primary emission only")
ax.plot(m_pbh_values, fit_m_square, color=colors[1], linestyle="dotted", label="$m^{-2}$ fit")
ax.set_xlabel("$m~[\mathrm{g}]$")
ax.set_ylabel(r"$\int \tilde{Q}_\gamma(E) E \mathrm{d}E~[\mathrm{GeV} \cdot \mathrm{s}^{-1}]$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(fontsize="small")
ax.set_xlim(2e13, 1e19)
ax.set_ylim(1e10, 1e23)

#%%
# Plot the integral of primary and total electron spectrum over energy.
integral_primary = []
integral_secondary = []

for i in range(len(m_pbh_values)):
    energies_primary, spectrum_primary = read_blackhawk_spectra(file_path_BlackHawk_data + "GC_mono_{:.0f}/".format(i+1) + "instantaneous_primary_spectra.txt", col=7)
    energies_tot, spectrum_tot = read_blackhawk_spectra(file_path_BlackHawk_data + "GC_mono_{:.0f}/".format(i+1) + "instantaneous_secondary_spectra.txt", col=2)
            
    integral_primary.append(np.trapz(spectrum_primary, energies_primary))
    integral_secondary.append(np.trapz(spectrum_tot, energies_tot))

fit_m_square = integral_primary[500] * np.power(m_pbh_values/m_pbh_values[500], -1)

fig, ax = plt.subplots(figsize=(6.5, 5))
ax.plot(m_pbh_values, integral_secondary, color=colors[0], label="Total")
ax.plot(m_pbh_values, integral_primary, linestyle="dashed", color=colors[0], label="Primary emission only")
ax.plot(m_pbh_values, fit_m_square, color=colors[1], linestyle="dotted", label="$m^{-1}$ fit")
ax.set_xlabel("$m~[\mathrm{g}]$")
ax.set_ylabel("$\mathrm{d} N_e/\mathrm{d}t~[\mathrm{s}^{-1}]$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(fontsize="small")
ax.set_xlim(1e14, 1e17)
ax.set_ylim(1e18, 1.5e22)


# Plot the integral of energy * primary and total electron spectrum over energy.
integral_primary = []
integral_secondary = []

for i in range(len(m_pbh_values)):
    energies_primary, spectrum_primary = read_blackhawk_spectra(file_path_BlackHawk_data + "GC_mono_{:.0f}/".format(i+1) + "instantaneous_primary_spectra.txt", col=7)
    energies_tot, spectrum_tot = read_blackhawk_spectra(file_path_BlackHawk_data + "GC_mono_{:.0f}/".format(i+1) + "instantaneous_secondary_spectra.txt", col=2)
            
    integral_primary.append(np.trapz(spectrum_primary*energies_primary, energies_primary))
    integral_secondary.append(np.trapz(spectrum_tot*energies_tot, energies_tot))

fit_m_square = integral_primary[500] * np.power(m_pbh_values/m_pbh_values[500], -2)

fig, ax = plt.subplots(figsize=(6.5, 5))
ax.plot(m_pbh_values, integral_secondary, color=colors[0], label="Total")
ax.plot(m_pbh_values, integral_primary, linestyle="dashed", color=colors[0], label="Primary emission only")
ax.plot(m_pbh_values, fit_m_square, color=colors[1], linestyle="dotted", label="$m^{-2}$ fit")
ax.set_xlabel("$m~[\mathrm{g}]$")
ax.set_ylabel(r"$\int \tilde{Q}_e(E) E \mathrm{d}E~[\mathrm{GeV} \cdot \mathrm{s}^{-1}]$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(fontsize="small")
ax.set_xlim(1e14, 1e18)
ax.set_ylim(1e13, 1e22)

#%% Account for the behaviour of the mass function at small PBH masses,
# following the approximate results from 1604.05349.

M_star = 5e14   # formation mass of PBH evaporating at present, in grams
m_q = 2e14    # formation mass of PBH with a temperature above the QCD temperature (see Eq. 2.4 of 1604.05349)
alpha_evap = 4   # related to emitted number of particle degrees of freedom (see Eqs. 2.7-2.8 of 1604.05349)

q = 0.4   # ratio of m_q to m_star
M_c = np.power(1 + q**3/alpha_evap, 1/3) * M_star   # characteristic mass (see Eq. (2.18) of 1604.05349)

def mf_evap_effects(m, mf, m_c, params):
    """
    Approximate present-day PBH mass function, obtained by considering the
    effect of Hawking evaporation on the mass function evaluated at the 
    formation mass, following the approach from 1604.05349.

    Parameters
    ----------
    m : Array-like
        PBH masses (at present).
    mf : Function
        PBH mass function (at formation).
    m_c : Float
        Characteristic PBH mass (at formation).
    params : Array-like
        Parameters of the PBH mass function.

    Returns
    -------
    mf_values : Array-like
        Values of the present-day PBH mass function, evaluated at the formation
        mass.

    """
    mf_values = []
    
    for i in range(len(m)):
        if m[i] > M_star:
            mf_values.append(mf(m[i], m_c, *params))
        elif m_q < m[i] <= M_star:
            mf_values.append(np.power(m[i]/M_star, 2) * mf(M_star, m_c, *params))
        elif m[i] < m_q:
            mf_values.append(np.power(m[i]/M_star, 2) * mf(M_star, m_c, *params) / alpha_evap)
          
    return mf_values


def m_0(M):
    """
    Calculate present value of the PBH mass from the PBH mass at formation,
    using Eq. (2.18) of 1604.05349.

    Parameters
    ----------
    M : Array-like
        PBH masses (at formation).

    Returns
    -------
    m0_values : Array-like (at present)
        PBH masses (at present).

    """
    m0_values = []
    
    for i in range(len(M)):
        if M[i] > M_c:
            m0_values.append(np.power(M[i]**3 - M_star**3 + (1+(1/alpha_evap))*(q*M_star)**3, 1/3))
        elif M[i] < M_star:
            m0_values.append(0)
        else:
            m0_values.append(np.power(alpha_evap * (M[i]**3 - M_star**3), 1/3))
            
    return np.array(m0_values)
             

if "__main__" == __name__:
    
    # Test: present PBH mass against formation mass.
    # Reproduce Fig. 2 (left-hand panel) of 1604.05349.
    M_values = np.logspace(11, 21, 1000)    # PBH masses, at formation
    
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(M_values, m_0(M_values))
    ax.set_xlabel("$M~[\mathrm{g}]$")
    ax.set_ylabel("$m~[\mathrm{g}]$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e14, 1e17)
    ax.set_ylim(1e12, 1e17)
    ax.vlines(M_star, color="grey", linestyle="dotted", ymin=1e12, ymax=1e17)
    fig.tight_layout()
    
    
    # Plot effect of evaporation on PBH mass function.
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)

    mc_values = np.logspace(14, 19, 5)
    m_present = m_0(M_values)
    
    for m_c in mc_values:
    
        for i in range(len(Deltas[:-1])):
                        
            params_LN = [sigmas_LN[i]]
            params_SLN = [sigmas_SLN[i], alphas_SLN[i]]
            params_CC3 = [alphas_CC3[i], betas[i]]
            
            LN_formation = LN(M_values, m_c, sigmas_LN[i])
            SLN_formation = SLN(M_values, m_c, sigmas_SLN[i], alphas_SLN[i])
            CC3_formation = CC3(M_values, m_c, alphas_CC3[i], betas[i])
    
            LN_present = mf_evap_effects(m_present, mf=LN, m_c=m_c, params=params_LN)
            SLN_present = mf_evap_effects(m_present, mf=SLN, m_c=m_c, params=params_SLN)
            CC3_present = mf_evap_effects(m_present, mf=CC3, m_c=m_c, params=params_CC3)
            
            fig, ax = plt.subplots(figsize=(7, 5))
            ax1 = ax.twinx()
            
            ax.plot(m_present, LN_present / max(LN_formation), color="r", label="LN")
            ax.plot(M_values, LN_formation / max(LN_formation), linestyle="dotted", color="r")

            ax.plot(m_present, SLN_present / max(SLN_formation), color="b", label="SLN")
            ax.plot(M_values, SLN_formation / max(SLN_formation), linestyle="dotted", color="b")

            ax.plot(m_present, CC3_present / max(CC3_formation), color="g", label="CC3")
            ax.plot(M_values, CC3_formation / max(CC3_formation), linestyle="dotted", color="g")
            
            ax.set_xlim(m_c / 100, m_c * 100)
            ax.set_ylim(1e-6, 2)

            ax.set_xlabel("$m~[\mathrm{g}]$")
            ax.set_ylabel("$\psi / \psi_{f, \mathrm{max}}$")
            ax.legend(title=r"$M_c={:.1e}".format(m_c) + "~[\mathrm{g}]," + "~\Delta={:.1f}$".format(Deltas[i]))
            ax1.set_xlabel("$M~[\mathrm{g}]$")
            ax.set_xscale("log")
            ax.set_yscale("log")
            fig.tight_layout()
          
