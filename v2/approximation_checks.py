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
import os

# Produce plots of the Subaru-HSC microlensing constraints on PBHs, for
# extended mass functions, using the method from 1705.05567.

# Specify the plot style
mpl.rcParams.update({'font.size': 20,'font.family':'serif'})
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

if "__main__" == __name__:

    m_pbh_values = np.logspace(11, 21, 1000)
    
    # Path to BlackHawk data files
    file_path_BlackHawk_data = "./../../Downloads/version_finale/results/"
    
    # Plot the primary and total photon spectrum at different BH masses.
    
    # Choose indices to plot
    #indices = [401, 501, 601, 701]
    indices = [1, 101, 201, 301]
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
    fig.tight_layout()
    
    # Plot the primary and total electron/positron spectrum at different BH masses.
    
    # Choose indices to plot
    indices = [301, 371, 401, 501, 601]
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
    fig.tight_layout()


#%%
if "__main__" == __name__:

    m_pbh_values = np.logspace(11, 21, 1000)    

    # Plot the integral of primary and total photon spectrum over energy (Hazma secondary spectrum and calculations).
    integral_primary = []
    integral_secondary = []
    
    for i in range(len(m_pbh_values)):
        energies_primary, spectrum_primary = read_blackhawk_spectra(file_path_BlackHawk_data + "GC_mono_{:.0f}/".format(i+1) + "instantaneous_primary_spectra.txt", col=1)
        energies_tot, spectrum_tot = read_blackhawk_spectra(file_path_BlackHawk_data + "GC_mono_{:.0f}/".format(i+1) + "instantaneous_secondary_spectra.txt", col=1)
                
        integral_primary.append(np.trapz(spectrum_primary, energies_primary))
        integral_secondary.append(np.trapz(spectrum_tot, energies_tot))
    
    fit_inv_m = integral_primary[600] * np.power(m_pbh_values/m_pbh_values[600], -1)
    fit_m_square = integral_primary[600] * np.power(m_pbh_values/m_pbh_values[600], -2)
    
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(m_pbh_values, integral_secondary, color=colors[0], label="Total")
    ax.plot(m_pbh_values, integral_primary, linestyle="dashed", color=colors[0], label="Primary emission only")
    ax.plot(m_pbh_values, fit_inv_m, color=colors[1], linestyle="dotted", label="$m^{-1}$ fit", linewidth=2)
    ax.plot(m_pbh_values, fit_m_square, color=colors[1], linestyle="dotted", label="$m^{-2}$ fit", linewidth=2)
    ax.set_xlabel("$m~[\mathrm{g}]$")
    ax.set_ylabel("$\mathrm{d} N_\gamma/\mathrm{d}t~[\mathrm{s}^{-1}]$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize="small")
    ax.set_xlim(2e13, 1e19)
    ax.set_ylim(2e15, 1e23)
    fig.tight_layout()

    
    # Plot the integral of energy * primary and total photon spectrum over energy (Hazma secondary spectrum and calculations).
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
    fig.tight_layout()

#%%
if "__main__" == __name__:
    
    m_pbh_values = np.logspace(11, 21, 1000)

    # Plot the integral of primary and total electron spectrum over energy (Hazma secondary spectrum and calculations).
    integral_primary = []
    integral_secondary = []
    
    # Maximum electron energy to include in integration (in GeV)
    E_max = 5
    
    for i in range(len(m_pbh_values)):
        energies_primary, spectrum_primary = read_blackhawk_spectra(file_path_BlackHawk_data + "GC_mono_{:.0f}/".format(i+1) + "instantaneous_primary_spectra.txt", col=7)
        energies_tot, spectrum_tot = read_blackhawk_spectra(file_path_BlackHawk_data + "GC_mono_{:.0f}/".format(i+1) + "instantaneous_secondary_spectra.txt", col=2)
        
        energies_primary_truncated = energies_primary[energies_primary <= E_max]
        spectrum_primary_truncated = spectrum_primary[energies_primary <= E_max]
        
        energies_tot_truncated = energies_tot[energies_tot <= E_max]
        spectrum_tot_truncated = spectrum_tot[energies_tot <= E_max]
        
        integral_primary.append(np.trapz(spectrum_primary_truncated, energies_primary_truncated) / 2)
        integral_secondary.append(np.trapz(spectrum_tot_truncated, energies_tot_truncated) / 2)
    
    fit_inv_m = integral_primary[400] * np.power(m_pbh_values/m_pbh_values[400], -1)
    fit_m_square_low_m = integral_primary[400] * np.power(m_pbh_values/m_pbh_values[400], -2)
   
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(m_pbh_values, integral_secondary, color=colors[1], label="Total")
    ax.plot(m_pbh_values, integral_primary, linestyle="dashed", color=colors[1], label="Primary emission only")
    ax.plot(m_pbh_values, fit_inv_m, color=colors[0], linestyle="dotted", label="$m^{-1}$ fit", linewidth=2)
    ax.plot(m_pbh_values, fit_m_square_low_m, color=colors[2], linestyle="dotted", label="$m^{-2}$ fit", linewidth=2)
    ax.set_xlabel("$m~[\mathrm{g}]$")
    ax.set_ylabel("$\mathrm{d} N_e/\mathrm{d}t~[\mathrm{s}^{-1}]$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize="small")
    ax.set_title("Hazma secondary spectrum \n (Integration range $511~\mathrm{keV} \leq E \leq $%s$~\mathrm{GeV}$)" % E_max, fontsize="small")
    ax.set_xlim(2e13, 1e17)
    ax.set_ylim(1e18, 1e24)
    fig.tight_layout()

    
    # Plot the integral of energy * primary and total electron spectrum over energy (Hazma secondary spectrum and calculations).
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
    fig.tight_layout()


#%%
if "__main__" == __name__:

    m_pbh_values = np.logspace(11, 15, 41)

    # Plot the integral of primary and total photon spectrum over energy (PYTHIA secondary spectrum and calculations).
    integral_primary = []
    integral_secondary = []
    
    for i in range(len(m_pbh_values)):
        energies_primary, spectrum_primary = read_blackhawk_spectra(file_path_BlackHawk_data + "PYTHIA_lowmass_oldtables_{:.0f}/".format(i+1) + "instantaneous_primary_spectra.txt", col=1)
        energies_tot, spectrum_tot = read_blackhawk_spectra(file_path_BlackHawk_data + "PYTHIA_lowmass_oldtables_{:.0f}/".format(i+1) + "instantaneous_secondary_spectra.txt", col=1)

        integral_primary.append(np.trapz(spectrum_primary, energies_primary))
        integral_secondary.append(np.trapz(spectrum_tot, energies_tot))
    
    fit_m_inv = integral_secondary[20] * np.power(m_pbh_values/m_pbh_values[20], -1)
    fit_m_square = integral_secondary[20] * np.power(m_pbh_values/m_pbh_values[20], -2)
    
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(m_pbh_values, integral_secondary, color=colors[0], label="Total")
    ax.plot(m_pbh_values, integral_primary, linestyle="dashed", color=colors[0], label="Primary emission only")
    ax.plot(m_pbh_values, fit_m_inv, color=colors[1], linestyle="dotted", label="$m^{-1}$ fit")
    ax.plot(m_pbh_values, fit_m_square, color=colors[2], linestyle="dotted", label="$m^{-2}$ fit")
    ax.set_xlabel("$m~[\mathrm{g}]$")
    ax.set_ylabel("$\mathrm{d} N_\gamma/\mathrm{d}t~[\mathrm{s}^{-1}]$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize="small")
    ax.set_xlim(1e11, 1e15)
    ax.set_ylim(1e20, 1e28)
    ax.set_title("PYTHIA secondary spectrum \n (Integration range $511~\mathrm{keV} \leq E \leq 10^5~\mathrm{GeV}$)", fontsize="small")
    fig.tight_layout()

    
    # Plot the integral of energy * primary and total photon spectrum over energy (Hazma secondary spectrum and calculations).
    integral_primary = []
    integral_secondary = []
    
    for i in range(len(m_pbh_values)):
        energies_primary, spectrum_primary = read_blackhawk_spectra(file_path_BlackHawk_data + "PYTHIA_lowmass_oldtables_{:.0f}/".format(i+1) + "instantaneous_primary_spectra.txt", col=1)
        energies_tot, spectrum_tot = read_blackhawk_spectra(file_path_BlackHawk_data + "PYTHIA_lowmass_oldtables_{:.0f}/".format(i+1) + "instantaneous_secondary_spectra.txt", col=1)
                
        integral_primary.append(np.trapz(spectrum_primary*energies_primary, energies_primary))
        integral_secondary.append(np.trapz(spectrum_tot*energies_tot, energies_tot))
    
    #fit_m_square = integral_primary[500] * np.power(m_pbh_values/m_pbh_values[500], -2)
    
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(m_pbh_values, integral_secondary, color=colors[0], label="Total")
    ax.plot(m_pbh_values, integral_primary, linestyle="dashed", color=colors[0], label="Primary emission only")
    #ax.plot(m_pbh_values, fit_m_square, color=colors[1], linestyle="dotted", label="$m^{-2}$ fit")
    ax.set_xlabel("$m~[\mathrm{g}]$")
    ax.set_ylabel(r"$\int \tilde{Q}_\gamma(E) E \mathrm{d}E~[\mathrm{GeV} \cdot \mathrm{s}^{-1}]$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize="small")
    ax.set_xlim(1e11, 1e15)
    ax.set_ylim(1e18, 1e29)
    ax.set_title("PYTHIA secondary spectrum \n (Integration range $511~\mathrm{keV} \leq E \leq 10^5~\mathrm{GeV}$)", fontsize="small")
    fig.tight_layout()


#%%
if "__main__" == __name__:
    
    m_pbh_values = np.logspace(11, 15, 41)

    # Plot the integral of primary and total electron spectrum over energy (PYTHIA secondary spectrum and calculations).
    integral_primary = []
    integral_secondary = []
    
    for i in range(len(m_pbh_values)):
        energies_primary, spectrum_primary = read_blackhawk_spectra(file_path_BlackHawk_data + "PYTHIA_lowmass_oldtables_{:.0f}/".format(i+1) + "instantaneous_primary_spectra.txt", col=7)
        energies_tot, spectrum_tot = read_blackhawk_spectra(file_path_BlackHawk_data + "PYTHIA_lowmass_oldtables_{:.0f}/".format(i+1) + "instantaneous_secondary_spectra.txt", col=2)
                
        integral_primary.append(np.trapz(spectrum_primary, energies_primary) / 2)
        integral_secondary.append(np.trapz(spectrum_tot, energies_tot) / 2)
    
    fit_m_square = integral_primary[-1] * np.power(m_pbh_values/m_pbh_values[-1], -1)
    fit_m_square_low_m = integral_secondary[-1] * np.power(m_pbh_values/m_pbh_values[-1], -2)
   
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(m_pbh_values, integral_secondary, color=colors[0], label="Total")
    ax.plot(m_pbh_values, integral_primary, linestyle="dashed", color=colors[0], label="Primary emission only")
    ax.plot(m_pbh_values, fit_m_square, color=colors[1], linestyle="dotted", label="$m^{-1}$ fit")
    ax.plot(m_pbh_values, fit_m_square_low_m, color=colors[2], linestyle="dotted", label="$m^{-2}$ fit")
    ax.set_xlabel("$m~[\mathrm{g}]$")
    ax.set_ylabel("$\mathrm{d} N_e/\mathrm{d}t~[\mathrm{s}^{-1}]$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize="small")
    ax.set_title("Secondary spectrum calcualted using PYTHIA \n (Integration range $511~\mathrm{keV} \leq E \leq 10^5~\mathrm{GeV}$)", fontsize="small")
    ax.set_xlim(1e11, 1e15)
    fig.tight_layout()
    
    # Plot the integral of energy * primary and total electron spectrum over energy (PYTHIA secondary spectrum and calculations).
    integral_primary = []
    integral_secondary = []
    
    for i in range(len(m_pbh_values)):
        energies_primary, spectrum_primary = read_blackhawk_spectra(file_path_BlackHawk_data + "PYTHIA_lowmass_{:.0f}/".format(i+1) + "instantaneous_primary_spectra.txt", col=7)
        energies_tot, spectrum_tot = read_blackhawk_spectra(file_path_BlackHawk_data + "PYTHIA_lowmass_{:.0f}/".format(i+1) + "instantaneous_secondary_spectra.txt", col=2)
                
        integral_primary.append(np.trapz(spectrum_primary*energies_primary, energies_primary))
        integral_secondary.append(np.trapz(spectrum_tot*energies_tot, energies_tot))
    
    fit_m_square = integral_primary[-1] * np.power(m_pbh_values/m_pbh_values[-1], -2)
    
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(m_pbh_values, integral_secondary, color=colors[0], label="Total")
    ax.plot(m_pbh_values, integral_primary, linestyle="dashed", color=colors[0], label="Primary emission only")
    ax.plot(m_pbh_values, fit_m_square, color=colors[1], linestyle="dotted", label="$m^{-2}$ fit")
    ax.set_xlabel("$m~[\mathrm{g}]$")
    ax.set_ylabel(r"$\int \tilde{Q}_e(E) E \mathrm{d}E~[\mathrm{GeV} \cdot \mathrm{s}^{-1}]$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize="small")
    ax.set_xlim(1e11, 1e15)
    #ax.set_ylim(1e13, 1e22)
    fig.tight_layout()


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


#%% Compare the log-normal in terms of the mass distribution to a log-normal
# in terms of the number density.

if "__main__" == __name__:
    
    def psi_LN_number_density(m, m_c, sigma):
        # Distribution function for PBH energy density, when the number density follows a log-normal in the mass 
                
        return LN(m, m_c, sigma) * (m / m_c) * np.exp(-sigma**2/2)

    def phi_LN_mass_density(m, m_c, sigma, log_m_factor=5, n_steps=100000):
        # Distribution function for PBH number density, when the mass density follows a log-normal in the mass 
        
        log_m_min = np.log10(m_c) - log_m_factor*sigma
        log_m_max = np.log10(m_c) + log_m_factor*sigma
    
        m_pbh_values = np.logspace(log_m_min, log_m_max, n_steps)
        normalisation = 1 / np.trapz(LN(m_pbh_values, m_c, sigma) / m_pbh_values, m_pbh_values)
        return (LN(m, m_c, sigma) / m) * normalisation

     
    m_c = 1e20
    m_p = 1e20

    sigma = 1
    m_pbh_values = np.logspace(np.log10(m_c)-9, np.log10(m_c)+6, 1000)

    sigmas = [0.373429, 0.5, 1, 1.84859]

    # Plot of the number density 
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    ax0 = axes[0][0]
    ax1 = axes[0][1]
    ax2 = axes[1][0]
    ax3 = axes[1][1]
    ax_loop = [ax0, ax1, ax2, ax3]
    ax_x_lims = [(1e18, 5e21), (1e18, 5e21), (2e16, 3e22), (5e11, 2e24)]
    ax_y_lims = [(1e-30, 1e-19), (1e-30, 1e-19), (1e-30, 1e-19), (1e-30, 1e-18)]
   
    for i in range(len(ax_loop)):
        # Plot of the number density
        ax = ax_loop[i]
        sigma = sigmas[i]
        ax.plot(m_pbh_values, LN(m_pbh_values, m_c, sigma), label="LN in number density")
        ax.plot(m_pbh_values, phi_LN_mass_density(m_pbh_values, m_c, sigma), label="LN in mass density")
        ax.plot(m_pbh_values, 2*phi_LN_mass_density(m_pbh_values, m_c, sigma), color="tab:orange", linestyle="dotted")
        ax.plot(m_pbh_values, 0.5*phi_LN_mass_density(m_pbh_values, m_c, sigma), color="tab:orange", linestyle="dotted")
        ax.set_xlabel("$M~[\mathrm{g}]$")
        ax.set_ylabel("$\phi(M) \propto \mathrm{d}n/\mathrm{d}M$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("$\sigma={:.2f}$".format(sigma))
        ax.set_xlim(ax_x_lims[i])
        ax.set_ylim(ax_y_lims[i])
        
    ax0.legend()
    fig.suptitle("Number density distribution ($M_c={:.1e}~".format(m_c) + "\mathrm{g})$")
    fig.tight_layout()

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    ax0 = axes[0][0]
    ax1 = axes[0][1]
    ax2 = axes[1][0]
    ax3 = axes[1][1]
    ax_loop = [ax0, ax1, ax2, ax3]
    
    ax_x_lims = [(1e18, 5e21), (1e18, 5e21), (5e16, 1e23), (1e13, 1e25)]
    ax_y_lims = [(1e-30, 1e-19), (1e-30, 1e-19), (1e-30, 1e-19), (5e-30, 5e-20)]

    for i in range(len(ax_loop)):
                
        # Plot of the mass density
        ax = ax_loop[i]
        sigma = sigmas[i]
        ax.plot(m_pbh_values, psi_LN_number_density(m_pbh_values, m_c, sigma), color="tab:green", label="LN in number density")
        ax.plot(m_pbh_values, LN(m_pbh_values, m_c, sigma), color="y", label="LN in mass density")
        #ax.plot(m_pbh_values, 2*LN(m_pbh_values, m_c, sigma), color="tab:orange", linestyle="dotted")
        #ax.plot(m_pbh_values, 0.5*LN(m_pbh_values, m_c, sigma), color="tab:orange", linestyle="dotted")
        ax.set_xlabel("$M~[\mathrm{g}]$")
        ax.set_ylabel("$\psi(M) \propto M\mathrm{d}n/\mathrm{d}M$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("$\sigma={:.2f}$".format(sigma))
        ax.set_xlim(ax_x_lims[i])
        ax.set_ylim(ax_y_lims[i])

    ax0.legend()
    fig.suptitle("Mass density distribution ($M_c={:.1e}~".format(m_c) + "\mathrm{g})$")
    fig.tight_layout()


    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    ax0 = axes[0][0]
    ax1 = axes[0][1]
    ax2 = axes[1][0]
    ax3 = axes[1][1]
    ax_loop = [ax0, ax1, ax2, ax3]
    
    ax_x_lims = [(1e18, 5e21), (1e18, 5e21), (5e16, 1e23), (1e13, 1e25)]
    ax_y_lims = [(1e-30, 1e-19), (1e-30, 1e-19), (1e-30, 1e-19), (5e-30, 5e-20)]

    for i in range(len(ax_loop)):
        
        # Plot of the mass density
        ax = ax_loop[i]
        sigma = sigmas[i]
        
        m_c = m_p * np.exp(sigma**2)
        
        ax.plot(m_pbh_values, psi_LN_number_density(m_pbh_values, m_p, sigma), color="tab:green", label="LN in number density")
        ax.plot(m_pbh_values, LN(m_pbh_values, m_c, sigma), color="y", label="LN in mass density", linestyle="dotted")
        #ax.plot(m_pbh_values, 2*LN(m_pbh_values, m_c, sigma), color="tab:orange", linestyle="dotted")
        #ax.plot(m_pbh_values, 0.5*LN(m_pbh_values, m_c, sigma), color="tab:orange", linestyle="dotted")
        ax.set_xlabel("$M~[\mathrm{g}]$")
        ax.set_ylabel("$\psi(M) \propto M\mathrm{d}n/\mathrm{d}M$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("$\sigma={:.2f}$".format(sigma))
        ax.set_xlim(ax_x_lims[i])
        ax.set_ylim(ax_y_lims[i])

    ax0.legend(fontsize="small")
    fig.suptitle("Mass density distribution ($M_p={:.1e}~".format(m_p) + "\mathrm{g})$")
    fig.tight_layout()


    
    
    # Plot the ratio of the mass distributions.

    fig_ratio, axes_ratio = plt.subplots(2, 2, figsize=(9, 9))
    ax_ratio_0 = axes_ratio[0][0]
    ax_ratio_1 = axes_ratio[0][1]
    ax_ratio_2 = axes_ratio[1][0]
    ax_ratio_3 = axes_ratio[1][1]
    ax_ratio_loop = [ax_ratio_0, ax_ratio_1, ax_ratio_2, ax_ratio_3]
    ax_ratio_y_lims = [(1e-2, 1e2), (1e-2, 1e2), (1e-3, 1e4), (1e-4, 1e6)]
   
    for i in range(len(ax_ratio_loop)):
        # Plot of the number density
        ax = ax_ratio_loop[i]
        sigma = sigmas[i]
        ax.plot(m_pbh_values, abs(phi_LN_mass_density(m_pbh_values, m_c, sigma)/LN(m_pbh_values, m_c, sigma)))
        ax.set_xlabel("$M~[\mathrm{g}]$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("$\sigma={:.2f}$".format(sigma))
        ax.set_xlim(ax_x_lims[i])
        ax.set_ylim(ax_ratio_y_lims[i])
        
    fig_ratio.suptitle("LN in mass density / LN in number density in $\phi$ ($M_c={:.1e}~".format(m_c) + "\mathrm{g})$")
    fig_ratio.tight_layout()

    fig_ratio, axes_ratio = plt.subplots(2, 2, figsize=(9, 9))
    ax_ratio_0 = axes_ratio[0][0]
    ax_ratio_1 = axes_ratio[0][1]
    ax_ratio_2 = axes_ratio[1][0]
    ax_ratio_3 = axes_ratio[1][1]
    ax_ratio_loop = [ax_ratio_0, ax_ratio_1, ax_ratio_2, ax_ratio_3]
    
    ax_x_lims = [(1e18, 5e21), (1e18, 5e21), (5e16, 1e23), (1e13, 1e25)]

    for i in range(len(ax_ratio_loop)):        
        # Plot of the mass density
        ax = ax_ratio_loop[i]
        sigma = sigmas[i]
        
        print("sigma={:.1f}".format(sigma))
        
        ax.plot(m_pbh_values, abs(LN(m_pbh_values, m_c, sigma)/psi_LN_number_density(m_pbh_values, m_c, sigma)))
        ax.set_xlabel("$M~[\mathrm{g}]$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("$\sigma={:.2f}$".format(sigma))
        ax.set_xlim(ax_x_lims[i])
        ax.set_ylim(ax_ratio_y_lims[i])

    fig_ratio.suptitle("LN in mass density / LN in number density in $\psi$ ($M_c={:.1e}~".format(m_c) + "\mathrm{g})$")
    fig_ratio.tight_layout()
    
    
    
    fig_ratio, axes_ratio = plt.subplots(2, 2, figsize=(9, 9))
    ax_ratio_0 = axes_ratio[0][0]
    ax_ratio_1 = axes_ratio[0][1]
    ax_ratio_2 = axes_ratio[1][0]
    ax_ratio_3 = axes_ratio[1][1]
    ax_ratio_loop = [ax_ratio_0, ax_ratio_1, ax_ratio_2, ax_ratio_3]
    
    ax_x_lims = [(1e18, 5e21), (1e18, 5e21), (5e16, 1e23), (1e13, 1e25)]

    for i in range(len(ax_ratio_loop)):        
        # Plot of the mass density
        ax = ax_ratio_loop[i]
        sigma = sigmas[i]
        
        m_c = m_p * np.exp(sigma**2)
        
        print("sigma={:.1f}".format(sigma))
        
        ax.plot(m_pbh_values, abs(LN(m_pbh_values, m_c, sigma)/psi_LN_number_density(m_pbh_values, m_p, sigma)) - 1)
        ax.set_xlabel("$M~[\mathrm{g}]$")
        ax.set_ylabel("$|\Delta\psi /\psi|$")
        ax.set_xscale("log")
        ax.set_title("$\sigma={:.2f}$".format(sigma))
        ax.set_xlim(ax_x_lims[i])
        ax.set_ylim(-0.1, 0.1)

    fig_ratio.suptitle("(LN in mass density / LN in number density in $\psi$) - 1 ($M_p={:.1e}~".format(m_p) + "\mathrm{g})$", fontsize="small")
    fig_ratio.tight_layout()

    
    
    
    """
    # Plot both MFs with the same peak mass

    m_c = m_p * np.exp(sigma**2)
    print(m_c)
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    ax0 = axes[0][0]
    ax1 = axes[0][1]
    ax2 = axes[1][0]
    ax3 = axes[1][1]
    ax_loop = [ax0, ax1, ax2, ax3]
    ax_x_lims = [(1e18, 5e21), (1e18, 5e21), (2e16, 3e22), (5e11, 2e24)]
    ax_y_lims = [(1e-30, 1e-19), (1e-30, 1e-19), (1e-30, 1e-19), (1e-30, 1e-18)]
   
    for i in range(len(ax_loop)):
        
        sigma = sigmas[i]
        m_c = m_p * np.exp(sigma**2)
        
        if sigma < 1:
            mc_test = m_c * (1+sigma**2)
        elif sigma==1:
            mc_test = m_c * 2.7
        else:
            mc_test = m_c * 30
        # Plot of the number density
        print(m_pbh_values[np.argmax(LN(m_pbh_values, m_c, sigma))])
        print(m_pbh_values[np.argmax(phi_LN_mass_density(m_pbh_values, mc_test, sigma))])
        
        ax = ax_loop[i]
        ax.plot(m_pbh_values, LN(m_pbh_values, m_c, sigma), label="LN in number density")
        ax.plot(m_pbh_values, phi_LN_mass_density(m_pbh_values, mc_test, sigma), linestyle="dotted", linewidth=4, label="LN in mass density")
        #ax.plot(m_pbh_values, 10*phi_LN_mass_density(m_pbh_values, mc_test, sigma), color="tab:orange", linestyle="dotted")
        #ax.plot(m_pbh_values, 0.1*phi_LN_mass_density(m_pbh_values, mc_test, sigma), color="tab:orange", linestyle="dotted")
        ax.set_xlabel("$M~[\mathrm{g}]$")
        ax.set_ylabel("$\phi(M) \propto \mathrm{d}n/\mathrm{d}M$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("$\sigma={:.2f}$".format(sigma))
        ax.set_xlim(ax_x_lims[i])
        ax.set_ylim(ax_y_lims[i])
    
        
        
    ax0.legend()
    fig.suptitle("Number density distribution ($M_p={:.1e}~".format(m_p) + "\mathrm{g})$")
    fig.tight_layout()
       
    
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    ax0 = axes[0][0]
    ax1 = axes[0][1]
    ax2 = axes[1][0]
    ax3 = axes[1][1]
    ax_loop = [ax0, ax1, ax2, ax3]
    
    ax_x_lims = [(1e18, 5e21), (1e18, 5e21), (5e16, 1e23), (1e13, 1e25)]
    ax_y_lims = [(1e-30, 1e-19), (1e-30, 1e-19), (1e-30, 1e-19), (5e-30, 5e-20)]

    for i in range(len(ax_loop)):
        # Plot of the mass density
        ax = ax_loop[i]
        sigma = sigmas[i]
        
        m_c = m_p * np.exp(sigma**2)        
        mc_test = m_p
        #print(m_pbh_values[np.argmax(LN(m_pbh_values, m_c, sigma))])
        #print(m_pbh_values[np.argmax(psi_LN_number_density(m_pbh_values, mc_test, sigma))])
       
        print(mc_test)
        print(m_c)
        print(sigma)
       
        ax.plot(m_pbh_values, psi_LN_number_density(m_pbh_values, mc_test, sigma), label="LN in number density")
        ax.plot(m_pbh_values, LN(m_pbh_values, m_c, sigma), linestyle="dotted", linewidth=4, label="LN in mass density")
        #ax.plot(m_pbh_values, 10*LN(m_pbh_values, m_c, sigma), color="tab:orange", linestyle="dotted")
        #ax.plot(m_pbh_values, 0.1*LN(m_pbh_values, m_c, sigma), color="tab:orange", linestyle="dotted")
        ax.set_xlabel("$M~[\mathrm{g}]$")
        ax.set_ylabel("$\psi(M) \propto M\mathrm{d}n/\mathrm{d}M$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("$\sigma={:.2f}$".format(sigma))
        ax.set_xlim(ax_x_lims[i])
        ax.set_ylim(ax_y_lims[i])

    ax0.legend()
    fig.suptitle("Mass density distribution ($M_p={:.1e}~".format(m_p) + "\mathrm{g})$")
    fig.tight_layout()
    """
