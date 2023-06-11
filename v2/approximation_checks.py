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


#%%
if "__main__" == __name__:

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
if "__main__" == __name__:

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

q = m_q / M_star   # ratio of m_q to m_star
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


def m_0(M, M_star=5e14, alpha_evap=4):
    """
    Calculate present value of the PBH mass from the PBH mass at formation,
    using Eq. (2.18) of 1604.05349.

    Parameters
    ----------
    M : Array-like
        PBH masses (at formation), in grams.
    M_star : Float, optional
        Formation mass of a PBH with lifetime equal to the age of the Universe, in grams. The default is 5e14.
    alpha_evap : Float, optional
        Factor by which the number of degrees of freedom emitted by the black hole increases when the black hole reaches a temperature equivalent to the QCD confinement scale. The default is 4.

    Returns
    -------
    m0_values : Array-like
        PBH masses (at present), in grams.

    """
    q = m_q / M_star   # ratio of m_q to m_star (m_q depends only on the BH temperature)
    M_c = np.power(1 + q**3/alpha_evap, 1/3) * M_star   # characteristic mass (see Eq. (2.18) of 1604.05349)
    m0_values = []
    
    for i in range(len(M)):
        if M[i] > M_c:
            m0_values.append(np.power(M[i]**3 - M_star**3 + (1+(1/alpha_evap))*(q*M_star)**3, 1/3))
        elif M[i] < M_star:
            m0_values.append(0)
        else:
            m0_values.append(np.power(alpha_evap * (M[i]**3 - M_star**3), 1/3))
            
    return np.array(m0_values)


def M_formation(m_values, M_star=5e14, alpha_evap=4):
    """
    Calculate mass of a PBH at formation, from an array of present-day PBH
    masses, using the inverse of Eq. (2.18) of 1604.05349.

    Parameters
    ----------
    m_values : Array-like
        PBH masses (at present), in grams.
    M_star : TYPE, optional
        Formation mass of a PBH with lifetime equal to the age of the Universe, in grams. The default is 5e14.
    alpha_evap : Float, optional
        Factor by which the number of degrees of freedom emitted by the black hole increases when the black hole reaches a temperature equivalent to the QCD confinement scale. The default is 4.

    Returns
    -------
    M_values : Array-like
        PBH masses (at formation), in grams.

    """
    q = m_q / M_star   # ratio of m_q to m_star (m_q depends only on the BH temperature)
    M_c = np.power(1 + q**3/alpha_evap, 1/3) * M_star   # characteristic mass (see Eq. (2.18) of 1604.05349)
    M_values = np.zeros(len(m_values))
    
    for i in range(len(M_values)):
        
        m = m_values[i]
        M_1 = np.power(m**3 + M_star**3 - (1-alpha_evap**(-1))*(q*M_star)**3, 1/3)
        M_2 = np.power(m**3 / alpha_evap + M_star**3, 1/3)
        
        if M_1 >= M_c:
            M_values[i] = M_1
        
        elif M_star <= M_2 <= M_c:
            M_values[i] = M_2
            
        elif M_1 < M_star and M_2 < M_star:
            M_values[i] = 0
            
        else:
            M_values[i] = np.nan
            
    return M_values
             
#%%

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
          
#%% Plot the evolution of a PBH mass

def mass_evolved_BlackHawk():
    """
    Calculates the present-day PBH mass for a range of formation masses,
    using data from BlackHawk_tot.

    Returns
    -------
    m_pbh_values_formation_calculated : Array-like
        PBH masses at formation.
    m_pbh_values_0_calculated : Array-like
        Present-day PBH masses.

    """
    m_pbh_values_formation_calculated = np.logspace(np.log10(4e14), 16, 50)
    
    # Find present-day PBH mass in terms of the formation PBH mass, calculated 
    # using BlackHawk.
    m_pbh_values_0_calculated = np.ones(len(m_pbh_values_formation_calculated))

    for i in range(len(m_pbh_values_formation_calculated)):
        
        # Load data from PBHs evolved to the present time (calculated using BlackHawk)
        destination_folder = "mass_evolution_v2" + "_{:.0f}".format(i+1)
        filename = os.path.expanduser('~') + "/Downloads/version_finale/results/" + destination_folder + "/life_evolutions.txt"
        data = np.genfromtxt(filename, delimiter="    ", skip_header=4, unpack=True, dtype='str')
        
        m = []
        t = []
        for m_value in data[2]:
            m.append(float(m_value))
        for t_value in data[0]:
            t.append(float(t_value))
        
        # Age of Universe, in seconds
        t_0 = 13.8e9 * 365.25 * 86400
        
        # PBH masses from formation time to present time
        m_pbh_values_to_present = np.array(m)[np.array(t) < t_0]
                 
        # PBH mass at present
        m_pbh_values_0_calculated[i] = m_pbh_values_to_present[-1]

    return m_pbh_values_formation_calculated, m_pbh_values_0_calculated


def mass_evolved(m_pbh_values_formation):
    """
    Estimate the present-day PBH mass (due to evaporation), given an array of 
    initial PBH masses.

    Parameters
    ----------
    m_pbh_values_formation : Array-like
        Initial PBH masses, in grams.

    Returns
    -------
    m_pbh_values_0_interp : Array-like
        Present-day values of the PBH mass.

    """
    m_pbh_values_formation_calculated, m_pbh_values_0_calculated = mass_evolved_BlackHawk()
    
    # Estimate the present-day PBH mass from formation masses given in the
    # array 'm_pbh_values_formation', using linear interpolation.
    m_pbh_values_0_interp = np.ones(len(m_pbh_values_formation))
    
    for i in range(len(m_pbh_values_0_interp)):
        
        # If a PBH forms with mass less than M_*=5e14g, it no longer exists.
        if m_pbh_values_formation[i] < 5e14:
            m_pbh_values_0_interp[i] = 0
        
        # If the formation PBH mass is larger than the maximum PBH mass
        # for which the BlackHawk calculation was performed, set the present
        # mass to the formation mass
        elif m_pbh_values_formation[i] > max(m_pbh_values_formation_calculated):
            m_pbh_values_0_interp[i] = m_pbh_values_formation[i]
        
        else:
            m_pbh_values_0_interp[i] = np.interp(m_pbh_values_formation[i], m_pbh_values_formation_calculated, m_pbh_values_0_calculated)
        
    return m_pbh_values_0_interp


def mass_formation(m_pbh_values_0):
    """
    Estimate the formation PBH mass (due to evaporation), given an array of 
    present-day PBH masses.

    Parameters
    ----------
    m_pbh_values_0 : Array-like
        Present-day PBH masses, in grams.

    Returns
    -------
    m_pbh_values_formation_interp : Array-like
        Formation values of the PBH mass.

    """    
    m_pbh_values_formation_calculated, m_pbh_values_0_calculated = mass_evolved_BlackHawk()

    # Estimate the formation mass of a PBH from its present mass, using linear
    # interpolation
    
    m_pbh_values_formation_interp = np.ones(len(m_pbh_values_0))
    
    for i in range(len(m_pbh_values_formation_interp)):
                
        # If the present PBH mass is larger than the maximum PBH mass
        # for which the BlackHawk calculation was performed, set the present
        # mass to the formation mass
        if m_pbh_values_0[i] > max(m_pbh_values_formation_calculated):
            m_pbh_values_formation_interp[i] = m_pbh_values_0[i]
        
        else:
            m_pbh_values_formation_interp[i] = np.interp(m_pbh_values_0[i], m_pbh_values_0_calculated, m_pbh_values_formation_calculated)
        
    return m_pbh_values_formation_interp
   
    

if "__main__" == __name__:

    m_pbh_values_formation = np.logspace(np.log10(4e14), 16, 50)
    m_pbh_values_0 = np.ones(len(m_pbh_values_formation))    
    
    for j in range(len(m_pbh_values_formation)):
        
        destination_folder = "mass_evolution_v2" + "_{:.0f}".format(j+1)
        filename = os.path.expanduser('~') + "/Downloads/version_finale/results/" + destination_folder + "/life_evolutions.txt"
        data = np.genfromtxt(filename, delimiter="    ", skip_header=4, unpack=True, dtype='str')
        
        print(j+1)
        print("Formation mass [g] : {:.2e}".format(m_pbh_values_formation[j]))
        m = []
        t = []
        for m_value in data[2]:
            m.append(float(m_value))
        for t_value in data[0]:
            t.append(float(t_value))
        
        # Age of Universe, in seconds
        t_0 = 13.8e9 * 365.25 * 86400
        
        # PBH masses from formation time to present time
        m_pbh_values_to_present = np.array(m)[np.array(t) < t_0]
        
        print("Maximum time t < t_0 [s] = {:.2e}".format(max(np.array(t)[np.array(t) < t_0])))
        
        # PBH mass at present
        m_pbh_values_0[j] = m_pbh_values_to_present[-1]
        
    
    # Test the method 'mass_evolution' by plotting the present PBH mass against
    # the initial PBH mass for initial massees that are not computed exactly using
    # BlackHawk.
    m_pbh_values_formation_test = np.logspace(np.log10(3e14), 17, 100)
    m_pbh_values_0_test = mass_evolved(m_pbh_values_formation_test)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(m_pbh_values_formation, m_pbh_values_formation, linestyle="dotted", color="k", label="Formation mass = Present mass")
    ax.plot(m_pbh_values_formation, m_pbh_values_0, label="BlackHawk_tot calculation")
    ax.plot(m_pbh_values_formation_test, m_pbh_values_0_test, marker="x", linestyle="None", label="Test of method mass_evolved")
    ax.plot(m_pbh_values_formation_test, m_0(m_pbh_values_formation_test), marker="+", linestyle="None", label="Analytic (CKSY '16 Eq. 2.18)")
    ax.set_xlabel("Formation mass $M_0$ [g]")
    ax.set_ylabel("Present mass $M$ [g]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(min(m_pbh_values_formation), 1e16)
    ax.set_ylim(1e-1 * min(m_pbh_values_formation), max(m_pbh_values_formation))
    ax.legend()
    fig.tight_layout()
    
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(m_pbh_values_formation, m_pbh_values_0/m_pbh_values_formation)
    ax.set_xlabel("Formation mass $m_f$ [g]")
    ax.set_ylabel("Present mass $m_0$ / Formation mass $m_f$")
    #ax.set_xscale("log")
    #ax.set_yscale("log")
    ax.set_xlim(min(m_pbh_values_formation), 1e16)
    ax.set_ylim(1e-1, 1.1)
    fig.tight_layout()
    
    m_pbh_values_formation_test = np.logspace(np.log10(3e14), 17, 100)
    m_pbh_values_0_test = mass_evolved(m_pbh_values_formation_test)
    m_pbh_values_formation_test_estimated = mass_formation(m_pbh_values_0_test)
    
    # Plot formation mass against present mass
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(m_pbh_values_0_test, m_pbh_values_0_test, linestyle="dotted", color="k", label="Formation mass = Present mass")
    ax.plot(m_pbh_values_0, m_pbh_values_formation, label="BlackHawk_tot calculation")
    ax.plot(m_pbh_values_0_test, m_pbh_values_formation_test_estimated, marker="x", linestyle="None", label="Test of method mass_formation")    
    ax.plot(m_pbh_values_0_test, M_formation(m_pbh_values_0_test), marker="+", linestyle="None", label="Analytic (CKSY '16 Eq. 2.18)")
    ax.set_ylabel("Formation mass $m_f$ [g]")
    ax.set_xlabel("Present mass $m_0$ [g]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.set_xlim(1e12, 1e16)
    ax.set_ylim(1e14, max(m_pbh_values_formation))
    fig.tight_layout()
    
    # Plot formation mass against present mass, with M_* = 1e14g
    fig, ax = plt.subplots(figsize=(6, 6))
    m_pbh_values_formation = np.logspace(13, 16, 50)
    ax.plot(m_pbh_values_formation, m_pbh_values_formation, linestyle="dotted", color="k", label="Formation mass = Present mass")
    ax.plot(m_pbh_values_formation, m_0(m_pbh_values_formation, M_star=1e14), marker="+", linestyle="None", label="Analytic (CKSY '16 Eq. 2.18)")
    ax.set_xlabel("Formation mass $m_f$ [g]")
    ax.set_ylabel("Present mass $m_0$ [g]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(min(m_pbh_values_formation), 1e16)
    ax.set_ylim(1e-1 * min(m_pbh_values_formation), max(m_pbh_values_formation))
    ax.legend()
    fig.tight_layout()


    # Consistency check of the method 'mass_formation' by checking that the 
    # values calculated for the formation mass match those used as an input
    # to 'mass_evolved'.
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(m_pbh_values_formation, m_pbh_values_formation, linestyle="dotted", color="k", label="Formation mass = Present mass")
    ax.plot(m_pbh_values_formation, m_pbh_values_0)
    ax.plot(m_pbh_values_formation_test, m_pbh_values_0_test, marker="x", linestyle="None")
    ax.plot(m_pbh_values_formation_test_estimated, m_pbh_values_0_test, marker="+", linestyle="None")
    ax.set_xlabel("Formation mass $m_f$ [g]")
    ax.set_ylabel("Present mass $m_0$ [g]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(min(m_pbh_values_formation), 1e16)
    ax.set_ylim(1e-1 * min(m_pbh_values_formation), max(m_pbh_values_formation))
    fig.tight_layout()
    
    
#%% Calculate the present-day mass function by binning the masses and evolving
# the masses to the present day using mass_evolved().

from preliminaries import m_max_SLN

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
            
    for i in range(len(Deltas)):
        
        m_pbh_values = np.logspace(14, 18, 1000)
        m_pbh_values_0 = mass_evolved(m_pbh_values)
                  
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        fig2, ax2 = plt.subplots(figsize=(7, 5))

        ymin_scaled, ymax_scaled = 1e-1, 1.1
        
        # Choose factors so that peak masses of the CC3 and SLN MF match
        # closely, at 1e20g (consider this range since constraints plots)
        # indicate the constraints from the SLN and CC3 MFs are quite
        # different at this peak mass.
        
        if Deltas[i] < 5:
            m_c = 2.5e14*np.exp(ln_mc_SLN[i])
            m_p = 2.5e14*mp_CC3[i]
            mp_numeric = m_p
        else:
            m_c = 3.1e14*np.exp(ln_mc_SLN[i])
            m_p = 2.9e14*mp_CC3[i]
            mp_numeric = m_p

        #m_c = 5.6e20*np.exp(ln_mc_SLN[i])
        #m_p = 5.25e20*mp_CC3[i]

        mp_SLN_est = m_max_SLN(m_c, sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=4, n_steps=1000)
        print("m_p (CC3) = {:.2e}".format(m_p))
        print("m_p (SLN) = {:.2e}".format(mp_SLN_est))
        print("m_p (numeric) = {:.2e}".format(mp_numeric))
      
        mf_SLN = SLN(m_pbh_values, m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i])
        mf_CC3 = CC3(m_pbh_values, m_p, alpha=alphas_CC3[i], beta=betas[i])
        mf_LN = LN(m_pbh_values, m_p*np.exp(sigmas_LN[i]**2), sigma=sigmas_LN[i])
        
        mf_scaled_SLN = SLN(m_pbh_values, m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i]) / max(SLN(m_pbh_values, m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i]))
        mf_scaled_CC3 = CC3(m_pbh_values, m_p, alpha=alphas_CC3[i], beta=betas[i]) / CC3(m_p, m_p, alpha=alphas_CC3[i], beta=betas[i])
        mf_scaled_LN = LN(m_pbh_values, m_p*np.exp(sigmas_LN[i]**2), sigma=sigmas_LN[i]) / max(LN(m_pbh_values, m_p*np.exp(sigmas_LN[i]**2), sigma=sigmas_LN[i]))
        
        # Approximate analytic form of the evolved mass function, from 1604.05349.
        m_pbh_values_0_analytic = m_0(m_pbh_values)    # Analytic estimate for the present PBH mass, using Eq. (2.18) of 1604.05349.
        mf_LN_analytic = mf_evap_effects(m_pbh_values_0_analytic, mf=LN, m_c=m_p*np.exp(sigmas_LN[i]**2), params=params_LN)
        mf_SLN_analytic = mf_evap_effects(m_pbh_values_0_analytic, mf=SLN, m_c=m_c, params=params_SLN)
        mf_CC3_analytic = mf_evap_effects(m_pbh_values_0_analytic, mf=CC3, m_c=m_p, params=params_CC3)

        ax1.plot(m_pbh_values, mf_scaled_SLN, color="b", label="SLN", linestyle="dotted")
        ax1.plot(m_pbh_values, mf_scaled_CC3, color="g", label="CC3", linestyle="dotted")
        ax1.plot(m_pbh_values, mf_scaled_LN, color="r", label="LN", linestyle="dotted")
        ax1.plot(m_pbh_values_0, mf_scaled_SLN, color="b", marker="x", linestyle="None")
        ax1.plot(m_pbh_values_0, mf_scaled_CC3, color="g", marker="x", linestyle="None")
        ax1.plot(m_pbh_values_0, mf_scaled_LN, color="r", marker="x", linestyle="None")
       
        ax2.plot(m_pbh_values, mf_SLN, color="b", linestyle="dotted")
        ax2.plot(m_pbh_values, mf_CC3, color="g", linestyle="dotted")
        ax2.plot(m_pbh_values, mf_LN, color="r", linestyle="dotted")
        ax2.plot(m_pbh_values_0, mf_SLN, color="b", marker="x", linestyle="None")
        ax2.plot(m_pbh_values_0, mf_CC3, color="g", marker="x", linestyle="None")
        ax2.plot(m_pbh_values_0, mf_LN, color="r", marker="x", linestyle="None")
        ax2.plot(m_pbh_values_0_analytic, mf_SLN, color="b", marker="+", linestyle="None")
        ax2.plot(m_pbh_values_0_analytic, mf_CC3, color="g", marker="+", linestyle="None")
        ax2.plot(m_pbh_values_0_analytic, mf_LN, color="r", marker="+", linestyle="None")
       
        xmin, xmax = min(m_pbh_values), max(m_pbh_values)
        
        for ax in [ax1, ax2]:
            # Show smallest PBH mass constrained by microlensing.
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid()
            ax.legend(fontsize="small")
            ax.set_xlabel("$m~[\mathrm{g}]$")
            ax.set_title("$\Delta={:.1f},~m_p={:.0e}$".format(Deltas[i], m_p) + "$~\mathrm{g}$", fontsize="small")
            ax.set_xlim(xmin, xmax)

        ax1.set_ylabel("$\psi / \psi_\mathrm{max}$")
        ax1.set_ylim(ymin_scaled, ymax_scaled)
        ax2.set_ylabel("$\psi$")
        #ax2.set_ylim(ymin, ymax)

        fig1.set_tight_layout(True)
        fig2.set_tight_layout(True)


#%% Try reproducing results from Mosbech & Picker (2022) [arXiv:2203.05743v2]

from preliminaries import load_data, LN

m_Pl = 2.176e-5    # Planck mass, in grams
t_Pl = 5.391e-44    # Planck time, in seconds
t_0 = 13.8e9 * 365.25 * 86400    # Age of Universe, in seconds


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
    #print(3 * alpha_eff_mixed(M0_values) * m_Pl**3 * (t / t_Pl) / M_values**3)
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
    
#%%    
if "__main__" == __name__:
    
    # Reproduce Fig. 2 of Mosbech & Picker (2022)
    m_pbh_values_formation = np.logspace(11, 17, 500)
    m_pbh_values_formation_to_evolve = np.concatenate((np.arange(7.4687715114e14, 7.4687715115e14, 5e2), np.arange(7.4687715115e14, 7.47e14, 5e7), np.logspace(np.log10(7.47e14), 17, 500)))
    m_pbh_values_evolved = m_pbh_evolved_MP23(m_pbh_values_formation_to_evolve, t_0)
    m_c = 1e15
    
    for sigma in [0.1, 0.5, 1, 1.5]:
        phi_initial = phi_LN(m_pbh_values_formation, m_c, sigma)
        phi_initial_to_evolve = phi_LN(m_pbh_values_formation_to_evolve, m_c, sigma)
        
        print("ratio (phi_initial / phi_initial_to_evolve) at 1e17g = {:.5f}".format(phi_initial[-1]/phi_initial_to_evolve[-1]))
        
        phi_present = phi_evolved_v2(phi_initial_to_evolve, m_pbh_values_evolved, m_pbh_values_formation_to_evolve, t_0)
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
        

#%%
if "__main__" == __name__:

    # Reproduce Fig. 3 of Mosbech & Picker (2022)
        
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
from preliminaries import load_data

if "__main__" == __name__:

    # Load data from HESS (Abramowski et al. 2016, 1603.07730)
    E_lower_y_HESS, flux_lower_y_HESS = load_data("1603.07730/1603.07730_lower_y.csv")
    E_upper_y_HESS, flux_upper_y_HESS = load_data("1603.07730/1603.07730_upper_y.csv")
    E_lower_HESS, flux_mid_HESS = load_data("1603.07730/1603.07730_x_bins.csv")
    
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
    
    f_PBH_Isatis_lower = np.array(constraints_Isatis_file_lower[-1]) / delta_Omega
    f_PBH_Isatis_upper = np.array(constraints_Isatis_file_upper[-1]) / delta_Omega
   
    
    # Constraints data for each energy bin of each instrument, calculated using isatis_reproduction.py   
    constraints_Isatis_reproduction_file_lower = np.transpose(np.genfromtxt("./Data/fPBH_GC_full_all_bins_Fermi-LAT_1512.01846_lower_monochromatic_wide.txt"))
    constraints_Isatis_reproduction_file_upper = np.transpose(np.genfromtxt("./Data/fPBH_GC_full_all_bins_Fermi-LAT_1512.01846_upper_monochromatic_wide.txt"))
     
    f_PBH_Isatis_reproduction_lower = envelope(constraints_Isatis_reproduction_file_lower) / delta_Omega
    f_PBH_Isatis_reproduction_upper = envelope(constraints_Isatis_reproduction_file_upper) / delta_Omega
       
    
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

#%% Calculate constraints for an extended MF (basic lognormal and extended lognormal)

def LN_number_density(m, m_c, sigma, log_m_factor=5, n_steps=100000):
    # Distribution function for PBH energy density, when the number density follows a log-normal in the mass 
    
    log_m_min = np.log10(m_c) - log_m_factor*sigma
    log_m_max = np.log10(m_c) + log_m_factor*sigma

    m_pbh_values = np.logspace(log_m_min, log_m_max, n_steps)
    normalisation = 1 / np.trapz(LN(m_pbh_values, m_c, sigma) * m_pbh_values, m_pbh_values)
    return LN(m, m_c, sigma) * m * normalisation

if "__main__" == __name__:
    M0_values = np.logspace(9, 18, 50)
    
    sigmas = [0.1, 0.5, 1., 1.5]
    mc_values = [3e14, 3e13, 1e12, 2e10]
    
    for i in range(len(sigmas)):
        fig, ax = plt.subplots(figsize=(6,6))
        initial_MF = LN_number_density(M0_values, mc_values[i], sigmas[i])
        
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

#%%

if "__main__" == __name__:
    # Constraints data for each energy bin of each instrument (extended MF)
    
    constraints_mono_file_lower = np.transpose(np.genfromtxt("./Data/fPBH_GC_full_all_bins_Fermi-LAT_1512.01846_lower_monochromatic_wide.txt")) * 4*np.pi / (delta_Omega)
    constraints_mono_file_upper = np.transpose(np.genfromtxt("./Data/fPBH_GC_full_all_bins_Fermi-LAT_1512.01846_upper_monochromatic_wide.txt")) * 4*np.pi / (delta_Omega)
        
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
        f_PBH_k = constraint_Carr(mc_values, m_mono=M_values_eval, f_max=constraint_energy_bin, mf=LN_number_density, params=params_LN)
        energy_bin_constraints_lower.append(f_PBH_k)

    for k in range(len(constraints_mono_file_upper)):

        # Constraint from a particular energy bin
        constraint_energy_bin = constraints_mono_file_upper[k]

        # Calculate constraint on f_PBH from each bin
        f_PBH_k = constraint_Carr(mc_values, m_mono=M_values_eval, f_max=constraint_energy_bin, mf=LN_number_density, params=params_LN)
        energy_bin_constraints_upper.append(f_PBH_k)
        
    constraint_lower.append(envelope(energy_bin_constraints_lower))
    constraint_upper.append(envelope(energy_bin_constraints_upper))
    

    # Evolved mass function
    
    # Final constraint
    constraint_lower_evolved = []
    constraint_upper_evolved = []
    
    M0_values = np.logspace(10, 18, 1000)
    M_values_evolved = m_pbh_evolved_MP23(M0_values, t_0)  # PBH masses evolved from the formation masses

    for m_c in mc_values:
                
        mf_initial = LN_number_density(M0_values, m_c, sigma)  # initial PBH distribution
        #print("Initial MF (integrated over all masses) = {:.4f}".format(np.trapz(mf_initial, M0_values)))

        # Evolved mass function
        mf_evolved = phi_evolved_v2(mf_initial, M_values_evolved, M0_values, t_0)   # evolved PBH distribution, evaluated at present masses corresponding to the formation masses in M0_values
        # Interpolate evolved mass function at the evolved masses at which the delta-function MF constraint is calculated
        mf_evolved_interp = np.interp(M_values_eval, M_values_evolved, mf_evolved)
        
        #print("Evolved MF (integrated over all masses) = {:.4f}".format(np.trapz(mf_evolved_interp, M_values_eval)))
        
        # Constraint from each energy bin
        f_PBH_energy_bin_lower = []
        for k in range(len(constraints_mono_file_lower)):
    
            # Constraint from a particular energy bin (delta function MF)
            constraint_energy_bin = constraints_mono_file_lower[k]
            
            integrand = mf_evolved_interp / constraint_energy_bin
            integral = np.trapz(integrand, M_values_eval)
            
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

            integral = np.trapz(integrand, M_values_eval)
            
            if integral == 0 or np.isnan(integral):
                f_PBH_energy_bin_upper.append(10)
            else:
                f_PBH_energy_bin_upper.append(1/integral)
            """  
            if m_c == mc_values[0] and k % 3 == 0:
                fig, ax = plt.subplots()
                ax.plot(M_values_eval, integrand)
                ax.set_xlabel("$M~[\mathrm{g}]$")
                ax.set_ylabel("Integrand")
                ax.set_xscale("log")
                ax.set_yscale("log")
                fig.tight_layout()
            """
        constraint_upper_evolved.append(min(f_PBH_energy_bin_upper))
        print(f_PBH_energy_bin_upper)
        
    print(constraint_upper_evolved)
        
    
    # Load data from Fig. 4 of Mosbech & Picker (2022)
    m_LN_lower, f_LN_lower = load_data("2203.05743/MP22_sigma_{:.1f}_LN_lower.csv".format(sigma))
    m_LN_upper, f_LN_upper = load_data("2203.05743/MP22_sigma_{:.1f}_LN_upper.csv".format(sigma))
    m_evolved_lower, f_evolved_lower = load_data("2203.05743/MP22_sigma_{:.1f}_evolved_lower.csv".format(sigma))
    m_evolved_upper, f_evolved_upper = load_data("2203.05743/MP22_sigma_{:.1f}_evolved_upper.csv".format(sigma))
   
    fig, ax = plt.subplots(figsize=(6,6))
    ax.fill_between(mc_values, constraint_lower[0], constraint_upper[0], color="tab:green", alpha=0.75)
    ax.plot(m_LN_lower, f_LN_lower, color="tab:green", linestyle="None", marker="x")   
    ax.plot(m_LN_upper, f_LN_upper, color="tab:green", linestyle="None", marker="x")

    ax.fill_between(mc_values, constraint_lower_evolved, constraint_upper_evolved, color="tab:purple", alpha=0.75)
    ax.plot(m_evolved_lower, f_evolved_lower, color="tab:purple", linestyle="None", marker="x")   
    ax.plot(m_evolved_upper, f_evolved_upper, color="tab:purple", linestyle="None", marker="x")
    
    ax.set_xlim(1e10, 1e18)
    ax.set_ylim(10**(-15), 1)
    ax.set_xlabel("$M_c~[\mathrm{g}]$")
    ax.set_ylabel("$f_\mathrm{PBH}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("$\sigma={:.1f}$".format(sigma))
    ax.legend()
    plt.tight_layout()


#%% Compare the log-normal in terms of the mass distribution to a log-normal
# in terms of the number density.

if "__main__" == __name__:
    
    def psi_LN_number_density(m, m_c, sigma, log_m_factor=5, n_steps=100000):
        # Distribution function for PBH energy density, when the number density follows a log-normal in the mass 
        
        log_m_min = np.log10(m_c) - log_m_factor*sigma
        log_m_max = np.log10(m_c) + log_m_factor*sigma
    
        m_pbh_values = np.logspace(log_m_min, log_m_max, n_steps)
        normalisation = 1 / np.trapz(LN(m_pbh_values, m_c, sigma) * m_pbh_values, m_pbh_values)
        return LN(m, m_c, sigma) * m * normalisation

    def phi_LN_mass_density(m, m_c, sigma, log_m_factor=5, n_steps=100000):
        # Distribution function for PBH number density, when the mass density follows a log-normal in the mass 
        
        log_m_min = np.log10(m_c) - log_m_factor*sigma
        log_m_max = np.log10(m_c) + log_m_factor*sigma
    
        m_pbh_values = np.logspace(log_m_min, log_m_max, n_steps)
        normalisation = 1 / np.trapz(LN(m_pbh_values, m_c, sigma) / m_pbh_values, m_pbh_values)
        return (LN(m, m_c, sigma) / m) * normalisation

     
    m_c = 1e20
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
        ax.plot(m_pbh_values, 10*phi_LN_mass_density(m_pbh_values, m_c, sigma), color="tab:orange", linestyle="dotted")
        ax.plot(m_pbh_values, 0.1*phi_LN_mass_density(m_pbh_values, m_c, sigma), color="tab:orange", linestyle="dotted")
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
        ax.plot(m_pbh_values, psi_LN_number_density(m_pbh_values, m_c, sigma), label="LN in number density")
        ax.plot(m_pbh_values, LN(m_pbh_values, m_c, sigma), label="LN in mass density")
        ax.plot(m_pbh_values, 10*LN(m_pbh_values, m_c, sigma), color="tab:orange", linestyle="dotted")
        ax.plot(m_pbh_values, 0.1*LN(m_pbh_values, m_c, sigma), color="tab:orange", linestyle="dotted")
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


    # Plot both MFs with the same peak mass
    m_p = 1e20
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
        ax.plot(m_pbh_values, phi_LN_mass_density(m_pbh_values, mc_test, sigma), label="LN in mass density")
        ax.plot(m_pbh_values, 10*phi_LN_mass_density(m_pbh_values, mc_test, sigma), color="tab:orange", linestyle="dotted")
        ax.plot(m_pbh_values, 0.1*phi_LN_mass_density(m_pbh_values, mc_test, sigma), color="tab:orange", linestyle="dotted")
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
       
        ax.plot(m_pbh_values, psi_LN_number_density(m_pbh_values, mc_test, sigma), label="LN in number density")
        ax.plot(m_pbh_values, LN(m_pbh_values, m_c, sigma), label="LN in mass density")
        ax.plot(m_pbh_values, 10*LN(m_pbh_values, m_c, sigma), color="tab:orange", linestyle="dotted")
        ax.plot(m_pbh_values, 0.1*LN(m_pbh_values, m_c, sigma), color="tab:orange", linestyle="dotted")
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
