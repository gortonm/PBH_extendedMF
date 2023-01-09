#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:33:43 2022
@author: Matthew Gorton
"""

# Calculate maximum allowed fraction of PBHs in dark matter, following Dasgupta
# , Laha & Ray (2020) (arxiv: 1912.01014), and compare to the results from 
# their Fig. 1

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from loadBH import read_blackhawk_spectra, load_data
from tqdm import tqdm
import os

# Specify the plot style
mpl.rcParams.update({'font.size': 24, 'font.family': 'serif'})
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


# Express all quantities in terms of [GeV, g, cm, s]

# unit conversions
kpc_to_cm = 3.0857e21
GeV_to_g = 1.782662e-24
yr_to_s = 365.25 * 86400

# energy range to integrate over (in GeV)
E_min = 5.11e-4
E_max = 3e-3

# radius range (in cm)
r_min = 1e-3
R = 3.5 * kpc_to_cm

# fraction of positrons injeted within distance R of the Galactic Centre which
# annihilate to produce the 511 keV signal.
annihilation_fraction = 0.8

rho_odot = 0.4 * GeV_to_g
r_odot = 8.5 * kpc_to_cm
r_s = 20 * kpc_to_cm

sigma = 0.5

# inferred rate of positron annihilation, {from observations of the 511 keV
# signal (in s^{-1}).
annihilation_rate = 1e50 / yr_to_s

# analytic result for integral over density profile, in Eq. 6 of 1912.01014.
density_integral = rho_odot * r_odot * (r_s + r_odot)**2 * (np.log(1 + (R / r_s)) - R / (R + r_s))

# if True, use primary spectra only to calculate constraint.
primary_only = True

# if True, use results for f_PBH obtained using a monochromatic mass function 
# extracted from Fig. 2 of 1912.01014.
use_extracted_mono = False

# if True, use secondary emission calculated using Hazma. Otherwise, use 
# secondary emission calculated using PYTHIA.
Hazma = False

# if True, calculate ratio between computed and expected values of f_PBH.
compute_ratios = False

# if True, when using the Carr et al. method, only include values of the
# monochromatic constraint where f_PBH <= 1.
exclude_unphysical_fPBH = False

# path to BlackHawk spectra files
if not Hazma:
    file_path_data_base = os.path.expanduser("~") + "/Downloads/blackhawk_v1.2/results/1000_steps/DLR20_LN/"
else:
    file_path_data_base = os.path.expanduser("~") + "/Downloads/version_finale/results/1000_steps/DLR20_LN/"

mu_pbh_values = [1e15, 3e15, 5e15, 1e16, 3e16, 5e16, 1e17, 3e17]

if sigma == 1:
    mu_pbh_2 = [3e17, 5e17, 1e18, 3e18]
    mu_pbh_values = np.concatenate((mu_pbh_values, mu_pbh_2))


def f_PBH(m_pbh, positron_spec, positron_energies):
    """Calculate maximum fraction of PBHs in dark matter on PBHs from 511 keV line.

    Parameters
    ----------
    m_pbh : Float
        PBH mass (in g).
    positron_spec : Array-like
        Positron spectrum.
    positron_energies : Array-like
        Positron energies.

    Returns
    -------
    Float
        Constraint on fraction of dark matter in PBHs.

    """
    # following 1912.01014, do not include positrons with energies larger than
    # E_max.
    spec_integrand_temp = positron_spec[positron_energies < E_max]
    energies_integrand_temp = positron_energies[positron_energies < E_max]

    spec_integrand = spec_integrand_temp[spec_integrand_temp > 0]
    energies_integrand = energies_integrand_temp[spec_integrand_temp > 0]

    spec_integral = np.trapz(spec_integrand, energies_integrand)
    if m_pbh < 1e16:
        print('spectrum integral = {:.2e}'.format(spec_integral))

    return annihilation_rate / (4 * np.pi * annihilation_fraction * spec_integral * density_integral)


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


def f_max(m):
    """Linearly interpolate the maximum fraction of dark matter in PBHs (monochromatic mass distribution).

    Parameters
    ----------
    m : Array-like
        PBH masses (in grams).

    Returns
    -------
    Array-like
        Maximum observationally allowed fraction of dark matter in PBHs for a
        monochromatic mass distribution.

    """
    return 10**np.interp(np.log10(m), np.log10(m_DLR20_mono), np.log10(f_max_DLR20_mono))


def integrand(A, m, m_c):
    """Compute integrand appearing in Eq. 12 of 1705.05567 (for reproducing constraints with an extended mass function following 1705.05567).

    Parameters
    ----------
    A : Float.
        Amplitude of log-normal mass function.
    m : Array-like
        PBH masses (in grams).
    m_c : Float
        Critical (median) PBH mass for log-normal mass function (in grams).

    Returns
    -------
    Array-like
        Integrand appearing in Eq. 12 of 1705.05567.

    """
    return LN_MF_number_density(m, m_c, sigma, A) / f_max(m)


def main(primary_only=True):
    """Compute constraint on PBHs at the specified range of PBH masses.

    Parameters
    ----------
    primary_only : Boolean
                   If True, only use primary emission of electrons/positrons.

    Returns
    -------
    f_pbh_values : Array-like
        Fraction of dark matter in primordial black holes.
    """
    f_pbh_values = []

    for m_pbh in tqdm(mu_pbh_values):

        print(primary_only)

        print("\nM_PBH = {:.2e} g".format(m_pbh))

        exponent = np.floor(np.log10(m_pbh))
        coefficient = m_pbh / 10**exponent

        file_path_data = file_path_data_base + "sigma={:.1f}/mu={:.1f}e{:.0f}g/".format(sigma,coefficient, exponent)

        if sigma < 0.5:
            file_path_data = file_path_data_base + "sigma={:.2f}/mu={:.1f}e{:.0f}g/".format(sigma,coefficient, exponent)

        if primary_only:
            print(primary_only)
            ep_energies_load, ep_spec_load = read_blackhawk_spectra(file_path_data + "instantaneous_primary_spectra.txt", col=7)
        else:
            print(primary_only)
            ep_energies_load, ep_spec_load = read_blackhawk_spectra(file_path_data + "instantaneous_secondary_spectra.txt", col=2)

        # factor of half to only include positron emission by PBHs
        # (BlackHawk spectrum includes both positrons and electrons).
        positron_spec = 0.5 * np.array(ep_spec_load)
        f_pbh_values.append(f_PBH(m_pbh, positron_spec, ep_energies_load))

    return f_pbh_values

if __name__ == "__main__":
    if sigma < 0.5:
        file_path_extracted = "./Extracted_files/"
        mu_pbh_DLR20_LN, f_pbh_DLR20_LN = load_data("DLR20_Fig2_a__0_newaxes_2.csv")
    else:
        file_path_extracted = "./Extracted_files/"
        mu_pbh_DLR20_LN, f_pbh_DLR20_LN = load_data("DLR20_Fig2_LN_sigma={:.1f}.csv".format(sigma))
    # calculate constraint on fraction of dark matter in PBHs.
    f_pbh_values = main(primary_only)

    if compute_ratios:
        # plot the results from Fig. 1 of 1912.01014, for PBHs with zero spin.
        plt.figure(figsize=(8, 7))
        plt.plot(mu_pbh_values, f_pbh_values, 'x', linestyle='none', label='Reproduction', color='tab:blue')
        plt.plot(mu_pbh_DLR20_LN, f_pbh_DLR20_LN, label='Fig. 2 \n (DLR (2020))', color='tab:orange')
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel('$f_\mathrm{PBH}$')
        plt.xlabel('$\mu_\mathrm{PBH}$ [g]')
        plt.xlim(1e15, 1e19)
        plt.ylim(1e-4, 1)
        if sigma < 0.5:
            plt.title("Log-normal ($\sigma={:.2f}$)".format(sigma))
        else:
            plt.title("Log-normal ($\sigma={:.1f}$)".format(sigma))
        plt.legend(fontsize='small')
        plt.tight_layout()
        print(f_pbh_values)

        # calculate ratio between reproduced results and those from Fig. 1 of
        # 1912.01014.
        print("Direct calculation:")
        f_pbh_interp = np.interp(mu_pbh_values, mu_pbh_DLR20_LN, f_pbh_DLR20_LN)
        ratio = np.array(f_pbh_values / f_pbh_interp)
        print(ratio)

        # plot the fractional difference between the results
        plt.figure(figsize=(7, 6))
        plt.plot(mu_pbh_values, ratio-1, 'x', linestyle='none', color='tab:blue')
        plt.xscale('log')
        #plt.yscale('log')
        plt.ylabel('$f_\mathrm{PBH, calculated}/f_\mathrm{PBH, extracted}$ - 1')
        plt.xlabel('$\mu_\mathrm{PBH}$ [g]')
        plt.xlim(1e15, 5e18)
        if sigma < 0.5:
            plt.title("Log-normal ($\sigma={:.2f}$)".format(sigma))
        else:
            plt.title("Log-normal ($\sigma={:.1f}$)".format(sigma))
        plt.tight_layout()

    if not Hazma:
        plt.title("Log-normal ($\sigma={:.1f}$) [Pythia]".format(sigma))
        hadronisation_code = "[Pythia"
    else:
        plt.title("Log-normal ($\sigma={:.1f}$) [Hazma]".format(sigma))
        hadronisation_code = "[Hazma"
    if primary_only:
        primsec = ", primary emission only]"
    else:
        primsec = ", total emission]"

    if sigma >= 0.5:
        # plot including results calculated for a log-normal mass function 
        # using the method outlined in 1705.05567.

        mu_pbh_DLR20_Carr = 10**np.linspace(15, 19, 100)

        if use_extracted_mono:
            m_DLR20_mono, f_max_DLR20_mono = load_data('DLR20_Fig2_a__0_newaxes_2.csv')

        else:
            if primary_only:
                spec_version = 'prim'
            else:
                spec_version = 'tot'

            if Hazma:
                m_DLR20_mono, f_max_DLR20_mono = np.loadtxt('./Extracted_files/' + 'fPBH_DLR20_mono_' + spec_version + '.txt', delimiter='\t', unpack=True)
            else:
                m_DLR20_mono, f_max_DLR20_mono = np.loadtxt('./Extracted_files/' + 'fPBH_DLR20_mono_PYTHIA_' + spec_version + '.txt', delimiter='\t', unpack=True)

        if exclude_unphysical_fPBH:
            m_DLR20_mono = m_DLR20_mono[f_max_DLR20_mono <= 1]
            f_max_DLR20_mono = f_max_DLR20_mono[f_max_DLR20_mono >= 1]

        # calculate constraints for extended MF from evaporation
        f_pbh_DLR20_Carr = []

        for m_c in mu_pbh_DLR20_Carr:
            f_pbh_DLR20_Carr.append(1/np.trapz(integrand(A=1, m=m_DLR20_mono, m_c=m_c), m_DLR20_mono))

        if compute_ratios:
            f_pbh_Carr_comparison_interp = []
            for m_c in mu_pbh_values:
                f_pbh_Carr_comparison_interp.append(1/np.trapz(integrand(A=1, m=m_DLR20_mono, m_c=m_c), m_DLR20_mono))

        plt.figure(figsize=(7, 7))
        #plt.plot(mu_pbh_DLR20_LN, f_pbh_DLR20_LN, label='Extracted \n (DLR (2020))', color='tab:orange')
        plt.plot(mu_pbh_DLR20_LN, f_pbh_DLR20_LN, label='Extracted', color='tab:orange')
        #plt.plot(mu_pbh_values, f_pbh_values, 'x', linestyle='none', label='Reproduced', color='tab:blue')
        plt.plot(mu_pbh_DLR20_Carr, f_pbh_DLR20_Carr, label='Carr et al. \n method', linestyle='dashed', color='tab:green')
        plt.plot(mu_pbh_values, f_pbh_values, 'x', linestyle='none', label='BlackHawk', color='tab:blue')

        plt.xlabel('$\mu_\mathrm{PBH}~[\mathrm{g}]$')
        plt.ylabel('$f_\mathrm{PBH}$')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(fontsize='small')
        plt.xlim(1e15, 1e19)
        plt.ylim(1e-4, 1)
        plt.title("Log-normal ($\sigma={:.1f}$) ".format(sigma) + hadronisation_code + primsec, fontsize='small')
        #plt.title("Log-normal ($\sigma={:.1f}$) ".format(sigma))
        plt.tight_layout()

        # calculate ratio between reproduced results (using method from
        # 1705.05567) and those from Fig. 1 of 1912.01014.
        if compute_ratios:
            print("Carr et al. method:")
            print(mu_pbh_values)
            ratio = np.array(f_pbh_values / np.array(f_pbh_Carr_comparison_interp))
            print(ratio)

    # Plot results including primary emission only and primary+secondary emission
    f_pbh_values_prim = main(primary_only=True)
    f_pbh_values_tot = main(primary_only=False)

    plt.figure(figsize=(7.5, 6))
    plt.plot(mu_pbh_DLR20_LN, f_pbh_DLR20_LN, label='Fig. 2 \n (DLR (2020))', color='tab:orange')
    plt.plot(mu_pbh_values, f_pbh_values_prim, linestyle='dashed', label='Primary \n emission only', color='tab:blue', linewidth=3)
    plt.plot(mu_pbh_values, f_pbh_values_tot, linestyle='dotted', label='Primary \n + secondary \n emission', color='tab:red', linewidth=4)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('$f_\mathrm{PBH}$')
    plt.xlabel('$\mu_\mathrm{PBH}$ [g]')
    plt.xlim(1e15, 1e19)
    plt.ylim(1e-4, 1)
    plt.tight_layout()
    plt.legend(fontsize='small')
    plt.title("Log-normal ($\sigma={:.1f}$) ".format(sigma) + hadronisation_code + "]", fontsize='small')

#%%
# plot the results from Fig. 2 of 1912.01014, and the reproduction, for PBHs 
# with zero spin.
m_DLR20_mono, f_max_DLR20_mono = load_data('DLR20_Fig2_a__0_newaxes_2.csv')
m_DLR20_mono_calc, f_max_DLR20_mono_calc = np.loadtxt('./Extracted_files/' + 'fPBH_DLR20_mono.txt', delimiter='\t', unpack=True)

plt.figure(figsize=(7.5, 6))
plt.plot(m_DLR20_mono, f_max_DLR20_mono, label='Fig. 2 (DLR (2020))', color='tab:orange')
plt.plot(m_DLR20_mono_calc, f_max_DLR20_mono_calc, 'x', linestyle='dotted', label='Reproduction', color='tab:blue')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('$f_\mathrm{PBH}$')
plt.xlabel('$M_\mathrm{PBH}$ [g]')
plt.xlim(1e15, 1e19)
plt.ylim(1e-4, 1)
plt.tight_layout()
plt.legend(fontsize='small')

f_pbh_interp = np.interp(m_DLR20_mono, m_DLR20_mono_calc, f_max_DLR20_mono_calc)
ratio = np.array(f_pbh_interp / f_max_DLR20_mono)
print(ratio)