#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:33:43 2022
@author: Matthew Gorton
"""

# Calculate maximum allowed fraction of PBHs in dark matter, following Dasgupta,
# Laha & Ray (2020) (arxiv: 1912.01014), and compare to the results from their
# Fig. 1

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from loadBH import read_blackhawk_spectra, load_data, read_col
from tqdm import tqdm
import os

# Specify the plot style
mpl.rcParams.update({'font.size': 24, 'font.family':'serif'})
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

# number of steps in numerical integration over distance from the Galactic
# Centre
n_steps = 10000

# energy range to integrate over (in GeV)
E_min = 5.11e-4
E_max = 3e-3

# radius range (in cm)
r_min = 1e-3
R = 3.5 * kpc_to_cm
r_values = 10 ** np.linspace(np.log10(r_min), np.log10(R), n_steps)

# fraction of positrons injeted within distance R of the Galactic Centre which
# annihilate to produce the 511 keV signal.
annihilation_fraction = 0.8

rho_odot = 0.4 * GeV_to_g
r_odot = 8.5 * kpc_to_cm
r_s = 20 * kpc_to_cm

sigma = 1.0

# inferred rate of positron annihilation, {from observations of the 511 keV
# signal (in s^{-1}).
annihilation_rate = 1e50 / yr_to_s

# analytic result for integral over density profile, in Eq. 6 of 1912.01014.
density_integral = rho_odot * r_odot * (r_s + r_odot)**2 * (np.log(1 + (R / r_s)) - R / (R + r_s))

primary_only = True    # if True, use primary spectra only to calculate constraint.

# path to BlackHawk spectra
file_path_data_base = os.path.expanduser("~") + "/Downloads/blackhawk_v1.2/results/1000_steps/DLR20_LN/"

mu_pbh_values = [1e15, 3e15, 5e15, 1e16, 3e16, 5e16, 1e17, 3e17]

if sigma == 1:
    mu_pbh_2 = [3e17, 5e17, 1e18, 3e18]
    mu_pbh_values = np.concatenate((mu_pbh_values, mu_pbh_2))

#%%

def f_PBH(m_pbh, positron_spec, positron_energies):
    """
    Calculate maximum allowed fraction of PBHs in dark matter on PBHs from
    positron emission and the inferred annihilation rate from 511 keV line
    measurements, following Dasgupta, Laha & Ray (2020) (arxiv: 1912.01014).

    Parameters
    ----------
    m_pbh : Float
        PBH mass (in g).
    positron_spec : Numpy array of type Float.
        Positron spectrum.
    positron_energies : Numpy array of type Float.
        Positron energies.

    Returns
    -------
    Float
        Constraint on fraction of PBH in dark matter.

    """
    # following 1912.01014, do not include positrons with energies larger than
    # E_max.
    spec_integrand_temp = positron_spec[positron_energies < E_max]
    energies_integrand_temp = positron_energies[positron_energies < E_max]

    spec_integrand = spec_integrand_temp[spec_integrand_temp > 0]
    energies_integrand = energies_integrand_temp[spec_integrand_temp > 0]

    spec_integral = np.trapz(spec_integrand, energies_integrand)

    print("E_min = {:.2e} GeV".format(min(energies_integrand)))
    print("E_max = {:.2e} GeV".format(max(energies_integrand)))
    print(len(energies_integrand))

    return annihilation_rate / (4 * np.pi * annihilation_fraction * spec_integral * density_integral)


def main():
    """
    Compute constraint on PBHs at the specified range of PBH masses.

    Returns
    -------
    None.

    """
    for m_pbh in tqdm(mu_pbh_values):

        print("\nM_PBH = {:.2e} g".format(m_pbh))

        exponent = np.floor(np.log10(m_pbh))
        coefficient = m_pbh / 10**exponent

        file_path_data = file_path_data_base + "sigma={:.1f}/mu={:.1f}e{:.0f}g/".format(sigma,coefficient, exponent)
        
        if sigma < 0.5:
            file_path_data = file_path_data_base + "sigma={:.2f}/mu={:.1f}e{:.0f}g/".format(sigma,coefficient, exponent)

        if primary_only:
            ep_energies_load, ep_spec_load = read_blackhawk_spectra(file_path_data + "instantaneous_primary_spectra.txt", col=7)
        else:
            ep_energies_load, ep_spec_load = read_blackhawk_spectra(file_path_data + "instantaneous_secondary_spectra.txt", col=2)
            
        # factor of half to only include positron emission by PBHs
        # (BlackHawk spectrum includes both positrons and electrons).
        positron_spec = 0.5 * np.array(ep_spec_load)

        f_pbh_values.append(f_PBH(m_pbh, positron_spec, ep_energies_load))


#%%
if __name__ == "__main__":

    if sigma < 0.5:
        file_path_extracted = "./Extracted_files/"
        m_pbh_NFW_3500pc, f_pbh_NFW_3500pc = load_data('DLR20_Fig2_a__0_newaxes_2.csv')
    else:
        file_path_extracted = "./Extracted_files/"
        m_pbh_NFW_3500pc, f_pbh_NFW_3500pc = load_data("DLR20_Fig2_LN_sigma={:.1f}.csv".format(sigma))
    

    # calculate constraint on fraction of dark matter in PBHs.
    f_pbh_values = []
    main()
    
    
    # plot the results from Fig. 1 of 1912.01014, for PBHs with zero spin.
    plt.figure(figsize=(7, 6))
    plt.plot(mu_pbh_values, f_pbh_values, 'x', linestyle='none', label='Reproduction', color='tab:blue')
    plt.plot(m_pbh_NFW_3500pc, f_pbh_NFW_3500pc, label='Fig. 2 \n (DLR (2020))', color='tab:orange')
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

    
    # calculate ratio between reproduced results and those from Fig. 1 of
    # 1912.01014.
    f_pbh_interp = np.interp(mu_pbh_values, m_pbh_NFW_3500pc, f_pbh_NFW_3500pc)
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


#%%

sigma = 0.5
m_pbh_full = np.logspace(14.5, 15.5, 1000)

def LN_MF_density(m, m_c, sigma, A=1):
    return A * np.exp(-np.log(m/m_c)**2 / (2*sigma**2)) / (np.sqrt(2*np.pi) * sigma * m**2)
    
def LN_MF_number_density(m, m_c, sigma, A=1):
    return A * np.exp(-np.log(m/m_c)**2 / (2*sigma**2)) / (np.sqrt(2*np.pi) * sigma * m)


def read_blackhawk_MF(fname):
    m = read_col(fname, first_row=4, col=0, convert=float)
    dndm = read_col(fname, first_row=4, col=1, convert=float)
    return np.array(m), np.array(dndm)

mu_pbh = 1e15
print("\n mu_PBH = {:.2e} g".format(mu_pbh))
print("\n exp(ln(mu_pbh) - sigma^2) = {:.2e} g".format(np.exp(np.log(mu_pbh) - sigma**2)))
#print("\n exp(ln(mu_pbh) - 2sigma^2) = {:.2e} g".format(np.exp(np.log(mu_pbh) - 2*sigma**2)))

exponent = np.floor(np.log10(mu_pbh))
coefficient = mu_pbh / 10**exponent

file_path_data = file_path_data_base + "sigma={:.1f}/mu={:.1f}e{:.0f}g/".format(sigma, coefficient, exponent)

if sigma >= 0.5:
    m_pbh, dndm = read_blackhawk_MF(file_path_data + "BH_spectrum.txt")
    print("{:.2e}".format(m_pbh[np.argmax(dndm)]))
    
plt.figure(figsize=(8, 7))
plt.plot(m_pbh, LN_MF_density(m_pbh, mu_pbh, sigma), label='Mass density LN')
plt.plot(m_pbh, LN_MF_number_density(m_pbh, mu_pbh, sigma), label='Number density LN')
if sigma >= 0.5:
    plt.plot(m_pbh, dndm, label='from BlackHawk')
plt.xlabel('$M_\mathrm{PBH}$ [g]')
plt.ylabel('$\mathrm{d}n / \mathrm{d}M_\mathrm{PBH} ~ [\mathrm{cm}^{-3}]$')
plt.yscale('log')
plt.xscale('log')
plt.ylim(1e-20 * max(dndm), 1e5 * max(dndm))
plt.xlim(min(m_pbh), max(m_pbh))
plt.legend(fontsize='small')
plt.title("$\mu_\mathrm{PBH} " + "= {:.1e}~$g$, ~\sigma={:.1f}$".format(mu_pbh, sigma))
plt.tight_layout()

print(dndm / LN_MF_density(m_pbh, mu_pbh, sigma))
print(dndm / LN_MF_number_density(m_pbh, mu_pbh, sigma))


plt.figure(figsize=(8, 7))
plt.plot(m_pbh, LN_MF_density(m_pbh, mu_pbh, sigma) / np.sum(LN_MF_density(m_pbh, mu_pbh, sigma)), label='Mass density LN')
plt.plot(m_pbh, LN_MF_number_density(m_pbh, mu_pbh, sigma) / np.sum(LN_MF_number_density(m_pbh, mu_pbh, sigma)), label='Number density LN')
plt.xlabel('$M_\mathrm{PBH}$ [g]')
#plt.ylabel('$\mathrm{d}n / \mathrm{d}M_\mathrm{PBH} ~ [\mathrm{cm}^{-3}]$')
plt.ylabel('$\mathrm{d}n / \mathrm{d}M_\mathrm{PBH}$ (normalised)')
plt.yscale('log')
plt.xscale('log')
plt.ylim(1e-15, 1e-1)
#plt.xlim(min(m_pbh), max(m_pbh))
#plt.xlim(1e13, 5e20)
plt.xlim(1e11, 1e18)
plt.legend(fontsize='small')
plt.title("$\mu_\mathrm{PBH} " + "= {:.1e}~$g$, ~\sigma={:.1f}$".format(mu_pbh, sigma))
plt.tight_layout()



#%% Reproduce the quantity in BH_spectrum.txt [i.e. (dn/dM)dM]

sigma = 1.0
mu_pbh = 1e15

exponent = np.floor(np.log10(mu_pbh))
coefficient = mu_pbh / 10**exponent

file_path_data = file_path_data_base + "sigma={:.1f}/mu={:.1f}e{:.0f}g/".format(sigma, coefficient, exponent)

m_pbh, dndm = read_blackhawk_MF(file_path_data + "BH_spectrum.txt")

m_min, m_max = min(m_pbh), max(m_pbh)
delta_log_m = 0.5 * (np.log10(m_max) - np.log10(m_min)) / (len(m_pbh) - 1)

def create_spec_table(m_pbh, dndm):
    spec_table = []
    
    for i in range(len(m_pbh)):
        x = np.log10(m_pbh[i]) + delta_log_m
        y = np.log10(m_pbh[i]) - delta_log_m
        spec_table.append(dndm[i] * (10**x - 10**y))
    
    return spec_table

dndm_LN_mass = LN_MF_density(m_pbh, mu_pbh, sigma, A=1)

print(create_spec_table(m_pbh, dndm_LN_mass)[10])
print(create_spec_table(m_pbh, dndm_LN_mass)[100])
print(create_spec_table(m_pbh, dndm_LN_mass)[500])