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
from loadBH import read_blackhawk_spectra, load_data
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

# inferred rate of positron annihilation, {from observations of the 511 keV
# signal (in s^{-1}).
annihilation_rate = 1e50 / yr_to_s

# analytic result for integral over density profile, in Eq. 6 of 1912.01014.
density_integral = rho_odot * r_odot * (r_s + r_odot)**2 * (np.log(1 + (R / r_s)) - R / (R + r_s))

upper_mass_range = False   # if True, computes constraint at PBH masses in the range 1e16-1e17g.
coarse_mass_range = True   # if True, computes constraint at PBH masses approximately matching those in Fig. 1 of 1912.01014.
primary_only = True    # if True, use primary spectra only to calculate constraint.

# avoid having both Booleans false, so that the rest of the code runs
if upper_mass_range == False and coarse_mass_range == False:
    coarse_mass_range = False

# path to BlackHawk spectra
file_path_data_base = os.path.expanduser("~") + "/Downloads/blackhawk_v1.2/results/1000_steps/DLR20_"

if upper_mass_range:
    m_pbh_values = np.linspace(1, 15, 15) * 10**16

elif coarse_mass_range:
    m_pbh_1 = np.linspace(1, 2, 11) * 10**15
    #m_pbh_1 = np.linspace(1, 20, 20) * 10**15
    m_pbh_2 = np.linspace(2, 10, 9) * 10**16
    #m_pbh_values = np.concatenate(([1e14, 5e14, 1.1e15, 1.2e15], m_pbh_1, m_pbh_2, [2e17]))
    m_pbh_values = np.concatenate((m_pbh_1, [2e15, 3e15], m_pbh_2, [2e17]))

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

    return annihilation_rate * m_pbh / (4 * np.pi * annihilation_fraction * spec_integral * density_integral)


def main():
    """
    Compute constraint on PBHs at the specified range of PBH masses.

    Returns
    -------
    None.

    """
    for m_pbh in tqdm(m_pbh_values):

        print("\nM_PBH = {:.2e} g".format(m_pbh))

        exponent = np.floor(np.log10(m_pbh))
        coefficient = m_pbh / 10**exponent

        file_path_data = file_path_data_base + "{:.1f}e{:.0f}g/".format(coefficient, exponent)

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

    # load results from Fig. 1 of 1912.01014, for PBHs with zero spin.
    file_path_extracted = "./Extracted_files/"
    m_pbh_NFW_3500pc, f_pbh_NFW_3500pc = load_data('DLR20_Fig2_a__0_newaxes_2.csv')

    # plot the results from Fig. 1 of 1912.01014, for PBHs with zero spin.
    plt.figure(figsize=(7, 6))
    plt.plot(m_pbh_NFW_3500pc, f_pbh_NFW_3500pc, label='Fig. 2 (DLR (2020))', color='tab:orange')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('$f_\mathrm{PBH}$')
    plt.xlabel('$M_\mathrm{PBH}$ [g]')
    plt.xlim(1e15, 1e19)
    plt.ylim(1e-4, 1)
    plt.tight_layout()

    # calculate constraint on fraction of dark matter in PBHs.
    f_pbh_values = []
    main()
    
    # calculate ratio between reproduced results and those from Fig. 1 of
    # 1912.01014.
    f_pbh_interp = np.interp(m_pbh_values, m_pbh_NFW_3500pc, f_pbh_NFW_3500pc)
    ratio = np.array(f_pbh_values / f_pbh_interp)
    print(ratio)


    # plot the results from Fig. 1 of 1912.01014, and the reproduction, for
    # PBHs with zero spin.
    plt.figure(figsize=(7, 6))
    plt.plot(m_pbh_NFW_3500pc, f_pbh_NFW_3500pc, label='Fig. 2 (DLR (2020))', color='tab:orange')
    plt.plot(m_pbh_values, f_pbh_values, 'x', linestyle='dotted', label='Reproduction', color='tab:blue')
    #plt.plot(m_pbh_values, f_pbh_interp, 'x', label='Interpolated', color='tab:red')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('$f_\mathrm{PBH}$')
    plt.xlabel('$M_\mathrm{PBH}$ [g]')
    plt.xlim(1e15, 1e19)
    plt.ylim(1e-4, 1)
    plt.tight_layout()
    plt.legend(fontsize='small')

    # plot the results from Fig. 1 of 1912.01014, and the reproduction, for
    # PBHs with zero spin.
    plt.figure(figsize=(7, 6))
    plt.plot(m_pbh_NFW_3500pc, f_pbh_NFW_3500pc, 'x', label='Fig. 2 (DLR (2020))', color='tab:orange')
    plt.plot(m_pbh_values, f_pbh_values, 'x', linestyle='dotted', label='Reproduction', color='tab:blue')
    #plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('$f_\mathrm{PBH}$')
    plt.xlabel('$M_\mathrm{PBH}$ [g]')
    plt.xlim(1e15, 2e15)
    plt.ylim(8e-4, 1e-2)
    plt.tight_layout()
    plt.legend(fontsize='small')


    plt.figure(figsize=(8, 6))
    plt.plot(m_pbh_values, ratio-1, 'x', color='tab:blue')
    plt.ylabel('$(f_\mathrm{PBH, reproduced} / f_\mathrm{PBH, extracted}) - 1$')
    plt.xlabel('$M_\mathrm{PBH}$ [g]')
    plt.xlim(1e15, 2e15)
    plt.ylim(-0.15, 0.05)
    plt.tight_layout()
