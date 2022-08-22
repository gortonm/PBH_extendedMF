#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 15:30:09 2022
@author: ppxmg2
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

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

# Script to reproduce Fig. 2 of Dasgupta, Laha & Ray '20 (1912.01014), using 
# PBH spectra calculated using BlackHawk 

# Unit conversion factors
s_to_yr = 1 / (365.25 * 86400)   # convert 1 s to yr
cm_to_kpc = 3.2408e-22    # convert 1 cm to kpc
g_to_GeV = 5.61e23    # convert 1 gram to GeV / c^2

# Parameters
r_odot = 8.5    # galactocentric radius of Sun, in kpc (Ng+ '14)
E_min = 5.11e-4    # minimum positron energy to consider, in GeV
E_max = 3e-3     # maximum positron energy to consider, in GeV

# Parameters (isothermal density profile)
rho_0 = 0.4   # local DM density [GeV / c^2 cm^{-3}] (Ng+ '14 middle value)
r_s_Iso = 3.5    # scale radius for Isothermal density profile, in kpc (Ng+ '14)
r_s_NFW = 20    # scale radius for NFW density profile, in kpc (Ng+ '14)

# distance from Galactic Centre to include positrons from, in kpc
R = 3.5

# positron annihilation rate in the Galactic Centre, in s^{-1}
#annihilation_rate = 3.5e43   # lower bound from Siegert et al. '15
#annihilation_rate = 6e43   # upper bound from Siegert et al. '15
#annihilation_rate = 2e43   # value from Prantzos et al. '11
#annihilation_rate = 1e43   # value from Prantzos et al. '11
annihilation_rate = 1e50 * (s_to_yr)

# prefactor for isothermal density profile
prefactor_Iso = annihilation_rate / (4 * np.pi * rho_0 * (r_s_Iso**2 + r_odot**2) * (R - r_s_Iso * np.arctan(R/r_s_Iso)))
# prefactor for NFW density profile
prefactor_NFW = annihilation_rate / (4 * np.pi * rho_0 * r_odot * (r_s_NFW + r_odot)**2 * (np.log(1 + (R/r_s_NFW)) - R / (R + r_s_NFW)) )



def read_col(fname, first_row=0, col=1, convert=int, sep=None, skip_lines=1):
    """Read text files with columns separated by `sep`.
    fname - file name
    col - index of column to read
    convert - function to convert column entry with
    sep - column separator
    If sep is not specified or is None, any
    whitespace string is a separator and empty strings are
    removed from the result.
    """
    
    data = []
    
    with open(fname) as fobj:
        i=0
        for line in fobj:
            i += 1
            #print(line)
            
            if i >= first_row:
                #print(line.split(sep=sep)[col])
                data.append(line.split(sep=sep)[col])
    return data    
     
def read_blackhawk_spectra(fname, col=1):
    """Read spectra files for a particular particle species, calculated
    from BlackHawk.
    fname - file name
    col - index of column to read (i.e. which particle type to use)
        - photons: col = 1 (primary and secondary spectra)
        - 7 for electrons (primary spectra)
    """
    energies_data = read_col(fname, first_row=2, col=0, convert=float)
    spectrum_data = read_col(fname, first_row=2, col=col, convert=float)
    
    energies = []
    for i in range(2, len(energies_data)):
        energies.append(float(energies_data[i]))

    spectrum = []
    for i in range(2, len(spectrum_data)):
        spectrum.append(float(spectrum_data[i]))
        
    return np.array(energies), np.array(spectrum)


# returns a number as a string in standard form
def string_scientific(val):
    exponent = np.floor(np.log10(val))
    coefficient = val / 10**exponent
    return r'${:.0f} \times 10^{:d}$'.format(coefficient, int(exponent))

file_path_extracted = './Extracted_files/'
def load_data(filename):
    return np.genfromtxt(file_path_extracted+filename, delimiter=',', unpack=True)


# fraction of positrons produced by PBHs within a radius R of the Galactic
# Centre that annihilate
annihilation_fraction = 1

if R == 3.5:
    annihilation_fraction = 0.8
    
m_pbh_DLR20, f_pbh_DLR20 = load_data('DLR20_Fig2_a__0.csv')
    

f_pbh_Iso_values = []
f_pbh_NFW_values = []
m_pbh_values = []

# PBH mass (in grams)
m_pbh_1 = np.linspace(1, 10, 10) * 10**15
m_pbh_2 = np.linspace(1, 15, 15) * 10**16
m_pbh_values = np.concatenate((m_pbh_1, m_pbh_2))

for m_pbh in m_pbh_values:
        
    exponent = np.floor(np.log10(m_pbh))
    coefficient = m_pbh / 10**exponent
    
    if m_pbh > 1e17:
        file_path_data = "../blackhawk_v2.0/results/Laha16_Fig1_" + "{:.1f}e{:.0f}g/".format(coefficient, exponent)
    else:
        file_path_data = "../blackhawk_v2.0/results/Laha16_Fig1_" + "{:.0f}e{:.0f}g/".format(coefficient, exponent)
    
    # Load electron primary spectrum
    energies_primary, primary_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_primary_spectra.txt", col=7)
    
    # Load electron secondary spectrum
    energies_secondary, secondary_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_secondary_spectra.txt", col=2)
    
    
    # Cut off primary spectra below 511 keV (already cut-off above 3 MeV)
    primary_spectrum_cutoff = primary_spectrum[energies_primary > E_min]
    energies_primary_cutoff = energies_primary[energies_primary > E_min]

    
    # Cut off secondary spectra below 511 keV, and above 3 MeV
    secondary_spectrum_cutoff_1 = secondary_spectrum[energies_secondary < E_max]
    energies_secondary_cutoff_1 = energies_secondary[energies_secondary < E_max]
    secondary_spectrum_cutoff = secondary_spectrum_cutoff_1[energies_secondary_cutoff_1 > E_min]
    energies_secondary_cutoff = energies_secondary_cutoff_1[energies_secondary_cutoff_1 > E_min]
    
    integral_primary = np.trapz(primary_spectrum_cutoff, energies_primary_cutoff)    
    integral_secondary = np.trapz(secondary_spectrum_cutoff, energies_secondary_cutoff)
    integral = integral_secondary
    
    # Isothermal density profile
    f_pbh_Iso = cm_to_kpc**3 * g_to_GeV * m_pbh * prefactor_Iso / (annihilation_fraction * integral)
    f_pbh_Iso_values.append(f_pbh_Iso)
    
    # NFW density profile
    f_pbh_NFW = cm_to_kpc**3 * g_to_GeV * m_pbh * prefactor_NFW / (annihilation_fraction * integral)
    f_pbh_NFW_values.append(f_pbh_NFW)

   
plt.figure(figsize=(6,6))
plt.plot(m_pbh_DLR20, f_pbh_DLR20)
#plt.plot(m_pbh_values, f_pbh_Iso_values, 'x', label='Iso: \n $R = {:.1f}$ kpc, \n $r_s = {:.1f}$ kpc'.format(R, r_s_Iso))
plt.plot(m_pbh_values, f_pbh_NFW_values, 'x', label='NFW: \n $R = {:.1f}$ kpc, \n $r_s = {:.1f}$ kpc'.format(R, r_s_NFW))

plt.xlim(1e15, 10**(19))
plt.ylim(10**(-4), 1)
plt.plot()
plt.xlabel('$M_\mathrm{PBH}$ [g]')
plt.ylabel('$f_\mathrm{PBH}$')
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize='small')
plt.tight_layout()


# compare output with interpolated extracted values
loaded_data_interp = np.interp(m_pbh_values, m_pbh_DLR20, f_pbh_DLR20)
print(loaded_data_interp / f_pbh_NFW_values)