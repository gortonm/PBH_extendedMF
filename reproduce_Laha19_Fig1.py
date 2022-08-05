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

# Script to reproduce Fig. 1 of Laha '19 (1906.09994), using results from
# BlackHawk 

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
r_s = 3.5    # scale radius, in kpc (Ng+ '14)
# prefactor for isothermal density profile
prefactor_Iso = 6.3e50 * (s_to_yr) / (4 * np.pi * rho_0 * (r_s**2 + r_odot**2))
# prefactor for NFW density profile
prefactor_NFW = 6.3e50 * (s_to_yr) / (4 * np.pi * rho_0 * (r_s + r_odot)**2)


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


# distance from galactic centre to include positrons from, in kpc
R = 1.5
m_pbh_L19_Iso_1500pc, f_pbh_L19_Iso_1500pc = load_data('Laha19_Iso_1.5kpc.csv')

f_pbh_Iso_values = []
f_pbh_NFW_values = []
m_pbh_values = []

# PBH mass (in grams)
for m_pbh in np.linspace(1, 10, 10) * 10**16:
    
    # PBH mass (in grams)
    m_pbh_values.append(m_pbh) 
    
    exponent = np.floor(np.log10(m_pbh))
    coefficient = m_pbh / 10**exponent

    file_path_data = "../BlackHawk_v2.1/results/Laha16_Fig1_" + "{:.0f}e{:.0f}g/".format(coefficient, exponent)
    
    # Load electron primary spectrum
    energies_primary, primary_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_primary_spectra.txt", col=7)
    #print(primary_spectrum)
    #print(energies_primary)
    
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
    

    # Load photon primary spectrum
    photon_energies_primary, photon_primary_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_primary_spectra.txt", col=1)    
    # Load photon secondary spectrum
    photon_energies_secondary, photon_secondary_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_secondary_spectra.txt", col=1)
    
    # Peak photon energy, from Eq. 14 of Swagat's summary of Hawking radiation calculations.
    peak_energy = 2.983e13 / m_pbh
    # Compare analytic result for the peak photon energy to the true maximum value.
    print(peak_energy / photon_energies_primary[np.argmax(photon_primary_spectrum)])
    
    plt.figure()
    plt.plot(photon_energies_primary, photon_primary_spectrum, label='Primary')
    plt.plot(photon_energies_secondary, photon_secondary_spectrum, 'x', label='Secondary')
    plt.xlim(E_min, E_max)
    plt.ylim(0, 1.1*max(photon_primary_spectrum))
    plt.xlabel('Energy E [GeV]')
    plt.ylabel('$\mathrm{d}^2 n_\gamma / (\mathrm{d}t\mathrm{d}E)$ [cm$^{-3}$ s$^{-1}$ GeV$^{-1}$]')
    
    # plot the peak energy
    plt.vlines(peak_energy, ymin=0, ymax=1.1*max(photon_primary_spectrum), color='k', linestyle='dotted')
    plt.title('$M_\mathrm{PBH}$ = ' + "{:.0f}e{:.0f}".format(coefficient, exponent) + 'g')
    plt.legend()
    plt.tight_layout()

    
    """
    plt.figure()
    plt.plot(energies_primary, primary_spectrum, label='Primary')
    plt.plot(energies_secondary, secondary_spectrum, 'x', label='Secondary')
    plt.xlim(E_min, E_max)
    plt.ylim(0, 1.1*max(primary_spectrum))
    plt.xlabel('Energy E [GeV]')
    plt.ylabel('$\mathrm{d}^2 n_e / (\mathrm{d}t\mathrm{d}E)$ [cm$^{-3}$ s$^{-1}$ GeV$^{-1}$]')
        plt.title('$M_\mathrm{PBH}$ = ' + "{:.0f}e{:.0f}".format(coefficient, exponent) + 'g')
    plt.legend()
    plt.tight_layout()
    """
    
    integral_primary = np.trapz(primary_spectrum_cutoff, energies_primary_cutoff)
    integral_secondary = np.trapz(secondary_spectrum_cutoff, energies_secondary_cutoff)
    integral = integral_primary + integral_secondary
    
    integral_primary = np.sum(primary_spectrum_cutoff[:-1] * np.diff(energies_primary_cutoff))
    
    # Isothermal density profile
    f_pbh_Iso = cm_to_kpc**3 * g_to_GeV * m_pbh * prefactor_Iso / (integral_primary * (R - r_s * np.arctan(R/r_s)))
    f_pbh_Iso_values.append(f_pbh_Iso)
    
    # NFW density profile
    f_pbh_NFW = cm_to_kpc**3 * g_to_GeV * m_pbh * prefactor_NFW / (integral_primary * (np.log(1 + (R/r_s)) - R / (R + r_s)))
    f_pbh_NFW_values.append(f_pbh_NFW)

   
plt.figure()
plt.plot(m_pbh_L19_Iso_1500pc, f_pbh_L19_Iso_1500pc)
plt.plot(m_pbh_values, f_pbh_Iso_values, 'x', label='Iso, 1.5 kpc')
plt.plot(m_pbh_values, f_pbh_NFW_values, 'x', label='NFW, 1.5 kpc')

plt.xlim(1e16, 10**(17.1))
plt.ylim(10**(-3.4), 1)
plt.plot()
plt.xlabel('$M_\mathrm{PBH}$ [g]')
plt.ylabel('$f_\mathrm{PBH}$')
plt.xscale('log')
#plt.yscale('log')
plt.legend()
plt.tight_layout()

