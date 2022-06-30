#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:29:58 2022

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
     

def read_blackhawk_spectra(fname, col=1, convert_GeV_MeV = False):
    """Read spectra files for a particular particle species, calculated
    from BlackHawk.

    fname - file name
    col - index of column to read (i.e. which particle type to use)
        - photons: col = 1 (primary and secondary spectra)
        - 7 for electrons (primary spectra)
    convert_GeV_MeV - if True, convert units from those in terms of GeV to MeV
    """
    energies_data = read_col(fname, first_row=2, col=0, convert=float)
    spectrum_data = read_col(fname, first_row=2, col=col, convert=float)
    
    energies = []
    for i in range(2, len(energies_data)):
        energies.append(float(energies_data[i]))

    spectrum = []
    for i in range(2, len(spectrum_data)):
        spectrum.append(float(spectrum_data[i]))
        
    if convert_GeV_MeV:
        return np.array(energies) * 1e3, np.array(spectrum) / 1e3
    
    else:
        return np.array(energies), np.array(spectrum)


# returns a number as a string in standard form
def string_scientific(val):
    exponent = np.floor(np.log10(val))
    coefficient = val / 10**exponent
    return r'${:.0f} \times 10^{:d}$'.format(coefficient, int(exponent))


file_path_extracted = './Extracted_files/'
def load_data(filename):
    return np.genfromtxt(file_path_extracted+filename, delimiter=',', unpack=True)


""" Capanema, Esmaeili & Esmaili '21 Fig. 1 """
plt.figure()

for m_pbh in 10**np.array([8, 10]):
        
    exponent = np.floor(np.log10(m_pbh))
    coefficient = m_pbh / 10**exponent

    file_path_data = "../blackhawk_v2.0/results/CEE21_Fig1_" + "{:.0f}e{:.0f}g/".format(coefficient, exponent)
    
    # Load photon primary spectrum from Fig. 1 of Capanema, Esmaeili & Esmaili '21
    E_CEE_21_primary, spectrum_CEE_21_primary = load_data("Capanema+_21_Fig1_" + "{:.0f}e{:.0f}".format(coefficient, exponent) + "_primary.csv")
    plt.plot(E_CEE_21_primary, spectrum_CEE_21_primary, color='k')
    
    
    peak_energy = 2.983e13 / m_pbh
    # Compare analytic result for the peak photon energy to the true maximum value.
    print(peak_energy / E_CEE_21_primary[np.argmax(spectrum_CEE_21_primary)])

    
    # Load photon primary spectrum, calculated from BlackHawk
    energies_primary, primary_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_primary_spectra.txt", col=1)
    plt.plot(energies_primary, primary_spectrum, linestyle='dashed', label=r'$M_\mathrm{PBH}$ = ' + "{:.0f}e{:.0f} g (primary)".format(coefficient, exponent))


plt.xlim(1, 1e7)
plt.ylim(1e15, 1e33)
plt.xlabel('Energy E [GeV]')
plt.ylabel('$\mathrm{d}^2 N_\gamma / (\mathrm{d}t\mathrm{d}E)$ [s$^{-1}$ GeV$^{-1}$]')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
    


""" Arbey & Auffinger '21 (2108.02737) Fig. 5 """
plt.figure()

m_pbh = 5.3e14
        
exponent = np.floor(np.log10(m_pbh))
coefficient = m_pbh / 10**exponent

file_path_data = "../Downloads/version_finale/results/AA21_Fig5/"

# Load electron primary and total spectrum from Fig. 5 of Arbey & Auffinger '21
E_AA_21_primary, spectrum_AA_21_primary = load_data("AA21_Fig5_primary.csv")
E_AA_21_tot, spectrum_AA_21_tot = load_data("AA21_Fig5_tot.csv")

# Convert to units of GeV
plt.plot(np.array(E_AA_21_primary), np.array(spectrum_AA_21_primary), color='k', alpha=0.5, linewidth=1)
plt.plot(np.array(E_AA_21_tot), np.array(spectrum_AA_21_tot), color='k', linewidth=2, alpha=0.9)

# Load electron primary spectrum, calculated from BlackHawk
energies_primary, primary_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_primary_spectra.txt", col=7, convert_GeV_MeV=True)
plt.plot(energies_primary, primary_spectrum, linestyle='dotted', label=r'Primary', color='b', linewidth=2)
# Load electron total spectrum, calculated from BlackHawk
energies_tot, tot_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_secondary_spectra.txt", col=2, convert_GeV_MeV=True)
plt.plot(energies_tot, tot_spectrum, linestyle='dotted', label=r'Total', color='r', linewidth=3)

plt.xlim(1e-3, 1e3)
plt.ylim(1e16, 1e21)
plt.xlabel('Energy E [MeV]')
plt.ylabel('$\mathrm{d}^2 N_{e^\pm} / (\mathrm{d}t\mathrm{d}E_{e^\pm})$ [s$^{-1}$ MeV$^{-1}$]')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
    