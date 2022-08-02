#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 11:51:13 2022

@author: ppxmg2
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Script to compare the fluxes used in the calculations of Auffinger '22
# Fig. 3 (2201.01265), to the constraints from Coogan, Morrison & Profumo '21
# (2010.04797)

# Specify the plot style
mpl.rcParams.update({'font.size': 24,'font.family':'serif'})
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


file_path_extracted = './Extracted_files/COMPTEL_Esquare_spectrum/'
def load_data(filename):
    return np.genfromtxt(file_path_extracted+filename, delimiter=',', unpack=True)

# Coogan, Morrison & Profumo '21 (2010.04797) cites Essig+ '13 (1309.4091) for
# their constraints on the flux, see Fig. 1 of Essig+ '13
E_Essig13_mean, spec_Essig13_mean = load_data('COMPTEL_Essig13_mean.csv')
E_Essig13_1sigma, spec_Essig13_1sigma = load_data('COMPTEL_Essig13_upper.csv')

E_Essig13_bin_lower, a = load_data('COMPTEL_Essig13_lower_x.csv')
E_Essig13_bin_upper, a = load_data('COMPTEL_Essig13_upper_x.csv')


error_Essig13 = spec_Essig13_1sigma - spec_Essig13_mean
spec_Essig_13_2sigma = spec_Essig13_1sigma + 2*(error_Essig13)
bins_upper_Essig13 = E_Essig13_bin_upper - E_Essig13_mean
bins_lower_Essig13 = E_Essig13_mean - E_Essig13_bin_lower

# Flux constraints from Auffinger '22 Fig. 2
E_Auffinger_mean, spec_Auffinger_mean = load_data('Auffinger_Fig2_COMPTEL_mean.csv')
E_Auffinger_bin_lower, a = load_data('Auffinger_Fig2_COMPTEL_lower_x.csv')
E_Auffinger_bin_upper, a  = load_data('Auffinger_Fig2_COMPTEL_upper_x.csv')

bins_upper_Auffinger = E_Auffinger_bin_upper - E_Auffinger_mean
bins_lower_Auffinger = E_Auffinger_mean - E_Auffinger_bin_lower


# Plot comparison spectra used to constrain PBH abundance
# For Coogan, Morrison & Profumo '21, plot mean flux + 2 * error bar 
# For Auffinger '22, plot mean flux 
plt.figure(figsize=(9, 8))
plt.ylim(1e-6, 3e-2)
plt.tight_layout()
plt.errorbar(E_Auffinger_mean, spec_Auffinger_mean / E_Auffinger_mean**2, xerr=(bins_lower_Auffinger, bins_upper_Auffinger), capsize=5, marker='x', elinewidth=1, linewidth=0, label="Auffinger '22 (mean flux)")
plt.errorbar(E_Essig13_mean, spec_Essig_13_2sigma / E_Essig13_mean**2, xerr=(bins_lower_Essig13, bins_upper_Essig13), capsize=5, marker='x', elinewidth=1, linewidth=0, label="CMP '21 \n (mean flux + 2 " + r"$\times$ error bar)")
plt.legend(fontsize='small')
plt.xlabel('E [MeV]')
plt.ylabel('${\\rm d}\Phi/{\\rm d}E\,\, ({\\rm MeV^{-1}} \cdot {\\rm s}^{-1}\cdot{\\rm cm}^{-3} \cdot {\\rm sr}^{-1})$')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()


# Plot energy^2 times spectra used to constrain PBH abundance, to illustrate
# the differences more clearly.
# For Coogan, Morrison & Profumo '21, plot mean flux + 2 * error bar 
# For Auffinger '22, plot mean flux 
plt.figure(figsize=(9, 8))
plt.ylim(1e-3, 3e-2)
plt.tight_layout()
plt.errorbar(E_Auffinger_mean, spec_Auffinger_mean, xerr=(bins_lower_Auffinger, bins_upper_Auffinger), capsize=5, marker='x', elinewidth=1, linewidth=0, label="Auffinger '22 (mean flux)")
plt.errorbar(E_Essig13_mean, spec_Essig_13_2sigma, xerr=(bins_lower_Essig13, bins_upper_Essig13), capsize=5, marker='x', elinewidth=1, linewidth=0, label="CMP '21 \n (mean flux + 2 " + r"$\times$ error bar)")
plt.legend(fontsize='small')
plt.xlabel('E [MeV]')
plt.ylabel('$E^2 {\\rm d}\Phi/{\\rm d}E\,\, ({\\rm MeV} \cdot {\\rm s}^{-1}\cdot{\\rm cm}^{-3} \cdot {\\rm sr}^{-1})$')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()

#%% Find bin widths in Auffinger and CMP '21

bin_widths_Auffinger = E_Auffinger_bin_upper - E_Auffinger_bin_lower
bin_widths_Essig13 = bins_upper_Essig13 - bins_lower_Essig13

print((bin_widths_Auffinger)[-1])
print((bin_widths_Essig13)[-1])


print((bin_widths_Auffinger)[0])
print((bin_widths_Essig13)[2])

#%% Find integral over different bins

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

# PBH mass (in grams)
for m_pbh in np.array([1, 4, 10]) * 10**16:
    print("\n\nM_PBH = " + string_scientific(m_pbh) + "g:")
        
    exponent = np.floor(np.log10(m_pbh))
    coefficient = m_pbh / 10**exponent
    file_path_data = "../blackhawk_v2.0/results/A22_Fig3_" + "{:.0f}e{:.0f}g/".format(coefficient, exponent)   # v2.1
    
    # Load photon spectra from BlackHawk outputs
    energies_primary, primary_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_primary_spectra.txt", col=1)
    
    # Plot photon spectra for different PBH masses to illustrate differences
    plt.figure()
    plt.plot(energies_primary, primary_spectrum)
    plt.xlim(min(energies_primary), max(energies_primary[primary_spectrum>1e18]))
    plt.ylim(0, 1.1*max(primary_spectrum))
    plt.xlabel('Energy E [GeV]')
    plt.ylabel('$\mathrm{d}^2 n_\gamma / (\mathrm{d}t\mathrm{d}E)$ [cm$^{-3}$ s$^{-1}$ GeV$^{-1}$]')
    plt.title('$M_\mathrm{PBH}$ = ' + "{:.0f}e{:.0f}".format(coefficient, exponent) + 'g')
    plt.tight_layout()
    
    # Find value of the integral of the photon spectrum over the energy range given
    # by the energy bins in COMPTEL data
    
    # COMPTEL data used in Coogan, Morrison & Profumo (2021) (2010.04797)
    print("\nCMP '21:")
    for i in (2, 4, 8):
        print("bin " + str(i+1))
        E_min = E_Essig13_bin_lower[i] / 1e3    # convert from MeV to GeV
        E_max = E_Essig13_bin_upper[i] / 1e3    # convert from MeV to GeV

        primary_spectrum_cutoff = 1e3 * primary_spectrum[energies_primary > E_min]
        energies_primary_cutoff = energies_primary[energies_primary > E_min]

        # Load photon primary spectrum
        energies_primary_interp = 10**np.linspace(np.log10(E_min), np.log10(E_max), 100000)
        primary_spectrum_interp = np.interp(energies_primary_interp, energies_primary_cutoff, primary_spectrum_cutoff)
        integral_primary = np.trapz(primary_spectrum_interp, energies_primary_interp)
        print(integral_primary)
        
    # COMPTEL data used in Auffinger (2022) (2201.01265)
    print("\nAuffinger '22:")
    for i in range(0, 3):
        print("bin " + str(i+1))
        
        E_min = E_Auffinger_bin_lower[i] / 1e3    # convert from MeV to GeV
        E_max = E_Auffinger_bin_upper[i] / 1e3    # convert from MeV to GeV
        
        primary_spectrum_cutoff = 1e3 * primary_spectrum[energies_primary > E_min]
        energies_primary_cutoff = energies_primary[energies_primary > E_min]
        
        # Load photon primary spectrum
        energies_primary_interp = 10**np.linspace(np.log10(E_min), np.log10(E_max), 100000)
        primary_spectrum_interp = np.interp(energies_primary_interp, energies_primary_cutoff, primary_spectrum_cutoff)
        integral_primary = np.trapz(primary_spectrum_interp, energies_primary_interp)
        print(integral_primary)

