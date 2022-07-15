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