#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 13:54:33 2022

@author: ppxmg2
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from reproduce_COMPTEL_constraints_v2 import load_data, read_col, read_blackhawk_spectra, string_scientific
from J_factor_A22 import j_avg

# Script to understand the meaning of the 'refined flux' values used in Isatis

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


file_path_extracted = './Extracted_files/A22_COMPTEL/'

#%%

def refined_energies(energies, n_refined):
    E_min = min(energies)
    E_max = max(energies)
    step = (np.log10(10*E_max) - np.log10(E_min/10)) / (n_refined - 1)
    
    ener_refined = []
    for i in range(0, n_refined):
        ener_refined.append( 10**(np.log10(E_min/10) + i*step) )
    return ener_refined

def refined_flux(flux, ener_spec, n_refined):
    ener_refined = refined_energies(energies, n_refined)
    
    nb_spec = len(ener_spec)
    
    flux_refined = []
    
    c = 0
    for i in range(n_refined):
        while c < nb_spec-1 and ener_spec[c] < ener_refined[i]:   # modified: has c < nb_ener - 1 instead of c < nb_ener
            c += 1
        if c > 0 and c < nb_spec and flux[c-1] !=0:
            y = np.log10(flux[c-1]) + ((np.log10(flux[c]) - np.log10(flux[c-1])) / (np.log10(ener_spec[c]) - np.log10(ener_spec[c-1]))) * (np.log10(ener_refined[i]) - np.log10(ener_spec[c-1]))
            flux_refined.append(10**y)
        else: 
            flux_refined.append(0)
            
    return flux_refined


def galactic(spectrum):
    n_spec = len(spectrum)
    
    # Calculate J-factor
    b_max_Auffinger, l_max_Auffinger = np.radians(15), np.radians(30)
    j_factor = 2 * j_avg(b_max_Auffinger, l_max_Auffinger)
    
    galactic = []
    for i in range(n_spec):
        val = j_factor[0] * spectrum[i] / (4*np.pi*m_pbh)
        galactic.append(val)
        
    return galactic

m_pbh = 1e15

# Load photon spectra from BlackHawk outputs
exponent = np.floor(np.log10(m_pbh))
coefficient = m_pbh / 10**exponent

file_path_data = "../blackhawk_v2.0/results/A22_Fig3_" + "{:.1f}e{:.0f}g/".format(coefficient, exponent)
print("{:.1f}e{:.0f}g/".format(coefficient, exponent))

ener_spec, spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_secondary_spectra.txt", col=1)

# COMPTEL data for flux values
flux_minus = np.array([5.40770e-01, 7.80073e-02, 7.83239e-03])
flux_plus = np.array([5.21580e-01, 7.35839e-02, 7.78441e-03])
flux = np.array([2.38601e+00, 3.44189e-01, 3.40997e-02])

energies = np.array([1.23204e-03, 4.47746e-03, 1.26645e-02])

n_refined = 500

flux_galactic = galactic(spectrum)
ener_refined = refined_energies(energies, n_refined)
flux_refined = refined_flux(flux_galactic, ener_spec, n_refined)
print(flux_refined)

plt.figure()
plt.plot(ener_spec, flux_galactic, 'x', label = 'Original')
plt.plot(ener_refined, flux_refined, 'x', label='Refined')
plt.title('$M_\mathrm{PBH}$ = ' + "{:.1f}e{:.0f}".format(coefficient, exponent) + 'g')
plt.xlabel('Energy [GeV]')
plt.ylabel('${\\rm d}\Phi/{\\rm d}E\,\, ({\\rm GeV^{-1}} \cdot {\\rm s}^{-1}\cdot{\\rm cm}^{-2} \cdot {\\rm sr}^{-1})$')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.tight_layout()
