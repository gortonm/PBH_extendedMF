#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 11:42:40 2022

@author: ppxmg2
"""

import numpy as np
from reproduce_COMPTEL_constraints_v2 import read_blackhawk_spectra, load_data
from scipy.integrate import dblquad
import matplotlib as mpl
import matplotlib.pyplot as plt
import cProfile, pstats
from astropy import units as u
from astropy.units import cds

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


m_e = 5.11e-4 # electron/positron mass, in GeV / c^2
epsilon = 0.5 # paper says this varies between 0.5-1
Lambda_0 = 1.4e-27 * u.erg * u.cm**(3) * u.s**(-1) * u.K**(-0.5) # in erg cm^3 s^{-1} K^{-1/2}

n_steps = 10000 # number of integration steps

# Parameters relating to clusters
A262 = True
A2199 = False
A85 = False
NGC5044 = False

# Booleans relating to choice of parameter values
Chan15 = False
upper_error = False

# Define new units
unit_GeV = 1e9 * u.eV
unit_keV = 1e3 * u.eV
unit_kpc = 1e3 * u.pc
unit_Mpc = 1e6 * u.pc
unit_microG = 1e-6 * u.G

E_min = m_e * unit_GeV  # minimum electron/positron energy calculated from BlackHawk
E_max = 5 * unit_GeV # maximum electron/positron energy calculated from BlackHawk


#%%

if A262:
    T_c = 1 * unit_keV / cds.k # maximum core temperature, in keV
    rho_s = 14.1 * 1e14 * u.solMass / unit_Mpc**3 # scale density, in solar masses * Mpc^{-3}
    r_s = 172 * unit_kpc # scale radius, in kpc
    R = 2 * unit_kpc # max radius to integrate out to
    z = 0.0161 # redshift
    beta = 0.433
    r_c = 30 * unit_kpc
    n_0 = 0.94 * 1e-2 * (u.cm)**(-3) # central number density of hot gas particles (thermal electrons/positrons), in cm^{-3}
    
    extension='A262'

    B_0 = 2.9 * unit_microG # maximum central magnetic field, in microgauss
    L_0 = 5.6e38 * u.erg * (u.s)**(-1) # maximum observed luminosity, in erg s^{-1}


# DM density (NFW), in solar masses kpc^{-3}
def rho_NFW(r):
    return rho_s * (r_s / r) * (1 + (r/r_s))**(-2)

# Lorentz factor
def gamma(E):
    return E / (m_e * unit_GeV)

# Number density of hot gas particles (thermal electrons/positrons), in cm^{-3}
def number_density(r):
    return n_0 * (1 + (r/r_c)**2)**(-3*beta/2)

# Cluster magnetic field, in microgauss
def magnetic_field(r):
    return 11 * epsilon**(-1/2) * np.sqrt((number_density(r) / u.cm**(-3)) / 0.1) * (T_c / (2 * unit_keV / cds.k))**(3/4) * unit_microG

def b_C(E, r):
    return 6.13 * (number_density(r)/u.cm**(-3)) * (1 + np.log(gamma(E) / (number_density(r) / u.cm**(-3)))/75)

def b_T(E, r):
    b_1 = 0.0254 * (E / unit_GeV)**2 * (magnetic_field(r) / unit_microG)**2
    b_2 = 0.25 * (E / unit_GeV)**2 * (1+z)**4
    b_3 = 1.51 * (number_density(r) / u.cm**(-3))  * (0.36 + np.log(gamma(E) / (number_density(r) / u.cm**(-3)))) 
    return (b_1 + b_2 + b_3) + b_C(E, r)

def luminosity_integrand_2(r, E):  
    E_prime = energies_ref[energies_ref > E]
    spectrum_integrand = spectrum_ref[energies_ref > E]
    
    return r**2 * np.sum(spectrum_integrand[:-1] * np.diff(E_prime)) * rho_NFW(r) * b_C(E, r) / (m_pbh * b_T(E, r))


def luminosity_predicted_2(): # predicted luminosity
    E_values = 10**np.linspace(np.log10(E_min / unit_GeV), np.log10(E_max / unit_GeV), n_steps) * unit_GeV
    r_values = 10**np.linspace(np.log10(1e-10), np.log10(R / u.kpc), n_steps) * u.kpc
      
    integrand_over_r = []
    for E in E_values:
        
        luminosity_integrand_terms = []
        
        for r in r_values:
            luminosity_integrand_terms.append(luminosity_integrand_2(r, E))
                
        luminosity_integrand_terms_val = []
        for l in luminosity_integrand_terms:
            luminosity_integrand_terms_val.append(l.value)
        luminosity_integrand_terms_unit = luminosity_integrand_terms[0].unit

        integrand_over_r.append(np.sum(luminosity_integrand_terms_val[:-1] * np.diff(r_values)) * luminosity_integrand_terms_unit)
    
        integrand_over_r_terms_val = []
        for a in integrand_over_r:
            integrand_over_r_terms_val.append(a.value)
        integrand_over_r_terms_unit = integrand_over_r[0].unit
    
    integral = np.sum(integrand_over_r_terms_val[:-1] * np.diff(E_values)) * integrand_over_r_terms_unit
    
    return 4 * np.pi * integral

def luminosity_observed(): # observed luminosity
    r_values = np.linspace(0 * u.pc, R, n_steps)
    integrand_terms_val = []
    
    for i in range(n_steps):
        integrand_terms_val.append((number_density(r_values[i])**2 * r_values[i]**2).value)
        
    integrand_unit = (number_density(r_values[0])**2 * r_values[0]**2).unit   

    return 4 * np.pi * Lambda_0 * np.sqrt(T_c) * np.sum(integrand_terms_val[:-1] * np.diff(r_values)) * integrand_unit

print('Magnetic field (microgauss):')
print(magnetic_field(r=0*u.pc))
print(B_0)

luminosity_calculated = luminosity_observed().to(u.erg / u.s)
print('Observed luminosity (calculated) = {:.2e} '.format(luminosity_calculated))
print("Observed luminosity (from LC '21) = {:.2e}".format(L_0))
print('Ratio (calculated to LC21) = {:.5f}'.format(luminosity_calculated / L_0))


#%%
#m_pbh_values = np.array([0.1, 0.12, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.5, 2, 3, 4, 6, 8]) * 10**16
#m_pbh_values = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.5, 3, 6, 8]) * 10**16
m_pbh_values = np.array([1e15])
#m_pbh_values = 10**np.linspace(14.5, 17, 25)
f_pbh_values = []

def main():
    global m_pbh
    for i, m_pbh_val in enumerate(m_pbh_values):
        m_pbh = m_pbh_val * u.g
        #exponent = np.floor(np.log10(m_pbh))
        #coefficient = m_pbh / 10**exponent
        #file_path_data = "../blackhawk_v2.0/results/A22_Fig3_" + "{:.1f}e{:.0f}g/".format(coefficient, exponent)
        #file_path_data = "../blackhawk_v2.0/results/Laha16_Fig1_" + "{:.0f}e{:.0f}g/".format(coefficient, exponent)
        file_path_data = "../Downloads/version_finale/results/LC21_{:.0f}/".format(i+1)
        
        # Load electron secondary spectrum
        global energies_secondary
        global secondary_spectrum
        energies_secondary, secondary_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_secondary_spectra.txt", col=2)
        #energies_secondary, secondary_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_primary_spectra.txt", col=7)
       
        # Evaluate photon spectrum at a set of pre-defined energies
        global energies_ref
        global spectrum_ref
        energies_ref = 10**np.linspace(np.log10(E_min / (unit_GeV)), np.log10(E_max / (unit_GeV)), n_steps) * unit_GeV
        spectrum_ref = np.interp(energies_ref / (unit_GeV), energies_secondary, secondary_spectrum) * (unit_GeV * u.s)**(-1)
        
        luminosity_predicted = luminosity_predicted_2().to(u.erg / u.s)
        f_pbh_values.append(L_0 / luminosity_predicted)


if __name__ == '__main__':
    
    file_path_extracted = './Extracted_files/'
    m_pbh_LC21_extracted, f_PBH_LC21_extracted = load_data("LC21_" + extension + "_NFW.csv")

    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler)
    # Print to file
    stats.sort_stats('cumtime').dump_stats('./cProfiler/LC21_reproduction.txt')
    
    f_pbh_values = []
    main()
    plt.figure()
    plt.plot(m_pbh_values, f_pbh_values)
    plt.xlabel('$M_\mathrm{PBH}$ [g]')
    plt.ylabel('$f_\mathrm{PBH}$')
    plt.title(extension)
    plt.tight_layout()
    plt.yscale('log')
    plt.xscale('log')
    
    extracted_interpolated = 10**np.interp(np.log10(m_pbh_values), np.log10(m_pbh_LC21_extracted), np.log10(f_PBH_LC21_extracted))
    ratio = (extracted_interpolated / (np.array(f_pbh_values)))**(-1)
    frac_diff = abs((extracted_interpolated - (f_pbh_values)) / ( f_pbh_values))
    plt.figure()
    plt.plot(m_pbh_values, ratio, 'x')
    plt.xlabel('$M_\mathrm{PBH}$ [g]')
    plt.ylabel('$f_\mathrm{PBH, extracted} / f_\mathrm{PBH, calculated}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(extension)
    plt.tight_layout()

    plt.figure()
    ratio_2 = (3e61 * np.array(extracted_interpolated) / (np.array(f_pbh_values)))**(-1)
    plt.plot(m_pbh_values, ratio_2, 'x')
    plt.xlabel('$M_\mathrm{PBH}$ [g]')
    plt.ylabel(r'$\frac{f_\mathrm{PBH, reproduced}}{3\times 10^{61} f_\mathrm{PBH, extracted}}$')
    plt.hlines(y=1, xmin=min(m_pbh_values), xmax=max(m_pbh_values), color='k', alpha=0.3)
    plt.xscale('log')
    #plt.yscale('log')
    plt.title(extension)
    plt.tight_layout()

    print(f_pbh_values[0])