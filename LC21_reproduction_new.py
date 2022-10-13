#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 11:42:40 2022

@author: ppxmg2
"""

import numpy as np
from reproduce_COMPTEL_constraints_v2 import read_blackhawk_spectra, load_data
import matplotlib as mpl
import matplotlib.pyplot as plt
import cProfile, pstats
#from astropy import units as u
#from astropy.units import cds

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


# Reproduce Lee & Chan (2021) Fig. 1, expressing intermediate quantities
# in units of [GeV, cm, s, K, microGauss]

n_steps = 10000 # number of integration steps

# Parameters relating to clusters
A262 = True
A2199 = False
A85 = False
NGC5044 = False

# Booleans relating to choice of parameter values
Chan15 = False
upper_error = False

# Unit conversion factors
keV_to_GeV = 1e-3
solMass_to_g = 1.98847e33
Mpc_to_cm = 3.0857e24
kpc_to_cm = 3.0857e21
GeV_to_erg = 0.00160218

c = 2.99792458e11  # speed of light, in cm / s
k_B = 8.617333262e-8  # Botlzmann constant, in keV/ K


m_e = 5.11e-4 / c**2 # electron/positron mass, in GeV / c^2
epsilon = 0.5 # paper says this varies between 0.5-1
Lambda_0 = 1.4e-27 # in erg cm^3 s^{-1} K^{-1/2}

E_min = m_e * c**2  # minimum electron/positron energy calculated from BlackHawk, in GeV
E_max = 5  # maximum electron/positron energy calculated from BlackHawk, in GeV


#%%

if A262:
    T_c_keV = 1  # maximum core temperature, in keV
    T_c_K = 1 / k_B  # maximum core temperature, in K
    
    rho_s = 14.1 * 1e14 * solMass_to_g / Mpc_to_cm**3 # scale density, in g / cm^3
    r_s = 172 * kpc_to_cm # scale radius, in cm
    R = 2 * kpc_to_cm # max radius to integrate out to, in cm
    z = 0.0161 # redshift
    beta = 0.433
    r_c = 30 * kpc_to_cm
    n_0 = 0.94 * 1e-2  # central number density of hot gas particles (thermal electrons/positrons), in cm^{-3}
    
    extension='A262'

    B_0 = 2.9 # maximum central magnetic field, in microgauss
    L_0 = 5.6e38 # maximum observed luminosity, in erg s^{-1}


# DM density (NFW), in g cm^{-3}
def rho_NFW(r):
    return rho_s * (r_s / r) * (1 + (r/r_s))**(-2)

# Lorentz factor
def gamma(E):
    return E / (m_e * c**2)

# Number density of hot gas particles (thermal electrons/positrons), in cm^{-3}
def number_density(r):
    return n_0 * (1 + (r/r_c)**2)**(-3*beta/2)

# Cluster magnetic field, in microgauss
def magnetic_field(r):
    return 11 * epsilon**(-1/2) * np.sqrt((number_density(r)) / 0.1) * (T_c_keV / (2))**(3/4)

def b_C(E, r):
    return 6.13 * (number_density(r)) * (1 + np.log(gamma(E) / (number_density(r)))/75)

def b_T(E, r):
    b_1 = 0.0254 * (E)**2 * (magnetic_field(r))**2
    b_2 = 0.25 * (E)**2 * (1+z)**4
    b_3 = 1.51 * (number_density(r))  * (0.36 + np.log(gamma(E) / (number_density(r)))) 
    return (b_1 + b_2 + b_3) + b_C(E, r)

def luminosity_integrand_2(r, E):  
    E_prime = energies_ref[energies_ref > E]
    spectrum_integrand = spectrum_ref[energies_ref > E]
    
    return r**2 * np.sum(spectrum_integrand[:-1] * np.diff(E_prime)) * rho_NFW(r) * b_C(E, r) / (m_pbh * b_T(E, r))


def luminosity_predicted_2(): # predicted luminosity, in erg s^{-1}
    E_values = 10**np.linspace(np.log10(E_min), np.log10(E_max), n_steps)
    r_values = 10**np.linspace(np.log10(1e-10 * kpc_to_cm), np.log10(R), n_steps)
      
    integrand_over_r = []
    for E in E_values:
        
        luminosity_integrand_terms = []
        
        for r in r_values:
            luminosity_integrand_terms.append(luminosity_integrand_2(r, E))
                
        integrand_over_r.append(np.sum(luminosity_integrand_terms[:-1] * np.diff(r_values)))
    print('integrand_over_r[-1] = ', integrand_over_r[-1])
    integral = np.sum(integrand_over_r[:-1] * np.diff(E_values))

    print('4 * np.pi * integral * GeV_to_erg = ', 4 * np.pi * integral * GeV_to_erg)
    
    return 4 * np.pi * integral * GeV_to_erg

def luminosity_observed(): # observed luminosity
    r_values = np.linspace(0, R, n_steps)
    integrand_terms = []
    

    for i in range(n_steps):
        integrand_terms.append((number_density(r_values[i])**2 * r_values[i]**2))
        
    print(np.sum(integrand_terms[:-1] * np.diff(r_values)))
    return 4 * np.pi * Lambda_0 * np.sqrt(T_c_K) * np.sum(integrand_terms[:-1] * np.diff(r_values))

print('Magnetic field (microgauss):')
print(magnetic_field(r=0))
print(B_0)

luminosity_calculated = luminosity_observed()
print('Observed luminosity (calculated) = {:.2e} '.format(luminosity_calculated))
print("Observed luminosity (from LC '21) = {:.2e}".format(L_0))
print('Ratio (calculated to LC21) = {:.5f}'.format(luminosity_calculated / L_0))



#%%
#m_pbh_values = np.array([0.1, 0.12, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.5, 2, 3, 4, 6, 8]) * 10**16
m_pbh_values = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.5, 3, 6, 8]) * 10**16
#m_pbh_values = np.array([1e15])
#m_pbh_values = 10**np.linspace(14.5, 17, 25)
f_pbh_values = []

def main():
    global m_pbh
    for i, m_pbh_val in enumerate(m_pbh_values):
        print('M_PBH = {:.2e} g'.format(m_pbh_val))
        m_pbh = m_pbh_val
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
        energies_ref = 10**np.linspace(np.log10(E_min), np.log10(E_max), n_steps)
        spectrum_ref = np.interp(energies_ref, energies_secondary, secondary_spectrum)
        
        luminosity_predicted = luminosity_predicted_2()
        f_pbh_values.append(L_0 / luminosity_predicted)


if __name__ == '__main__':
    
    file_path_extracted = './Extracted_files/'
    m_pbh_LC21_extracted, f_PBH_LC21_extracted = load_data("LC21_" + extension + "_NFW.csv")
    """
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler)
    # Print to file
    stats.sort_stats('cumtime').dump_stats('./cProfiler/LC21_reproduction.txt')
    """
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
    ratio = extracted_interpolated / np.array(f_pbh_values)
    frac_diff = abs((extracted_interpolated - f_pbh_values) / f_pbh_values)
    
    plt.figure()
    plt.plot(m_pbh_values, ratio, 'x')
    plt.xlabel('$M_\mathrm{PBH}$ [g]')
    plt.ylabel('$f_\mathrm{PBH, extracted} / f_\mathrm{PBH, calculated}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(extension)
    plt.tight_layout()
    
    plt.figure()
    plt.plot(m_pbh_values, frac_diff, 'x')
    plt.xlabel('$M_\mathrm{PBH}$ [g]')
    plt.ylabel('$(f_\mathrm{PBH, extracted} / f_\mathrm{PBH, calculated}) - 1$')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(extension)
    plt.tight_layout()

    print('ratio =', ratio)
    print('fractional difference =', frac_diff)