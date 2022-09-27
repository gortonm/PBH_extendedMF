#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 11:42:40 2022

@author: ppxmg2
"""

import numpy as np
from reproduce_COMPTEL_constraints_v2 import read_blackhawk_spectra
from scipy.integrate import dblquad
import matplotlib as mpl
import matplotlib.pyplot as plt

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
Lambda_0 = 1.4e-27 # in erg cm^{-3} s^{-1} K^{-1/2}

n_steps = 1000 # number of integration steps
E_max = 5 # maximum energy calculated from BlackHawk, in GeV

# Parameters relating to clusters
A262 = True
A2199 = False
A85 = False
NGC5044 = False

# unit conversions
erg_to_GeV = 624.15    # erg to GeV
g_to_solar_mass = 1 / 1.989e33    # g to solar masses
kpc_to_cm = 3.09e21    # kpc to cm
keV_to_K = 11604525    # keV to Kelvin

#%%
if A262:
    T_c = 1 # maximum core temperature, in keV
    rho_s = 14.1 * 1e14 # scale density, in solar masses * Mpc^{-3}
    r_s = 172 # scale radius, in kpc
    R = 2 # max radius to integrate out to
    z = 0.0161 # redshift
    beta = 0.433
    r_c = 30
    n_0 = 0.94 * 1e-2 # central number density of hot gas particles (thermal electrons/positrons), in cm^{-3}

    B_0 = 2.9 # maximum central magnetic field, in microgauss
    L_0 = 5.6e38 # luminosity, in erg s^{-1}
    
if A2199:
    T_c = 2 # maximum core temperature, in keV
    rho_s = 9.56 * 1e14 # scale density, in solar masses * Mpc^{-3}
    r_s = 334 # scale radius, in kpc
    R = 3 # max radius to integrate out to, in kpc
    z = 0.0302 # redshift
    beta = 0.665
    r_c = 102
    n_0 = 0.97 * 1e-2 # central number density of hot gas particles (thermal electrons/positrons), in cm^{-3}

    B_0 = 4.9 # maximum central magnetic field, in microgauss
    L_0 = 2.3e39 # luminosity, in erg s^{-1}
    
if A85:
    T_c = 3 # maximum core temperature, in keV
    rho_s = 8.34 * 1e14 # scale density, in solar masses * Mpc^{-3}
    r_s = 444 # scale radius, in kpc
    R = 3 # max radius to integrate out to, in kpc
    z = 0.0556 # redshift
    beta = 0.532
    r_c = 60
    n_0 = 3.00 * 1e-2 # central number density of hot gas particles (thermal electrons/positrons), in cm^{-3}

    B_0 = 11.6 # maximum central magnetic field, in microgauss
    L_0 = 2.7e40 # luminosity, in erg s^{-1}
 
if NGC5044:
    T_c = 0.8 # maximum core temperature, in keV
    rho_s = 14.7 * 1e14 # scale density, in solar masses * Mpc^{-3}
    r_s = 127 # scale radius, in kpc
    R = 1 # max radius to integrate out to, in kpc
    z = 0.009 # redshift
    beta = 0.524
    r_c = 8
    n_0 = 4.02 * 1e-2 # central number density of hot gas particles (thermal electrons/positrons), in cm^{-3}

    B_0 = 5.0 # maximum central magnetic field, in microgauss
    L_0 = 6.3e38 # luminosity, in erg s^{-1}


# DM density profile
def rho_NFW(r):
    return rho_s * (r_s / r) * (1 + (r/r_s))**(-2)

# Lorentz factor
def gamma(E):
    return E / m_e

# Number density of hot gas particles (thermal electrons/positrons), in cm^{-3}
def number_density(r):
    return n_0 * (1 + (r/r_c)**2)**(-3*beta/2)

# Cluster magnetic field, in microgauss
def magnetic_field(r):
    return 11 * epsilon**(-1/2) * np.sqrt(number_density(r) / 0.1) * (T_c / 2)**(3/4)

def b_C(E, r):
    return 6.13 * number_density(r) * (1 + np.log(gamma(E)/number_density(r))/75)

def b_T(E, r):
    b_1 = 0.0254 * E**2 * magnetic_field(r)**2
    b_2 = 0.25 * E**2 * (1+z)**4
    b_3 = 1.51 * number_density(r) * (0.36 + np.log(gamma(E) / number_density(r)))
    return b_1 + b_2 + b_3 + b_C(E, r)

def Q(E, r):
    # Load electron secondary spectrum
    energies_secondary, secondary_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_secondary_spectra.txt", col=2)
    
    return np.interp(E, energies_secondary, secondary_spectrum) * rho_NFW(r) / m_pbh

def dn_dE(E, r):
    E_prime = 10**np.linspace(np.log10(E), np.log10(E_max), n_steps)
    
    Q_values = []
    for i in range(n_steps):
        Q_values.append(Q(E_prime[i], r))
    
    return np.trapz(Q_values, E_prime) / b_T(E, r)


def luminosity_integrand(E, r):
    return dn_dE(E, r) * b_C(E, r)

def luminosity_predicted(): # predicted luminosity, in erg s^{-1}
    return 4 * np.pi * dblquad(luminosity_integrand, 0, R, m_e, E_max)

def luminosity_osberved(): # observed luminosity, in erg s^{-1}
    r_values = np.linspace(0, R, n_steps)
    integrand_values = []
    
    for i in range(n_steps):
        integrand_values.append(number_density(r_values[i])**2 * r_values[i]**2)
    
    return 4 * np.pi * Lambda_0 * np.sqrt(T_c) * np.trapz(integrand_values, r_values)

print('Magnetic field (microgauss):')
print(magnetic_field(r=0))
print(B_0)

print('Observed luminosity [ erg s^{-1} ]')
print(luminosity_osberved() * np.sqrt(keV_to_K) * kpc_to_cm**3)
print(L_0)

#%%
m_pbh_values = np.array([0.1, 0.12, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.5, 2, 3, 4, 6, 8]) * 10**16
f_pbh_values = []
for m_pbh in m_pbh_values:
    exponent = np.floor(np.log10(m_pbh))
    coefficient = m_pbh / 10**exponent
    file_path_data = "../blackhawk_v2.0/results/A22_Fig3_" + "{:.1f}e{:.0f}g/".format(coefficient, exponent)
    
    f_pbh_values.append(erg_to_GeV * (g_to_solar_mass)**(-1) * keV_to_K**0.5 * kpc_to_cm**3 * luminosity_osberved() / luminosity_predicted())
    
plt.plot(m_pbh_values, f_pbh_values)
plt.xlabel('$M_\mathrm{PBH}$ [g]')
plt.ylabel('$f_\mathrm{PBH}$')
plt.tight_layout()
plt.yscale('log')
plt.xscale('log')