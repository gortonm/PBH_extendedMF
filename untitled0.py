#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 11:42:40 2022

@author: ppxmg2
"""

import numpy as np

m_e # electron/positron mass
r_s # scale radius
rho_s # scale density

m_pbh # PBH mass

# empirical parameters fitted from surface brightness profile
n_0 # cetnral number density of hot gas particles (thermal electrons/positrons), in cm^{-3}
beta # exponent
r_c # radius, in 
epsilon 
T_c # core temperature
Lambda_0

n_steps = 10000 # number of integration steps
E_max = 5 # maximum energy calculated from BlackHawk, in GeV

z # redshift

R # max radius to integrate out to

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
        Q_values.append(Q(E_prime[i]), r)
    
    return np.trapz(Q_values, E_prime) / b_T(E, r)


def luminosity_integrand(E, r):
    return dn_dE(E, r) * b_C(E, r)

def luminosity_predicted(): # predicted luminosity, in erg s^{-1}
    return 4 * np.pi * scipy.dblquad(luminosity_integrand, 0, R, m_e, E_max)

def luminosity_osberved(): # observed luminosity, in erg s^{-1}
    r_values = np.arange(0, R, n_steps)
    integrand_values = []
    
    for i in range(n_steps):
        integrand_values.apend(number_density(r_values[i])**2 * r_values[i]**2)
    
    return 4 * np.pi * Lambda_0 * np.sqrt(T_c) * np.trapz(integrand_values, r_values)