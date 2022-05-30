#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 13:32:36 2022
@author: ppxmg2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import erf

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


filepath = './Extracted_files/'

m_star = 5e14 / 1.989e33    # use value of M_* from Carr+ '17
sigma = 2
epsilon = 0.4

#m2 = 7e16 / 1.989e33    # using maximum mass applicable for extragalactic gamma-ray constraints from Carr+ '10
m2 = 1e18 / 1.989e33    # using maximum mass applicable for extragalactic gamma-ray constraints from Table I of Carr, Kuhnel & Sandstad '16
m2 = np.power(5e9, 1/(3+epsilon)) * m_star    # using value of M_2 for which f_max(M_2) = 100
m1 = m_star
#m1 = 1e15 / 1.989e33
#m1 = 1e-20

def log_normal_MF(m, m_c):
    return np.exp(-np.log(m/m_c)**2 / (2*sigma**2)) / (np.sqrt(2*np.pi) * sigma * m)


def load_data(filename):
    return np.genfromtxt(filepath+filename, delimiter=',', unpack=True)


def f(beta_prime, m, prefactor):
    return prefactor * beta_prime / np.sqrt(m)    # prefactor from Carr, Kuhnel & Sandstad '16 Eq. 8


def f_evap_gamma_1(m, prefactor):
    # Extragalactic and Galactic gamma-ray backgrounds 
    # Secondary flux dominates (M < M_*)
    # Eq. (5.9) Carr+ '10
    beta_prime = 3e-27 * (m/m_star)**(-2.5-2*epsilon)
    return f(beta_prime, m, prefactor)
    

def f_evap_gamma_2(m, prefactor):
    # Extragalactic and Galactic gamma-ray backgrounds 
    # Secondary photon emission negligible (M > M_*)
    # Eq. (5.10) Carr+ '10
    beta_prime = 4e-26 * (m/m_star)**(3.5+epsilon)
    return f(beta_prime, m, prefactor)


def f_evap_CMB(m, prefactor):
    # CMB anisotropies
    # Eq. (6.12) Carr+ '10
    beta_prime = 3e-30 * (m/(1e13 / 1.989e33))**3.1
    return f(beta_prime, m, prefactor)


def integral_gamma_1(m_c, prefactor):
    m1 = 3e13 / 1.989e33
    m1 = 1e-50
    m2 = m_star
    m_values = 10**np.linspace(np.log10(m1), np.log10(m2), 10000)    

    integrand = log_normal_MF(m_values, m_c) / f_evap_gamma_1(m_values, prefactor)
    return np.trapz(integrand, m_values)


def integral_gamma_2(m_c, prefactor):
    m1 = m_star
    #m2 = np.power(5e9, 1/(3+epsilon)) * m_star
    #m2 = 7e16 / 1.989e33
    m2 = 1e10
    m_values = 10**np.linspace(np.log10(m1), np.log10(m2), 10000)    
    
    integrand = log_normal_MF(m_values, m_c) / f_evap_gamma_2(m_values, prefactor)
    return np.trapz(integrand, m_values)


def integral_CMB(m_, prefactor):
    m1 = 2.5e13 / 1.989e33
    m2 = 2.4e14 / 1.989e33
    m_values = 10**np.linspace(np.log10(m1), np.log10(m2), 10000)
    
    integrand = log_normal_MF(m_values, m_c) / f_evap_CMB(m_values, prefactor)
    return np.trapz(integrand, m_values)


def combined_constraint_gamma(m_c, prefactor):
    print('M_c = ', m_c)
    print(integral_gamma_1(m_c, prefactor))
    print(integral_gamma_2(m_c, prefactor))
    # first integral is consistently the largest
    
    return np.power(integral_gamma_1(m_c, prefactor)**2 + integral_gamma_2(m_c, prefactor)**2, -0.5)

def combined_constraint_gamma_CMB(m_c, prefactor):
    print('M_c = ', m_c)
    print(integral_gamma_1(m_c, prefactor))
    print(integral_gamma_2(m_c, prefactor))
    print(integral_CMB(m_c, prefactor))
    # first integral is consistently the largest
    
    return np.power((integral_gamma_1(m_c, prefactor)**2 + integral_gamma_2(m_c, prefactor)**2 + integral_CMB(m_c, prefactor)**2), -0.5)

# returns a number as a string in standard form
def string_scientific(val):
    exponent = np.floor(np.log10(val))
    coefficient = val / 10**exponent
    return r'${:.2f} \times 10^{:.0f}$'.format(coefficient, exponent)


m_c_evaporation = 10**np.linspace(-18, -13, 100)
m_evaporation_mono, f_max_evaporation_mono = load_data('Gamma-ray_mono.csv')


if "__main__" == __name__:
    
    
    # Plot the evaporation constraints for a monochromatic MF
    m_values = 10**np.linspace(-18, -15, 100)
    plt.figure()
    plt.plot(m_evaporation_mono, f_max_evaporation_mono, color='k', alpha=0.25, linewidth=4, label='Extracted (Carr 21)')
    for prefactor in ([1.7e8, 3.5e8, 3.8e8]):
        plt.plot(m_values, f_evap_gamma_2(m_values, prefactor), linestyle='dotted', linewidth=5, label=r"Calculated, $f_\mathrm{max}(M) = $" +  string_scientific(prefactor) + r"$ \beta'(M) (M/M_\odot)^{-1/2}$")
    plt.xlabel('$M_\mathrm{c}~[M_\odot]$')
    plt.ylabel('$f_\mathrm{PBH}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
      

    # Plot evaporation constraints for a log-normal MF
    fig, ax1 = plt.subplots(figsize=(12,8))
    m_c_evaporation_LN, f_pbh_evaporation_LN = load_data('Gamma-ray_LN.csv')
                    
    ax1.plot(m_c_evaporation_LN, f_pbh_evaporation_LN, color='k', alpha=0.25, linewidth=4, label='Extracted (Carr 21)')
    
    prefactor = 3.5e8
    
    f_pbh_evap = []
    f_pbh_evap_gamma = []
    f_pbh_evap_gamma_CMB = []

    for m_c in m_c_evaporation:
        f_pbh_evap.append(1/integral_gamma_2(m_c, prefactor))
        f_pbh_evap_gamma.append(combined_constraint_gamma(m_c, prefactor))
        f_pbh_evap_gamma_CMB.append(combined_constraint_gamma_CMB(m_c, prefactor))
  
    ax1.plot(m_c_evaporation, f_pbh_evap, linestyle='dotted', linewidth=5, label=r"Gamma-ray $(M > M_*)$")
    ax1.plot(m_c_evaporation, f_pbh_evap_gamma, linestyle='dotted', linewidth=5, label=r"Gamma-ray (all)")
    ax1.plot(m_c_evaporation, f_pbh_evap_gamma_CMB, linestyle='dotted', linewidth=4, label=r"Gamma-ray (all) + CMB anisotropies")

    ax1.set_xlabel('$M_\mathrm{c}~[M_\odot]$')
    ax1.set_ylabel('$f_\mathrm{PBH}$')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylim(1e-4, 1)
    ax1.set_xlim(1e-15, 1e-13)

    ax1.legend()
    
    ax2 = plt.gca().twiny()
    ax2.plot(np.array(m_c_evaporation)*1.989e33, np.zeros(len(m_c_evaporation)))
    ax2.set_xlabel('$M_\mathrm{c}~[g]$')
    ax2.set_xscale('log')    
    ax2.tick_params(axis='x')
    
    ax2.set_title('Log-normal ($\sigma = {:.0f}$)'.format(sigma), pad=20)
    plt.tight_layout()
    