#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 21:11:32 2022

@author: ppxmg2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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

m_star = 5e14  # all masses in grams
sigma = 2
epsilon = 0.4

m1 = 3e13   # minimum mass applicable for extragalactic gamma-ray constraints from Carr+ '10
#m2 = 7e16   # maximum mass applicable for extragalactic gamma-ray constraints from Carr+ '10
m2 = 1e18  # maximum mass shown in Fig. 7 of Carr+ '21

def load_data(filename):
    return np.genfromtxt(filepath+filename, delimiter=',', unpack=True)


def beta_prime_gamma_1_Carr10(m):
    # Extragalactic and Galactic gamma-ray backgrounds 
    # Secondary flux dominates (M < M_*)
    # Eq. (5.9) Carr+ '10
    return 3e-27 * (m/m_star)**(-2.5-2*epsilon)
    
def beta_prime_gamma_2_Carr10(m):
    # Extragalactic and Galactic gamma-ray backgrounds 
    # Secondary photon emission negligible (M > M_*)
    # Eq. (5.10) Carr+ '10
    return 4e-26 * (m/m_star)**(3.5+epsilon)

def beta_prime_gamma_1_Carr21(m):
    # Extragalactic and Galactic gamma-ray backgrounds 
    # Secondary flux dominates (M < M_*)
    # Eq. (32) Carr+ '21
    return 5e-28 * (m/m_star)**(-2.5-2*epsilon)
    
def beta_prime_gamma_2_Carr21(m):
    # Extragalactic and Galactic gamma-ray backgrounds 
    # Secondary photon emission negligible (M > M_*)
    # Eq. (33) Carr+ '21
    return 5e-26 * (m/m_star)**(3.5+epsilon)

# returns a number as a string in standard form
def string_scientific(val):
    exponent = np.floor(np.log10(val))
    coefficient = val / 10**exponent
    return r'${:.2f} \times 10^{:.0f}$'.format(coefficient, exponent)


m, beta_prime = load_data('Carr+21_Fig7.csv')


if "__main__" == __name__:
    
    # Plot the evaporation constraints for a monochromatic MF
    m_values_1 = 10**np.linspace(np.log10(m1), np.log10(m_star), 100)
    m_values_2 = 10**np.linspace(np.log10(m_star), np.log10(m2), 100)

    plt.plot(m_values_1, beta_prime_gamma_1_Carr10(m_values_1), linestyle='dotted', linewidth=5, label=r"Carr+ '10")
    plt.plot(m_values_2, beta_prime_gamma_2_Carr10(m_values_2), linestyle='dotted', linewidth=5, label=r"Carr+ '10")
    
    plt.plot(m_values_1, beta_prime_gamma_1_Carr21(m_values_1), linestyle='dotted', linewidth=4, label=r"Carr+ '21")
    plt.plot(m_values_2, beta_prime_gamma_2_Carr21(m_values_2), linestyle='dotted', linewidth=4, label=r"Carr+ '21")
    
    plt.plot(m, beta_prime, linewidth=6, color='k', alpha=0.25)
    
    plt.xlabel('$M_\mathrm{c}~[M_\odot]$')
    plt.ylabel('$f_\mathrm{PBH}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r"$\beta'(M)$")
    plt.xlabel('$M~[g]$')
    plt.legend()
      

