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

gamma = 0.2  # fraction of horizon mass which collapses to form PBHs

def log_normal_MF(m, m_c):
    return np.exp(-np.log(m/m_c)**2 / (2*sigma**2)) / (np.sqrt(2*np.pi) * sigma * m)


def load_data(filename):
    return np.genfromtxt(filepath+filename, delimiter=',', unpack=True)



def f_Carr10(beta_prime, m):
    # use Eq. (7.1) from Carr+ '10
    #return 4.11e8 * beta_prime / np.sqrt(m)
    return 3.5e8 * beta_prime / np.sqrt(m)
  
def f_Carr21(beta_prime, m):
    # use Eq. (57) from Carr+ '21
    return 3.81e8 * beta_prime / np.sqrt(m)
    
def find_beta_prime(beta_prime):
    return beta_prime * np.sqrt(gamma)


def f_evap_gamma_1_Carr10(m):
    # Extragalactic and Galactic gamma-ray backgrounds 
    # Secondary flux dominates (M < M_*)
    # Eq. (5.9) Carr+ '10
    beta_prime = 3e-27 * (m/m_star)**(-2.5-2*epsilon) 
    return f_Carr10(beta_prime, m)


def f_evap_gamma_1_Carr21(m):
    # Extragalactic and Galactic gamma-ray backgrounds 
    # Secondary flux dominates (M < M_*)
    # Eq. (32) Carr+ '21
    beta_prime = 5e-28 * (m/m_star)**(-2.5-2*epsilon)
    return f_Carr21(beta_prime, m)
    

def f_evap_gamma_2_Carr10(m):
    # Extragalactic and Galactic gamma-ray backgrounds 
    # Secondary photon emission negligible (M > M_*)
    # Eq. (5.10) Carr+ '10
    beta_prime = 4e-26 * (m/m_star)**(3.5+epsilon)
    return f_Carr10(beta_prime, m)


def f_evap_gamma_2_Carr21(m):
    # Extragalactic and Galactic gamma-ray backgrounds 
    # Secondary photon emission negligible (M > M_*)
    # Eq. (33) Carr+ '21
    beta_prime = 5e-26 * (m/m_star)**(3.5+epsilon)
    return f_Carr21(beta_prime, m)

def f_evap_gamma_3_Carr21(m):
    beta_prime = 3e-30 * (m / (1e13/1.989e33))**(3.1)
    return f_Carr21(beta_prime, m)

def integral_gamma_1_Carr10(m_c):
    #m1 = 3e13 / 1.989e33
    m1 = 1e-50
    m2 = m_star
    m_values = 10**np.linspace(np.log10(m1), np.log10(m2), 10000)    

    integrand = log_normal_MF(m_values, m_c) / f_evap_gamma_1_Carr10(m_values)
    return np.trapz(integrand, m_values)

def integral_gamma_1_Carr21(m_c):
    #m1 = 3e13 / 1.989e33
    m1 = 1e-50
    m2 = m_star
    m_values = 10**np.linspace(np.log10(m1), np.log10(m2), 10000)    

    integrand = log_normal_MF(m_values, m_c) / f_evap_gamma_1_Carr21(m_values)
    return np.trapz(integrand, m_values)


def integral_gamma_2_Carr10(m_c):
    m1 = m_star
    #m2 = np.power(5e9, 1/(3+epsilon)) * m_star
    #m2 = 7e16 / 1.989e33
    m2 = 1e10
    m_values = 10**np.linspace(np.log10(m1), np.log10(m2), 10000)    
    
    integrand = log_normal_MF(m_values, m_c) / f_evap_gamma_2_Carr10(m_values)
    return np.trapz(integrand, m_values)


def integral_gamma_2_Carr21(m_c):
    m1 = m_star
    #m2 = np.power(5e9, 1/(3+epsilon)) * m_star
    #m2 = 7e16 / 1.989e33
    m2 = 1e10
    m_values = 10**np.linspace(np.log10(m1), np.log10(m2), 10000)    
    
    integrand = log_normal_MF(m_values, m_c) / f_evap_gamma_2_Carr21(m_values)
    return np.trapz(integrand, m_values)

def integral_gamma_3_Carr21(m_c):
    m1 = 2.5e13 / 1.989e33
    m2 = 2.4e14 / 1.989e33
    m_values = 10**np.linspace(np.log10(m1), np.log10(m2), 10000)    
    
    integrand = log_normal_MF(m_values, m_c) / f_evap_gamma_3_Carr21(m_values)
    return np.trapz(integrand, m_values)


def combined_constraint_gamma_Carr10(m_c):
    return np.power(integral_gamma_1_Carr10(m_c)**2 + integral_gamma_2_Carr10(m_c)**2, -0.5)

def combined_constraint_gamma_Carr21(m_c):
    return np.power(integral_gamma_1_Carr21(m_c)**2 + integral_gamma_2_Carr21(m_c)**2, -0.5)

def combined_constraint_all_Carr21(m_c):
    return np.power(integral_gamma_1_Carr21(m_c)**2 + integral_gamma_2_Carr21(m_c)**2 + integral_gamma_3_Carr21(m_c)**2, -0.5)


# returns a number as a string in standard form
def string_scientific(val):
    exponent = np.floor(np.log10(val))
    coefficient = val / 10**exponent
    return r'${:.1f} \times 10^{:.0f}$'.format(coefficient, exponent)


m_c_evaporation = 10**np.linspace(-18, -13, 100)
m_evaporation_mono, f_max_evaporation_mono = load_data('Gamma-ray_mono.csv')


if "__main__" == __name__:
    
    
    # Plot the evaporation constraints for a monochromatic MF
    m_values = 10**np.linspace(-18, -15, 100)
    plt.figure(figsize=(10, 8))
    plt.plot(m_evaporation_mono, f_max_evaporation_mono, color='k', alpha=0.25, linewidth=6, label="Extracted (Carr+' 21 Fig. 20)")
    
    #plt.plot(m_values, 2e-8 * (m_values/m_star)**(3+epsilon), linewidth=4, linestyle='dotted', color='k', label=r"$f_\mathrm{max} = 2\times10^{8}(M/M_*)^{3+\epsilon}$")

    
    #plt.plot(m_values, f_evap_gamma_2_Carr10(m_values), linestyle='dotted', linewidth=5, label=r"Using Carr+ '10 Eq. (5.10)")
    #plt.plot(m_values, f_evap_gamma_2_Carr10(m_values), linestyle='dotted', linewidth=5, label=r"Carr+ '10 (using $\beta'$ for $\beta$ in Eq. (7.1))")

    plt.plot(m_values, f_evap_gamma_2_Carr21(m_values), linestyle='dotted', color='tab:orange', linewidth=4, label=r"Using Carr+ '21 Eq. (33)")
    plt.plot(m_values, 0.5*np.array(f_evap_gamma_2_Carr21(m_values)), linestyle='dotted', color='tab:green', linewidth=4, label=r"Using halved prefactor in Carr+ '21 Eq. (33)")

    plt.xlabel('$M~[M_\odot]$')
    plt.ylabel('$f_\mathrm{max}(M)$')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0.8*min(m_evaporation_mono), 1.2*max(m_evaporation_mono))
    plt.ylim(0.8*min(f_max_evaporation_mono), 1.2*max(f_max_evaporation_mono))
    plt.legend()
    plt.tight_layout()


    # Plot the evaporation constraints for a monochromatic MF
    m_values = 10**np.linspace(-18, -15, 100)
    plt.figure(figsize=(10, 8))
    plt.plot(m_evaporation_mono, f_max_evaporation_mono, color='k', alpha=0.25, linewidth=6, label="Extracted (Carr+' 21 Fig. 20)")
    
    plt.plot(m_values, 2e-8 * (m_values/m_star)**(3+epsilon), linewidth=4, linestyle='dotted', color='k', label=r"$f_\mathrm{max} = 2\times10^{8}(M/M_*)^{3+\epsilon}$")

    
    plt.plot(m_values, f_evap_gamma_2_Carr10(m_values), linestyle='dotted', 'color='tab:orange', linewidth=5, label=r"Using Carr+ '10 Eq. (5.10)")
    plt.plot(m_values, 0.5*np.array(f_evap_gamma_2_Carr10(m_values)), linestyle='dotted', color='tab:green', linewidth=3, label=r"Using halved prefactor in Carr+ '10 Eq. (5.10)")

    plt.xlabel('$M~[M_\odot]$')
    plt.ylabel('$f_\mathrm{max}$(M)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(min(m_evaporation_mono), max(m_evaporation_mono))
    plt.ylim(min(f_max_evaporation_mono), max(f_max_evaporation_mono))
    plt.legend()
    plt.tight_layout()
      
    
    # Plot evaporation constraints for a log-normal MF (using expected expressions for beta)
    fig, ax1 = plt.subplots(figsize=(12,8))
    m_c_evaporation_LN, f_pbh_evaporation_LN = load_data('Gamma-ray_LN.csv')
                    
    ax1.plot(m_c_evaporation_LN, f_pbh_evaporation_LN, color='k', alpha=0.25, linewidth=4, label="Extracted (Carr+' 21 Fig. 20)")
        
    f_pbh_evap_Carr10 = []
    f_pbh_evap_Carr21 = []
    f_pbh_evap_gamma_Carr10 = []
    f_pbh_evap_gamma_Carr21 = []

    for m_c in m_c_evaporation:
        f_pbh_evap_Carr10.append(1/integral_gamma_2_Carr10(m_c))
        f_pbh_evap_Carr21.append(1/integral_gamma_2_Carr21(m_c))
        f_pbh_evap_gamma_Carr10.append(combined_constraint_gamma_Carr10(m_c))
        f_pbh_evap_gamma_Carr21.append(combined_constraint_gamma_Carr21(m_c))
  
    ax1.plot(m_c_evaporation, f_pbh_evap_Carr10, linestyle='dotted', linewidth=5, label=r"Carr+ '10 Gamma-ray $(M > M_*)$")
    ax1.plot(m_c_evaporation, f_pbh_evap_Carr21, linestyle='dotted', linewidth=5, label=r"Carr+ '21 Gamma-ray $(M > M_*)$")
    
    ax1.plot(m_c_evaporation, f_pbh_evap_gamma_Carr10, linestyle='dotted', linewidth=5, label=r"Carr+ '10 Gamma-ray (all)")
    ax1.plot(m_c_evaporation, f_pbh_evap_gamma_Carr21, linestyle='dotted', linewidth=5, label=r"Carr+ '21 Gamma-ray (all)")

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
    
    
    # Plot evaporation constraints for a log-normal MF (using expected expressions for beta)
    fig, ax1 = plt.subplots(figsize=(12,8))
    m_c_evaporation_LN, f_pbh_evaporation_LN = load_data('Gamma-ray_LN.csv')
                    
    ax1.plot(m_c_evaporation_LN, f_pbh_evaporation_LN, color='k', alpha=0.25, linewidth=4, label="Extracted (Carr+' 21 Fig. 20)")
        
    f_pbh_evap_Carr21 = []
    f_pbh_evap_gamma_Carr21 = []
    f_pbh_evap_gamma_Carr21_halved = []
    f_pbh_evap_all_Carr21 = []

    for m_c in m_c_evaporation:
        f_pbh_evap_Carr21.append(1/integral_gamma_2_Carr21(m_c))
        f_pbh_evap_gamma_Carr21.append(combined_constraint_gamma_Carr21(m_c))
        f_pbh_evap_gamma_Carr21_halved.append(0.5*combined_constraint_gamma_Carr21(m_c))
        f_pbh_evap_all_Carr21.append(combined_constraint_all_Carr21(m_c))
  
    ax1.plot(m_c_evaporation, f_pbh_evap_Carr21, linestyle='dotted', linewidth=5, label=r"Using Eq. (32) only")
    ax1.plot(m_c_evaporation, f_pbh_evap_gamma_Carr21, linestyle='dotted', linewidth=5, label=r"Using Eq. (32) and Eq. (33)")
    ax1.plot(m_c_evaporation, f_pbh_evap_all_Carr21, linestyle='dotted', linewidth=3, label=r"Using Eq. (32), Eq. (33) and Eq. (28)")
    ax1.plot(m_c_evaporation, f_pbh_evap_gamma_Carr21_halved, linestyle='dotted', linewidth=5, label=r"Using Eq. (32) and Eq. (33) " + "\n " + " (prefactors multiplied by 0.5)")

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
    
    
    
    # Plot evaporation constraints for a log-normal MF (using expected expressions for beta)
    fig, ax1 = plt.subplots(figsize=(12,8))
    m_c_evaporation_LN, f_pbh_evaporation_LN = load_data('Gamma-ray_LN.csv')
                    
    ax1.plot(m_c_evaporation_LN, f_pbh_evaporation_LN, color='k', alpha=0.25, linewidth=4, label="Extracted (Carr+' 21 Fig. 20)")
        
    f_pbh_evap_Carr10 = []
    f_pbh_evap_Carr21 = []
    f_pbh_evap_gamma_Carr10 = []
    f_pbh_evap_gamma_Carr21_halved = []

    for m_c in m_c_evaporation:
        f_pbh_evap_Carr10.append(1/integral_gamma_2_Carr10(m_c))
        f_pbh_evap_Carr21.append(1/integral_gamma_2_Carr21(m_c))
        f_pbh_evap_gamma_Carr10.append(combined_constraint_gamma_Carr10(m_c))
        f_pbh_evap_gamma_Carr21_halved.append(0.5*combined_constraint_gamma_Carr21(m_c))
  
    ax1.plot(m_c_evaporation, f_pbh_evap_Carr21, linestyle='dotted', linewidth=5, label=r"Using Eq. (32) only")
    ax1.plot(m_c_evaporation, f_pbh_evap_gamma_Carr21, linestyle='dotted', linewidth=5, label=r"Using Eq. (32) and Eq. (33)")
    ax1.plot(m_c_evaporation, f_pbh_evap_gamma_Carr21_halved, linestyle='dotted', linewidth=5, label=r"Using Eq. (32) and Eq. (33) " + "\n " + " (prefactors multiplied by 0.5)")

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
