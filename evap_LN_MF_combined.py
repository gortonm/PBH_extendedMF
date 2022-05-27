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


def f(beta_prime, m):
    #return 1.7e8 * beta_prime / np.sqrt(m)    # prefactor from Carr, Kuhnel & Sandstad '16 Eq. 8
    return 1.7e8 * beta_prime / np.sqrt(m)   # prefactor from email from Vaskonen


def f_evap_analytic_1(m):
    beta_prime = 3e-27 * (m/m_star)**(-2.5-2*epsilon)
    return f(beta_prime, m)
    

def f_evap_analytic_2(m):
    beta_prime = 4e-26 * (m/m_star)**(3.5+epsilon)
    return f(beta_prime, m)


def f_evap_analytic_3(m):
    beta_prime = 3e-30 * (m/(1e13 / 1.989e33))**3.1
    return f(beta_prime, m)


def integral_1(m_c):
    m1 = 1e-30
    m2 = m_star
    m_values = 10**np.linspace(np.log10(m1), np.log10(m2), 10000)    

    integrand = log_normal_MF(m_values, m_c) / f_evap_analytic_1(m_values)
    return np.trapz(integrand, m_values)


def integral_2(m_c):
    m1 = m_star
    m2 = np.power(5e9, 1/(3+epsilon)) * m_star
    m_values = 10**np.linspace(np.log10(m1), np.log10(m2), 10000)    
    
    integrand = log_normal_MF(m_values, m_c) / f_evap_analytic_2(m_values)
    return np.trapz(integrand, m_values)


def integral_3(m_c):
    m1 = 2.5e13 / 1.989e33
    m2 = 2.4e14 / 1.989e33
    m_values = 10**np.linspace(np.log10(m1), np.log10(m2), 10000)
    
    integrand = log_normal_MF(m_values, m_c) / f_evap_analytic_3(m_values)
    return np.trapz(integrand, m_values)


def combined_constraint(m_c):
    return np.power((integral_1(m_c)**2 + integral_2(m_c)**2 + integral_3(m_c)**2), -0.5)


m_c_evaporation = 10**np.linspace(-18, -13, 100)
m_evaporation_mono, f_max_evaporation_mono = load_data('Gamma-ray_mono.csv')


if "__main__" == __name__:

    # Plot evaporation constraints for a log-normal MF
    fig, ax1 = plt.subplots(figsize=(12,8))
    m_c_evaporation_LN, f_pbh_evaporation_LN = load_data('Gamma-ray_LN.csv')

    f_pbh_evap = []
    
    for m_c in m_c_evaporation:
        f_pbh_evap.append(combined_constraint(m_c))
                    
    ax1.plot(m_c_evaporation_LN, f_pbh_evaporation_LN, color='k', alpha=0.25, linewidth=4, label='Extracted (Carr 21)')
    ax1.plot(m_c_evaporation, f_pbh_evap, linestyle='dotted', linewidth=5, label='Calculated')
    ax1.set_xlabel('$M_\mathrm{c}~[M_\odot]$')
    ax1.set_ylabel('$f_\mathrm{PBH}$')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    #ax1.set_ylim(10**(-4), 1)
    #ax1.set_xlim(1e-16, 1e-12)
    
    ax2 = plt.gca().twiny()
    ax2.plot(np.array(m_c_evaporation)*1.989e33, np.zeros(len(m_c_evaporation)))
    ax2.set_xlabel('$M_\mathrm{c}~[g]$')
    ax2.set_xscale('log')    
    ax2.tick_params(axis='x')
    
    ax2.set_title('Log-normal ($\sigma = {:.0f}$)'.format(sigma), pad=20)
    plt.tight_layout()