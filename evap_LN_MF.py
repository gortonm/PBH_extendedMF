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

m2 = 7e16 / 1.989e33    # using maximum mass applicable for extragalactic gamma-ray constraints from Carr+ '10
m2 = 1e18 / 1.989e33    # using maximum mass applicable for extragalactic gamma-ray constraints from Table I of Carr, Kuhnel & Sandstad '16
#m2 = np.power(5e9, 1/(3+epsilon)) * m_star    # using value of M_2 for which f_max(M_2) = 100
m1 = m_star
m1 = 1e15 / 1.989e33

def log_normal_MF(m, m_c):
    return np.exp(-np.log(m/m_c)**2 / (2*sigma**2)) / (np.sqrt(2*np.pi) * sigma * m)

def load_data(filename):
    return np.genfromtxt(filepath+filename, delimiter=',', unpack=True)

def integrand(m, m_c):
    return log_normal_MF(m, m_c) / f_evap(m)

def integrand_subaru(m, m_c):
    return log_normal_MF(m, m_c) / f_subaru(m)

def constraint_mono_analytic(m):
    return 2e-8 * np.array(m/m_star)**(3+epsilon)

def integral_analytic(m_c, m):
    return erf((((epsilon + 3) * sigma**2) + np.log(m/m_c))/(np.sqrt(2) * sigma)) 

def constraint_analytic(m_range, m_c):
    prefactor = 4e-8 * (m_c / m_star)**(3+epsilon) * np.exp(-0.5 * (epsilon + 3)**2 * sigma**2) 
    return prefactor / (integral_analytic(m_c, max(m_range)) - integral_analytic(m_c, min(m_range)))

m_c_evaporation = 10**np.linspace(-18, -13, 100)
m_evaporation_mono, f_max_evaporation_mono = load_data('Gamma-ray_mono.csv')

m_c_subaru = 10**np.linspace(-15, -4, 100)
m_subaru_mono, f_max_subaru_mono = load_data('Subaru-HSC_mono.csv')

def f_evap(m):
    return np.interp(m, m_evaporation_mono, f_max_evaporation_mono)

def f_subaru(m):
    return np.interp(m, m_subaru_mono, f_max_subaru_mono)

# Try creating some new data to match the monochromatic MF constraint
def integrand_2(m, m_c):
    return log_normal_MF(m, m_c) / constraint_mono_analytic(m)


if "__main__" == __name__:

    # Plot evaporation constraints for a log-normal MF
    fig, ax1 = plt.subplots(figsize=(12,8))
    m_c_evaporation_LN, f_pbh_evaporation_LN = load_data('Gamma-ray_LN.csv')

    f_pbh_evap = []
    f_pbh_evap_2 = []

    f_pbh_evap_analytic = []
    f_pbh_evap_analytic_2 = []
    f_pbh_evap_analytic_3 = []
    f_pbh_evap_analytic_4 = []
    
    #m1 = min(m_evaporation_mono)
    #m2 = max(m_evaporation_mono)
    
    
    for m_c in m_c_evaporation:
        
        m_range = 10**np.linspace(np.log10(max(m1, m_star)), np.log10(m2), 10000)
        #m_range = np.linspace(max(m1, m_star), m2, 100000)   # no noticeable difference
                
        f_pbh_evap.append(1/np.trapz(integrand(m=m_range, m_c=m_c), m_range))
        f_pbh_evap_2.append(1/np.trapz(integrand_2(m=m_range, m_c=m_c), m_range))

        f_pbh_evap_analytic.append(constraint_analytic(m_range, m_c))
        f_pbh_evap_analytic_2.append(0.1*constraint_analytic(m_range, m_c))
        f_pbh_evap_analytic_3.append(0.05*constraint_analytic(m_range, m_c))
        f_pbh_evap_analytic_4.append(0.01*constraint_analytic(m_range, m_c))

        
    #ax1.plot(m_c_evaporation, f_pbh_evap, label='Trapezium rule', linestyle = 'dotted', linewidth=6)
    #x1.plot(m_c_evaporation, f_pbh_evap_2, label='Trapezium rule ($f_\mathrm{max}$ analytic)', linestyle = 'dotted', linewidth=4)

    ax1.plot(m_c_evaporation, f_pbh_evap_analytic, label='Analytic', linestyle = 'dotted', linewidth=5)
    ax1.plot(m_c_evaporation, f_pbh_evap_analytic_2, label=r'0.1 $\times$ Analytic', linestyle = 'dotted', linewidth=5)
    ax1.plot(m_c_evaporation, f_pbh_evap_analytic_3, label=r'0.05 $\times$ Analytic', linestyle = 'dotted', linewidth=5)
    ax1.plot(m_c_evaporation, f_pbh_evap_analytic_4, label=r'0.01 $\times$ Analytic', linestyle = 'dotted', linewidth=5)

    ax1.plot(m_c_evaporation_LN, f_pbh_evaporation_LN, color='k', alpha=0.25, linewidth=4, label='Extracted (Carr 21)')    
    ax1.set_xlabel('$M_\mathrm{c}~[M_\odot]$')
    ax1.set_ylabel('$f_\mathrm{PBH}$')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.set_ylim(10**(-4), 1)
    
    ax2 = plt.gca().twiny()
    ax2.plot(np.array(m_c_evaporation)*1.989e33, np.zeros(len(m_c_evaporation)))
    ax2.set_xlabel('$M_\mathrm{c}~[g]$')
    ax2.set_xscale('log')    
    ax2.tick_params(axis='x')
    
    ax2.set_title('Log-normal ($\sigma = {:.0f}$)'.format(sigma) + ', $(M_1, M_2) = ({:.0e}, {:.0e})$ g'.format(m1*1.989e33, m2*1.989e33), pad=20)
    plt.tight_layout()

    """
    # Plot Subaru-HSC constraints for a log-normal MF
    fig, ax1 = plt.subplots(figsize=(12,8))
    m_c_subaru_LN, f_max_subaru_LN = load_data('Subaru-HSC_LN.csv')
    f_pbh_subaru = []
    
    for m_c in m_c_subaru:

        m_range = 10**np.linspace(np.log10(min(m_subaru_mono)), np.log10(max(m_subaru_mono)), 10000)
        #m_range = np.linspace(min(m_subaru_mono), max(m_subaru_mono), 100000)
        f_pbh_subaru.append(1/np.trapz(integrand_subaru(m=m_range, m_c=m_c), m_range))
        
    ax1.plot(m_c_subaru, f_pbh_subaru, label='Trapezium rule', linestyle = 'dotted', linewidth=6)
    ax1.plot(m_c_subaru_LN, f_max_subaru_LN, color='k', alpha=0.25, linewidth=4, label='Extracted (Carr 21)')

    ax1.set_xlabel('$M_\mathrm{c}~[M_\odot]$')
    ax1.set_ylabel('$f_\mathrm{PBH}$')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.set_ylim(10**(-4), 1)
    
    ax2 = plt.gca().twiny()
    ax2.plot(np.array(m_c_subaru)*1.989e33, np.zeros(len(m_c_subaru)))
    ax2.set_xlabel('$M_\mathrm{c}~[g]$')
    ax2.set_xscale('log')    
    ax2.tick_params(axis='x')
    
    ax2.set_title('Log-normal ($\sigma = {:.0f}$)'.format(sigma), pad=20)
    plt.tight_layout()
 

    # Plot evaporation constraints for a monochromatic MF
    plt.figure(figsize=(12,8))
    plt.plot(m_evaporation_mono, f_max_evaporation_mono, color='k', alpha=0.25, linewidth=4, label='Extracted (Carr 21)')

    for m_star in np.array([4e14, 5e14]) / 1.989e33:
        plt.plot(m_evaporation_mono, constraint_mono_analytic(m_evaporation_mono), label='$M_* = {:.1e}$ g, $\epsilon$={:.2f}'.format(m_star*1.989e33, epsilon), linestyle = 'dotted', linewidth=6)

    for m_star in np.array([5.1e14]) / 1.989e33:
        epsilon= 0.43
        plt.plot(m_evaporation_mono, constraint_mono_analytic(m_evaporation_mono), label='$M_* = {:.1e}$ g, $\epsilon$={:.2f}'.format(m_star*1.989e33, epsilon), linestyle = 'dotted', linewidth=6)

    
    plt.plot(m_evaporation_mono, f_evap(m_evaporation_mono), linestyle='dashed', label='Interpolated')
    
    plt.xlabel('$M~[M_\odot]$')
    plt.ylabel('$f_\mathrm{PBH}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.title('Monochromatic')
    plt.tight_layout()
    """

