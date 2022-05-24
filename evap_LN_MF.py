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
m1 = m_star

def log_normal_MF(m, m_c):
    return np.exp(-np.log(m/m_c)**2 / (2*sigma**2)) / (np.sqrt(2*np.pi) * sigma * m)

def load_data(filename):
    return np.genfromtxt(filepath+filename, delimiter=',', unpack=True)

def integrand(m, m_c):
    return log_normal_MF(m, m_c) / f_evap(m)    

def constraint_mono_analytic(m):
    return 2e-8 * np.array(m/m_star)**(3+epsilon)

def integral_analytic(m_c, m):
    return erf((((epsilon + 3) * sigma**2) + np.log(m/m_c))/(np.sqrt(2) * sigma)) 

def constraint_analytic(m_range, m_c):
    prefactor = 2e-8 * (m_c / m_star)**(3+epsilon) * np.exp(-0.5 * (epsilon + 3)**2 * sigma**2) 
    return prefactor / (integral_analytic(m_c, max(m_range)) - integral_analytic(m_c, min(m_range)))

m_c_evaporation = 10**np.linspace(-20, -10, 100)
m_evaporation_mono, f_max_evaporation_mono = load_data('Gamma-ray_mono.csv')

def f_evap(m):
    return np.interp(m, np.array(m_evaporation_mono), f_max_evaporation_mono)


if "__main__" == __name__:

    # Plot constraints for a log-normal MF
   
    plt.figure(figsize=(12,8))
    m_c_evaporation_LN, f_pbh_evaporation_LN = load_data('Gamma-ray_LN.csv')

    f_pbh_evap = []
    f_pbh_evap_analytic = []
    for m_c in m_c_evaporation:

        m_range = 10**np.linspace(np.log10(max(m1, m_star)), np.log10(m2), 10000)
                
        f_pbh_evap.append(1/np.trapz(integrand(m=m_range, m_c=m_c), m_range))
        f_pbh_evap_analytic.append(constraint_analytic(m_range, m_c))
        
    plt.plot(m_c_evaporation, f_pbh_evap, label='Trapezium rule', linestyle = 'dotted', linewidth=6)
    plt.plot(m_c_evaporation, f_pbh_evap_analytic, label='Analytic', linestyle = 'dotted', linewidth=6)
    plt.plot(m_c_evaporation_LN, f_pbh_evaporation_LN, color='k', alpha=0.25, linewidth=4, label='Extracted (Carr 21)')

    plt.xlabel('$M_\mathrm{c}~[M_\odot]$')
    plt.ylabel('$f_\mathrm{PBH}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.ylim(10**(-4), 1)
    plt.title('Log-normal ($\sigma = {:.0f}$)'.format(sigma))
    plt.tight_layout()

    
    if "__main__" == __name__:

        # Plot constraints for a monochromatic MF
       
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
        

