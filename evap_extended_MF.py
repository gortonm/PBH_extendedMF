#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 13:32:36 2022

@author: ppxmg2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from reproduce_extended_MF import log_normal_MF

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

m_star = 5.1e14 / 1.989e33
sigma = 2

def load_data(filename):
    return np.genfromtxt(filepath+filename, delimiter=',', unpack=True)

def f_evap(m, epsilon=0.4):    # Eq. 62 of Carr+ '21
    return 2e-8 * np.array(m/m_star)**(3+epsilon)

def integrand(m, m_c, f_pbh):
    #print(log_normal_MF(f_pbh, m, m_c) )
    return log_normal_MF(f_pbh, m, m_c) / f_evap(m)

def f_constraint_function_evap(f_pbh):
    #print(np.trapz(integrand(m_range, m_c, f_pbh), m_range) - 1)
    return np.trapz(integrand(m_range, m_c, f_pbh), m_range) - 1

def constraint_analytic(m_c, epsilon):
    return 2e-8 * (m_c / m_star)**(3+epsilon) * np.exp(-0.5*sigma**2*(epsilon+3)**2)

def findroot(f, a, b, tolerance_1, tolerance_2, n_max):
    n = 1
    while n <= n_max:
        c = (a + b) / 2
        #print('\n' + 'New:')
        #print('c = ', c)
        #print('f(c) = ', f(c))
        
        if abs(f(c)) < tolerance_1 or abs((b - a) / 2) < tolerance_2:
        #if abs(f(c)) < tolerance_1:
            return c
            break
        n += 1
        
        # set new interval
        if np.sign(f(c)) == np.sign(f(a)):
            a = c
        else:
            b = c
    print("Method failed")


n_max = 100

def power_law_fit(x, y):
    alpha = np.log10(y[-1] / y[1]) / np.log10(x[-1] / x[1])
    beta = y[5] * x[5]**(-alpha)
    return alpha, beta

if "__main__" == __name__:
        
    mc_evaporation = 10**np.linspace(-17, -11, 100)
    m_evaporation_mono, f_max_evaporation_mono = load_data('Gamma-ray_mono.csv')
    mc_evaporation_LN, f_pbh_evaporation_LN = load_data('Gamma-ray_LN.csv')
    mc_evaporation_LN_Carr17, f_pbh_evaporation_LN_Carr17 = load_data('Carr_17_LN_evap.csv')

    # calculate constraints for extended MF from evaporation
    f_pbh_evap = []
    f_pbh_evap_analytic = []
    
    for m_c in mc_evaporation:
        
        m_range = 10**np.linspace(min(m_star, np.log10(m_c) - 20), np.log10(m_c) + 20, 100000)
        
        print(m_c)
        #f_pbh_evap_analytic.append(0.3 * constraint_analytic(m_c=m_c, epsilon=-0.4))  # gives an oddly good fit
        f_pbh_evap_analytic.append(constraint_analytic(m_c=m_c, epsilon=0.4))
        f_pbh_evap.append(1/np.trapz(integrand(m_range, m_c, f_pbh=1), m_range))
        
    plt.figure()
    plt.plot(mc_evaporation_LN, f_pbh_evaporation_LN, label='Extracted (Carr+ 21)')
    plt.plot(mc_evaporation_LN_Carr17, f_pbh_evaporation_LN_Carr17, label='Extracted (Carr+ 17)')
    
    alpha, beta = power_law_fit(mc_evaporation_LN, f_pbh_evaporation_LN)
    print(alpha)
    print(beta)
    plt.plot(mc_evaporation_LN, (beta * mc_evaporation_LN**alpha) , label='Power law fit')

    plt.plot(mc_evaporation, f_pbh_evap, label='Calculated')
    plt.plot(mc_evaporation, np.array(f_pbh_evap_analytic), label='Calculated (analytic)', linestyle='dotted')

    plt.xlabel('$M_\mathrm{c}~[M_\odot]$')
    plt.ylabel('$f_\mathrm{PBH}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.ylim(1e-4, 1)
    plt.tight_layout()
