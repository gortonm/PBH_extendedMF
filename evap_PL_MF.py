#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 23:01:53 2022

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

m_star = 4e14 / 1.989e33    # use value of M_* from Carr+ '17
gamma = 1
epsilon = 0.4

m_min = m_star

m_2 = np.power(5e9, 1/(3+epsilon)) * m_star



def load_data(filename):
    return np.genfromtxt(filepath+filename, delimiter=',', unpack=True)

def power_law_MF(m, m_max):
    return (gamma / (m_max**gamma - m_min**gamma)) * m**(gamma-1)

def f_evap(m, epsilon=0.4):    # Eq. 62 of Carr+ '21
    return 2e-8 * np.array(m/m_star)**(3+epsilon)

def integrand(m, m_max):
    return power_law_MF(m, m_max) / f_evap(m)

def constraint_analytic(m_range, m_max, epsilon):
    m2 = max(m_range)
    m1 = min(m_range)
    return 2e-8 * ((m_max**gamma - m_min**gamma) / gamma) * (gamma - (3+epsilon)) * (m2**(gamma-(3+epsilon)) - m1**(gamma-(3+epsilon)) )**(-1) / (m_star**(3+epsilon))

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
    mc_evaporation_PL, f_pbh_evaporation_PL = load_data('PL_gamma=1_evap.csv')

    # calculate constraints for extended MF from evaporation
    f_pbh_evap = []
    f_pbh_evap_analytic = []
    
    for m_c in mc_evaporation:
        m_max=m_c*np.exp(1/gamma)
        
        m_range = 10**np.linspace(max(np.log10(2*m_star), np.log10(m_max) - 20), np.log10(m_2), 100000)
        
        print(m_max)
        f_pbh_evap_analytic.append(constraint_analytic(m_range=m_range, m_max=m_max, epsilon=0.4))
        f_pbh_evap.append(1/np.trapz(integrand(m_range, m_max), m_range))
        
    plt.figure()
    plt.plot(mc_evaporation_PL, f_pbh_evaporation_PL, label='Extracted (Carr+ 21)')
    
    #alpha, beta = power_law_fit(mc_evaporation_LN, f_pbh_evaporation_LN)
    #print(alpha)
    #print(beta)
    #plt.plot(mc_evaporation_LN, (beta * mc_evaporation_LN**alpha), label='Power law fit')

    plt.plot(mc_evaporation, f_pbh_evap, label='Calculated')
    plt.plot(mc_evaporation, np.array(f_pbh_evap_analytic), label='Calculated (analytic)', linestyle='dotted')

    plt.xlabel('$M_\mathrm{c}~[M_\odot]$')
    plt.ylabel('$f_\mathrm{PBH}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.ylim(1e-4, 1)
    plt.tight_layout()
