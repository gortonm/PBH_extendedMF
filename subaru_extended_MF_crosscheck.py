#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 13:32:36 2022

@author: ppxmg2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Specify the plot style
mpl.rcParams.update({'font.size': 14,'font.family':'serif'})
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
sigma = 2


def log_normal_MF(f_pbh, m, m_c):
    return f_pbh * np.exp(-np.log(m/m_c)**2 / (2*sigma**2)) / (np.sqrt(2*np.pi) * sigma * m)

def load_data(filename):
    return np.genfromtxt(filepath+filename, delimiter=',', unpack=True)

def integrand(f_pbh, m, m_c):
    return log_normal_MF(f_pbh, m, m_c) / f_evap(m)

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


mc_subaru = 10**np.linspace(-15, -4, 100)
m_subaru_mono, f_max_subaru_mono = load_data('Subaru-HSC_mono.csv')
mc_subaru_LN, f_pbh_subaru_LN = load_data('Subaru-HSC_LN.csv')

def f_evap(m):
    return np.interp(m, m_subaru_mono, f_max_subaru_mono)

def constraint_function(f_pbh):
    return np.trapz(integrand(f_pbh, m_range, m_c), m_range) - 1

n_max = 100

if "__main__" == __name__:
    
    plt.figure()
    plt.plot(m_subaru_mono, f_max_subaru_mono, label='Extracted')
    plt.plot(m_subaru_mono, f_evap(m_subaru_mono), label='Interpolated')
    plt.xlabel('$M~[M_\odot]$')
    plt.ylabel('$f_\mathrm{PBH}$')
    plt.xscale('log')
    plt.yscale('log')
   
    # calculate constraints for extended MF from evaporation
    f_pbh_subaru = []
    f_pbh_subaru_rootfinder = []
    
    for m_c in mc_subaru:
        
        m_range = m_subaru_mono
        #f_pbh_evap.append(1/np.trapz(integrand(m_range, m_c, f_pbh=1), m_range))
        f_pbh_subaru.append(1/np.trapz(integrand(f_pbh=1, m=m_subaru_mono, m_c=m_c), m_subaru_mono))
        f_pbh_subaru_rootfinder.append(findroot(constraint_function, 1e-5, 10, tolerance_1=1e-4, tolerance_2=1e-4, n_max=1000))
        
    plt.figure(figsize=(5,4))
    plt.plot(mc_subaru_LN, f_pbh_subaru_LN, label='Extracted (Carr+ 21)')
    plt.plot(mc_subaru, f_pbh_subaru, label='Calculated', linestyle='dashed')
    #plt.plot(mc_subaru, f_pbh_subaru_rootfinder, label='Calculated (root-finder)', linestyle='dotted')

    plt.xlabel('$M_\mathrm{c}~[M_\odot]$')
    plt.ylabel('$f_\mathrm{PBH}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.xlim(10**(-12), 1e-4)
    plt.ylim(1e-3, 1)
    plt.title('Log-normal MF ($\sigma = 2$)')
    plt.tight_layout()
    plt.savefig('Figures/subaru_constraints_updated.pdf')