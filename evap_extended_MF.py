#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 13:32:36 2022

@author: ppxmg2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cProfile
import pstats

from reproduce_extended_MF import log_normal_MF, findroot

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
    return log_normal_MF(f_pbh, m, m_c) / f_evap(m)

def f_constraint_function_evap(f_pbh):
    #print(m_c)
    return np.trapz(integrand(m_range, m_c, f_pbh), m_range) - 1

n_max = 1000

if "__main__" == __name__:
        
    mc_evaporation = 10**np.linspace(-18, -11, 100)
    m_evaporation_mono, f_max_evaporation_mono = load_data('Gamma-ray_mono.csv')
    mc_evaporation_LN, f_pbh_evaporation_LN = load_data('Gamma-ray_LN.csv')
        
    m_range = 10**np.linspace(-20, 0, 10000)
    
    # calculate constraints for extended MF from evaporation
    f_pbh_evap = []
    
    for m_c in mc_evaporation:
        print(m_c)
        f_pbh_evap.append(findroot(f_constraint_function_evap, 1, 1e-6, tolerance = 1e-10, n_max = n_max))
    
    plt.figure()
    plt.plot(mc_evaporation_LN, f_pbh_evaporation_LN, label='Extracted')
    plt.plot(mc_evaporation, f_pbh_evap, label='Calculated')
    plt.xlabel('$M_\mathrm{c}~[M_\odot]$')
    plt.ylabel('$f_\mathrm{PBH}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.ylim(1e-4, 1)
    plt.tight_layout()
