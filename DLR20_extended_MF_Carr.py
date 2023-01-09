#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 15:14:26 2022

@author: ppxmg2
"""

# Reproduce constraints for the log-normal mass function from Dasgupta,
# Laha & Ray (2020), using the method by Carr et al. (2017)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from loadBH import read_blackhawk_spectra, load_data, read_col
from tqdm import tqdm
import os

# Specify the plot style
mpl.rcParams.update({'font.size': 24, 'font.family':'serif'})
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

sigma = 1.0

def log_normal_MF(f_pbh, m, m_c):
    return f_pbh * np.exp(-np.log(m/m_c)**2 / (2*sigma**2)) / (np.sqrt(2*np.pi) * sigma * m)

def integrand(f_pbh, m, m_c):
    return log_normal_MF(f_pbh, m, m_c) / f_evap(m)

def findroot(f, a, b, tolerance_1, tolerance_2, n_max):
    n = 1
    while n <= n_max:
        c = (a + b) / 2
        
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

file_path_extracted = "./Extracted_files/"
n_max = 1000

mc_DLR20 = 10**np.linspace(15, 19, 100)

m_DLR20_mono, f_max_DLR20_mono = load_data('DLR20_Fig2_a__0_newaxes_2.csv')
mc_DLR20_LN, f_pbh_DLR20_LN = load_data("DLR20_Fig2_LN_sigma={:.1f}.csv".format(sigma))


def f_evap(m):
    return np.interp(m, m_DLR20_mono, f_max_DLR20_mono)

def constraint_function(f_pbh):
    return np.trapz(integrand(f_pbh, m_range, m_c), m_range) - 1

if "__main__" == __name__:
    
    plt.figure(figsize=(7, 6))
    plt.plot(m_DLR20_mono, f_max_DLR20_mono, label='Extracted')
    plt.plot(m_DLR20_mono, f_evap(m_DLR20_mono), 'x', label='Interpolated', linestyle='none')
    plt.xlabel('$M~[M_\odot]$')
    plt.ylabel('$f_\mathrm{PBH}$')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
   
    # calculate constraints for extended MF from evaporation
    f_pbh_DLR20 = []
    f_pbh_DLR20_rootfinder = []
    
    for m_c in mc_DLR20:
        
        m_range = m_DLR20_mono
        f_pbh_DLR20.append(1/np.trapz(integrand(f_pbh=1, m=m_DLR20_mono, m_c=m_c), m_DLR20_mono))
        f_pbh_DLR20_rootfinder.append(findroot(constraint_function, 1e-5, 10, tolerance_1=1e-4, tolerance_2=1e-4, n_max=1000))
        
    plt.figure(figsize=(7, 6))
    plt.plot(mc_DLR20_LN, f_pbh_DLR20_LN, label='Extracted \n (DLR (2020))')
    plt.plot(mc_DLR20, f_pbh_DLR20, label='Calculated \n (Carr et al. method)', linestyle='dashed')
    #plt.plot(mc_DLR20, f_pbh_DLR20_rootfinder, label='Calculated (rootfinder)', linestyle='dashed')

    plt.xlabel('$\mu_\mathrm{PBH}~[\mathrm{g}]$')
    plt.ylabel('$f_\mathrm{PBH}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(fontsize='small')
    plt.xlim(1e15, 1e19)
    plt.ylim(1e-4, 1)
    plt.title("Log-normal ($\sigma={:.1f}$)".format(sigma))
    plt.tight_layout()
    #plt.savefig('Figures/DLR20_LN_Carr.pdf')
    