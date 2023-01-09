#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 11:46:30 2022
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


sigma = 2
filepath = './Extracted_files/'


def log_normal_MF(f_pbh, m, m_c):
    return f_pbh * np.exp(-np.log(m/m_c)**2 / (2*sigma**2)) / (np.sqrt(2*np.pi) * sigma * m)

def left_riemann_sum(y, x):
    return np.sum(np.diff(x) * y[:-1])

def integrand(m, m_c, f_max, f_pbh):
    integrand = []
    for i in range(len(m)):
        integrand.append(log_normal_MF(f_pbh, m[i], m_c) / f_max[i])
    return integrand

def integral_2(m, m_c, f_max, f_pbh):
    #m_range = 10**np.linspace(np.log10(min(m)), np.log10(max(m)), 1000)
    m_range = np.linspace(min(m), max(m), 5000)
    m_range = m
    f_max_func = np.interp(m_range, m, f_max)
    return left_riemann_sum(log_normal_MF(f_pbh, m_range, m_c) / f_max_func, m_range)

def load_data(filename):
    return np.genfromtxt(filepath+filename, delimiter=',', unpack=True)

def findroot(f, a, b, tolerance, n_max):
    n = 1
    while n <= n_max:
        c = (a + b) / 2
        #print(c)
        #print(f(c))

        if f(c) == 0 or abs((b - a) / 2) < tolerance:
            return c
            break
        n += 1

        # set new interval
        if np.sign(f(c)) == np.sign(f(a)):
            a = c
        else:
            b = c
    print("Method failed")


# Evaporation constraints (from gamma-rays)
mc_evaporation = 10**np.linspace(-18, -12, 100)
m_evaporation_mono, f_max_evaporation_mono = load_data('Gamma-ray_mono.csv')
mc_evaporation_LN, f_pbh_evaporation_LN = load_data('Gamma-ray_LN.csv')

m_range = 10**np.arange(-20, -10, 0.1)

# Subaru-HSC constraints
mc_subaru = 10**np.linspace(-12, -4, 100)
m_subaru_mono, f_max_subaru_mono = load_data('Subaru-HSC_mono.csv')
mc_subaru_LN, f_pbh_subaru_LN = load_data('Subaru-HSC_LN.csv')


# calculate constraints for extended MF from evaporation
f_pbh_evap = []
f_pbh_evap_extrapolated = []
n_max = 1000
for m_c in mc_evaporation:

    def f_constraint_function_evap(f_pbh):
        return integral_2(m_evaporation_mono, m_c, f_max_evaporation_mono, f_pbh) - 1

    f_pbh_evap.append(findroot(f_constraint_function_evap, 1, 1e-4, tolerance = 1e-8, n_max = n_max))

# calculate constraints for extended MF from Subaru-HSC
f_pbh_subaru = []
for m_c in mc_subaru:

    def f_constraint_function_subaru(f_pbh):
        return integral_2(m_subaru_mono, m_c, f_max_subaru_mono, f_pbh) - 1

    f_pbh_subaru.append(findroot(f_constraint_function_subaru, 1, 1e-4, tolerance = 1e-8, n_max = n_max))


# Test plot
plt.figure()
plt.plot(m_evaporation_mono, f_max_evaporation_mono, label='Evaporation (extracted)', color='k', alpha=0.25)
plt.plot(m_subaru_mono, f_max_subaru_mono, label='Subaru-HSC (extracted)', color='tab:blue')

plt.xlabel('$M_\mathrm{PBH}~[M_\odot]$')
plt.ylabel('$f_\mathrm{PBH}$')
plt.title('Monochromatic')
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-4, 1)
plt.legend()


# Test plot
f_pbh_evap = np.array(f_pbh_evap)
f_pbh_evap_plotting = f_pbh_evap[f_pbh_evap > 1e-3]
mc_evaporation = mc_evaporation[f_pbh_evap > 1e-3]

plt.figure()
plt.plot(mc_evaporation, f_pbh_evap_plotting, label='Evaporation (computed)', color='k', linewidth=4, linestyle='dotted')
#plt.plot(mc_evaporation, f_pbh_evap_extrapolated, label='Evaporation (extracted, computed)', color='k')
plt.plot(mc_evaporation_LN, f_pbh_evaporation_LN, label='Evaporation (extracted)', color='k', linewidth=2, alpha=0.25)

f_pbh_subaru = np.array(f_pbh_subaru)
f_pbh_subaru_plotting = f_pbh_subaru[f_pbh_subaru>1e-3]
mc_subaru = np.array(mc_subaru)[f_pbh_subaru>1e-3]

plt.plot(mc_subaru_LN, f_pbh_subaru_LN, label='Subaru-HSC (extracted)', color='tab:blue', linewidth=2)
plt.plot(mc_subaru, f_pbh_subaru_plotting, label='Subaru-HSC (computed)', color='tab:orange', linewidth=4, linestyle='dotted')
plt.xlabel('$M_c ~[M_\odot]$')
plt.ylabel('$f_\mathrm{PBH}$')
plt.title('Log-normal $(\sigma = 2)$')
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-4, 1)
plt.legend()