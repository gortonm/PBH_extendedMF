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
mc_evaporation = 10**np.linspace(-18, -15, 100)
m_evaporation_mono, f_max_evaporation_mono = load_data('Gamma-ray_mono.csv')
mc_evaporation_LN, f_pbh_evaporation_LN = load_data('Gamma-ray_LN.csv')


# approximate the evaporation constraints using a power law
power = (np.log10(f_max_evaporation_mono[-1]) - np.log10(f_max_evaporation_mono[0])) / (np.log10(m_evaporation_mono[-1]) - np.log10(m_evaporation_mono[0]))
prefactor = f_max_evaporation_mono[0] * (m_evaporation_mono[0]) ** (-power)
prefactor2 = f_max_evaporation_mono[-1] * (m_evaporation_mono[-1]) ** (-power)

m_range = 10**np.arange(-20, -10, 0.05)
f_max_evaporation_mono_extrapolated = prefactor * np.power(m_range, power)

# Range of characteristic masses in log-normal mass function for the power-law approximation
mc_evaporation_extrapolated = 10**np.linspace(-20, -10, 100)

# Subaru-HSC constraints
mc_subaru = 10**np.linspace(-11.5, -4.5, 100)
m_subaru_mono, f_max_subaru_mono = load_data('Subaru-HSC_mono.csv')
mc_subaru_LN, f_pbh_subaru_LN = load_data('Subaru-HSC_LN.csv')


# calculate constraints for extended MF from evaporation
f_pbh_evap = []
f_pbh_evap_extrapolated = []
n_max = 1000

for m_c in mc_evaporation:
    
    def f_constraint_function_evap(f_pbh):
        return left_riemann_sum(integrand(m_evaporation_mono, m_c, f_max_evaporation_mono, f_pbh), m_evaporation_mono) - 1
    f_pbh_evap.append(findroot(f_constraint_function_evap, 1, 1e-4, tolerance = 1e-8, n_max = n_max))

for m_c in mc_evaporation_extrapolated:
    def f_constraint_function_evap_extrapolated(f_pbh):
        return left_riemann_sum(integrand(m_range, m_c, f_max_evaporation_mono_extrapolated, f_pbh), m_range) - 1

    f_pbh_evap_extrapolated.append(findroot(f_constraint_function_evap_extrapolated, 1, 1e-4, tolerance = 1e-8, n_max = n_max))

# calculate constraints for extended MF from Subaru-HSC
f_pbh_subaru = []
for m_c in mc_subaru:
    
    def f_constraint_function_subaru(f_pbh):
        return left_riemann_sum(integrand(m_subaru_mono, m_c, f_max_subaru_mono, f_pbh), m_subaru_mono) - 1
   
    f_pbh_subaru.append(findroot(f_constraint_function_subaru, 1, 1e-4, tolerance = 1e-8, n_max = n_max))


# Test plot
plt.figure()
plt.plot(m_evaporation_mono, f_max_evaporation_mono, label='Evaporation (extracted)', color='purple')
plt.plot(m_subaru_mono, f_max_subaru_mono, label='Subaru-HSC (extracted)', color='tab:blue')
plt.plot(m_range, f_max_evaporation_mono_extrapolated, label='Evaporation (extrapolated)', color='y', linestyle='dotted')

plt.xlabel('$M_\mathrm{PBH}~[M_\odot]$')
plt.ylabel('$f_\mathrm{PBH}$')
plt.title('Monochromatic')
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-4, 1)
plt.legend()
plt.savefig('Extracted_constraints.png')

# Test plot
plt.figure(figsize=(6,5))
plt.plot(mc_evaporation[:-1], f_pbh_evap[:-1], label='Computed')
plt.plot(mc_evaporation_LN, f_pbh_evaporation_LN, label='Extracted')
plt.plot(mc_evaporation, f_pbh_evap_extrapolated, label='Computed (power-law fit)')
plt.xlabel('$M_c ~[M_\odot]$')
plt.ylabel('$f_\mathrm{PBH}$')
plt.title('Log-normal $(\sigma = 2)$')
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-4, 1)
plt.legend(fontsize='small')
plt.savefig('./Figures/evaporation_initial.png')

plt.figure(figsize=(6,5))
plt.plot(mc_subaru, f_pbh_subaru, label='Computed', color='tab:orange')
plt.plot(mc_subaru_LN[2:-5], f_pbh_subaru_LN[2:-5], label='Extracted', color='tab:blue')
plt.xlabel('$M_c ~[M_\odot]$')
plt.ylabel('$f_\mathrm{PBH}$')
plt.title('Log-normal $(\sigma = 2)$')
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-4, 1)
plt.legend()
plt.savefig('./Figures/HSC_initial.png')
