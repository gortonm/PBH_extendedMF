#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 16:16:11 2022

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

f_pbh = 1

m_c = 1e-13   # choose a characteristic PBH mass in the centre of the asteroid-mass region

m_min = 1e-15
m_max = 1e-10
gamma = 0.5

m_f = 1e-10

def log_normal(m, sigma):
    return f_pbh * np.exp(-np.log(m/m_c)**2 / (2 * sigma**2)) / (np.sqrt(2*np.pi) * sigma * m)

def power_law(m):
    # needs normalisation factor
    return f_pbh * gamma * m**(gamma - 1) / (m_max**gamma - m_min**gamma)

def critical_collapse(m):
    # needs normalisation factor
    return m**(2.85) * np.exp(-(m/m_f)**2.85)

m = 10**np.arange(-18, -8, 0.1)

colors = ['tab:blue', 'tab:orange', 'tab:green']
plt.figure()
y_min = 1e9
y_max = 1e15
plt.yscale('log')
plt.xscale('log')
for i, sigma in enumerate([1, 2, 3]):
    plt.plot(m, log_normal(m, sigma), label='$\sigma = {:.0f}$'.format(sigma), color=colors[i])
    #plt.axvline(m_c * np.exp(sigma**2), y_min, y_max, linestyle='dotted', label='$M_c e^{\sigma^2}$', color=colors[i])
plt.axvline(m_c, y_min, y_max, linestyle='dotted', label='$M_c$', color='k')
plt.ylim(y_min, y_max)
plt.legend()
plt.ylabel('$\psi(M) ~ [M_\odot^{-1}]$')
plt.xlabel('$M_\mathrm{PBH} [M_\odot]$')

plt.figure()
for sigma in (1, 2, 3):
    plt.plot(m, m*log_normal(m, sigma), label='$\sigma = {:.0f}$'.format(sigma))
plt.yscale('log')
plt.xscale('log')
y_min = 1e-7
y_max = 1
plt.axvline(m_c, y_min, y_max, linestyle='dotted', label='$M_c$', color='k')
plt.ylim(y_min, y_max)
plt.legend()
plt.ylabel('$M\psi(M)$')
plt.xlabel('$M_\mathrm{PBH} [M_\odot]$')

