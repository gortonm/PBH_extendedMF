#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 13:17:49 2022

@author: ppxmg2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Specify the plot style
mpl.rcParams.update({'font.size': 24,'font.family':'serif'})
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


def approx(E, m_pbh):
    T_BH = 1.06 * (1e13 / m_pbh)
    return E**2 / (np.exp(E/T_BH) + 1)

E_values = 10**np.linspace(-4, np.log10(5), 50)
m_pbh_values = np.array([0.1, 0.4, 1.0, 3, 10]) * 10**16

for m_pbh in m_pbh_values:
    plt.plot(E_values*m_pbh, m_pbh*approx(E_values, m_pbh), label='{:.0e}'.format(m_pbh))
    plt.xlabel('$ME$ [GeV g]')
    plt.ylabel('$\mathrm{d}^2 N_{e^\pm} / (\mathrm{d}t~\mathrm{d}E_{e^\pm})$ [s$^{-1}$ GeV$^{-1}$]')
    #plt.xscale('log')
    plt.yscale('log')
    #plt.ylim(1e-9, 1e-4)
    plt.legend(title='$M_\mathrm{PBH}$ [g]')
