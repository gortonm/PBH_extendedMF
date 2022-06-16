#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 18:25:20 2022

@author: ppxmg2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Script to reproduce plots from Auffinger '22 (2206.02672)

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

def load_data(filename):
    return np.genfromtxt(filepath+filename, delimiter=',', unpack=True)

m_D20, beta_D20 = load_data('Auffinger21_Fig10_D20.csv')
m_A20_LN, beta_A20_LN = load_data('Auffinger21_Fig12_EXGB_LN.csv')
m_GC, beta_GC = load_data('Auffinger21_Fig14_GC.csv')

plt.plot(m_D20, beta_D20, label = "Dasgupta+ '20")
plt.plot(m_A20_LN, beta_A20_LN, label = "A20 (LN)")
plt.plot(m_GC, beta_GC , label = "GC (Fig. 14)")
#plt.xlabel("$M_\mathrm{PBH}$ [g]")
#plt.ylabel("$\beta$'")
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-28, 1e-12)
plt.xlim(1e8, 1e18)
plt.tight_layout()
plt.legend()