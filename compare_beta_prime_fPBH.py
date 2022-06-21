#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 18:16:00 2022

@author: ppxmg2
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

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

file_path_extracted = './Extracted_files/'
def load_data(filename):
    return np.genfromtxt(file_path_extracted+filename, delimiter=',', unpack=True)

def f_Carr21(beta_prime, m):
    # use Eq. (57) from Carr+ '21
    return 3.81e8 * beta_prime / np.sqrt(m / 1.989e33)

m_pbh_D20_Fig2_LH, f_pbh_D20_Fig2_LH = load_data('Dasgupta+20_Fig2_LH.csv')

m_pbh_D20, beta_D20 = load_data('Auffinger21_Fig10_D20.csv')

f_D20 = f_Carr21(beta_D20, m_pbh_D20)

plt.figure(figsize=(6, 4))
plt.plot(m_pbh_D20_Fig2_LH, f_pbh_D20_Fig2_LH, label="Dasgupta '20 Fig. 2 (extracted)")
plt.plot(m_pbh_D20, f_D20, label="Auffinger '22 Fig. 10 \n (converted to $f_\mathrm{PBH}$)")
plt.ylabel('$f_\mathrm{PBH}$')
plt.xlabel('$M_\mathrm{PBH}$ [g]')
plt.tight_layout()
plt.legend(fontsize='small')
plt.xscale('log')
plt.yscale('log')
plt.savefig('./Figures/fPBH_beta_comparison.pdf')