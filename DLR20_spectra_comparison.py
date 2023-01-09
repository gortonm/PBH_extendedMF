#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 17:29:18 2022

@author: ppxmg2

Show how PYTHIA spectra are overpredicted at low PBH masses at low energies.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from loadBH import read_blackhawk_spectra
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

m_pbh_values = np.linspace(1.2, 1.9, 8) * 10**15

# path to BlackHawk spectra (PYTHIA)
file_path_data_base = os.path.expanduser("~") + "/Downloads/blackhawk_v1.2/results/1000_steps/DLR20_"


fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 11))
for i, m_pbh in tqdm(enumerate(m_pbh_values)):
    
    if i <= 3:
        ax = axes[0][i]
    if i > 3:
       ax = axes[1][i-4]

    print("\nM_PBH = {:.2e} g".format(m_pbh))

    exponent = np.floor(np.log10(m_pbh))
    coefficient = m_pbh / 10**exponent

    file_path_data = file_path_data_base + "{:.1f}e{:.0f}g/".format(coefficient, exponent)
    ep_energies, ep_spec = read_blackhawk_spectra(file_path_data + "instantaneous_primary_spectra.txt", col=7)
    
    spec_integral = np.trapz(ep_spec, ep_energies)
    print('Integral : ', spec_integral)
    
    ax.plot(ep_energies, ep_spec)
    ax.set_xlim(4e-4, 3e-3)
    ax.set_ylim(1e18, 1e21)
    #ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('$M_\mathrm{PBH}$' + '= {:.1e} g'.format(m_pbh), fontsize='small')
    if i > 3:
        ax.set_xlabel('$E$ [GeV]')
    if i % 4 == 0:
        ax.set_ylabel('$\mathrm{d}^2 N_{e^\pm} / (\mathrm{d}t~\mathrm{d}E_{e^\pm})$ [s$^{-1}$ GeV$^{-1}$]')

    # print value of integral over spectrum, from 5.11e-4 GeV to 3 MeV
        
        
        
# add an invisible axis, for the axis labels that apply for the whole figure
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
fig.tight_layout()