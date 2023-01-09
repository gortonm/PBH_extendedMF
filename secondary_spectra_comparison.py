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

m_pbh_values = [1e15, 3e15, 5e15, 1e16, 5e16, 1e17]

# path to BlackHawk spectra (PYTHIA)
file_path_data_base_PYTHIA = os.path.expanduser("~") + "/Downloads/blackhawk_v1.2/results/1000_steps/DLR20_"
file_path_data_base_Hazma = os.path.expanduser("~") + "/Downloads/version_finale/results/1000_steps/DLR20_"


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 11))
for i, m_pbh in tqdm(enumerate(m_pbh_values)):
    
    if i <= 2:
        ax = axes[0][i]
    if i > 2:
       ax = axes[1][i-3]

    print("\nM_PBH = {:.2e} g".format(m_pbh))

    exponent = np.floor(np.log10(m_pbh))
    coefficient = m_pbh / 10**exponent

    file_path_data = file_path_data_base_PYTHIA + "{:.1f}e{:.0f}g/".format(coefficient, exponent)
    ep_energies_PYTHIA_prim, ep_spec_PYTHIA_prim = read_blackhawk_spectra(file_path_data + "instantaneous_primary_spectra.txt", col=7)
    ep_energies_PYTHIA, ep_spec_PYTHIA = read_blackhawk_spectra(file_path_data + "instantaneous_secondary_spectra.txt", col=2)

    file_path_data = file_path_data_base_Hazma + "{:.1f}e{:.0f}g/".format(coefficient, exponent)
    ep_energies_Hazma, ep_spec_Hazma = read_blackhawk_spectra(file_path_data + "instantaneous_secondary_spectra.txt", col=2)

    ax.plot(ep_energies_PYTHIA, ep_spec_PYTHIA, label='PYTHIA', color='r')
    ax.plot(ep_energies_Hazma, ep_spec_Hazma, label='Hazma', color='k', linewidth=3, alpha=0.5)
    ax.plot(ep_energies_PYTHIA_prim, ep_spec_PYTHIA_prim, label='Primary', color='tab:blue', linestyle='dotted', linewidth=4)
    ax.set_xlim(3e-4, 0.5)
    ax.set_ylim(1e18, 5e22)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('$M_\mathrm{PBH}$' + '= {:.0e} g'.format(m_pbh))
    if i > 2:
        ax.set_xlabel('$E$ [GeV]')
    if i % 3 == 0:
        ax.set_ylabel('$\mathrm{d}^2 N_{e^\pm} / (\mathrm{d}t~\mathrm{d}E_{e^\pm})$ [s$^{-1}$ GeV$^{-1}$]')

# add an invisible axis, for the axis labels that apply for the whole figure
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
fig.tight_layout()