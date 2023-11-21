#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 13:36:46 2023

@author: ppxmg2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from preliminaries import load_data
from plot_constraints import frac_diff

# Specify the plot style
mpl.rcParams.update({'font.size': 24,'font.family': 'serif'})
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
plt.style.use('tableau-colorblind10')


"""Unmodified version of BlackHawk"""
# Path to BlackHawk and Isatis
BlackHawk_path_unmodified = os.path.expanduser('~') + "/Downloads/version_finale_unmodified/"
tables_path_unmodified = BlackHawk_path_unmodified + "src/tables/fM_tables/"

data_table_unmodified = np.genfromtxt(tables_path_unmodified+"fM.txt", unpack=False)
g_to_GeV = 5.61e35 * 1e-3 * 1e-9  # from 2110.12251

# with this loading, data_talbe[0] = range of dimensionless spins a*
# np.transpose(data_table)[0] / g_to_GeV might be masses [in grams]

masses_poss_unmodified = np.transpose(data_table_unmodified)[0] / g_to_GeV
fM_zero_spin_unmodified = np.transpose(data_table_unmodified)[2]


"""Modified version of BlackHawk"""
# Path to BlackHawk and Isatis

BlackHawk_path = os.path.expanduser('~') + "/Downloads/version_finale/"
#tables_path = BlackHawk_path + "scripts/greybody_scripts/fM/"
#data_table = np.genfromtxt(tables_path+"fM_QCD_modified.txt", unpack=False)
tables_path = BlackHawk_path + "src/tables/fM_tables/"
data_table = np.genfromtxt(tables_path+"fM_nograv.txt", unpack=False)
g_to_GeV = 5.61e35 * 1e-3 * 1e-9  # from 2110.12251

# with this loading, data_table[0] = range of dimensionless spins a*
# np.transpose(data_table)[0] / g_to_GeV might be masses [in grams]

masses_poss = np.transpose(data_table)[0] / g_to_GeV
fM_zero_spin = np.transpose(data_table)[2]


"""From Carr, Kohri, Sendouda & Yokoyama (2016) [1604.05349]"""
m_CKSY, fM_CKSY = load_data("1604.05349/1604.05349_Fig1.csv")

"""alpha_eff from Mosbech & Picker (2023)"""
m_MP23, alpha_eff_MP23 = load_data("2203.05743/2203.05743_Fig1.csv")
f_MP23 = alpha_eff_MP23 / min(alpha_eff_MP23)

"""why is the min value so small? = 0.001 """
# normalise to the value at >> 1e17 g (as in CKSY '16)
fM_zero_spin_normalised_unmodified = fM_zero_spin_unmodified / min(fM_zero_spin_unmodified[fM_zero_spin_unmodified > 1])
fM_zero_spin_normalised = fM_zero_spin / min(fM_zero_spin[fM_zero_spin > 1])

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(masses_poss_unmodified, fM_zero_spin_normalised_unmodified, label="BlackHawk")
ax.plot(m_CKSY, fM_CKSY, label="Carr et al. (2016) Fig. 1 [1604.05349]")
ax.plot(m_MP23, f_MP23, label="Mosbech \& Picker (2022) Fig. 1 [2203.05743]", color="r") 
ax.set_xscale("log")
ax.set_xlabel("$M~[\mathrm{g}]$")
ax.set_ylabel("$f(M)$")
ax.set_xlim(1e10, 1e18)
ax.legend(fontsize="small")
ax.grid()
x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
ax.xaxis.set_major_locator(x_major)
x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 5)
ax.xaxis.set_minor_locator(x_minor)
ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
fig.tight_layout()

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(m_MP23, frac_diff(f_MP23, fM_zero_spin_normalised_unmodified, m_MP23, masses_poss_unmodified))
ax.set_xscale("log")
ax.set_xlabel("$M~[\mathrm{g}]$")
ax.set_ylabel(r"$\alpha_\mathrm{eff} / \alpha(M) - 1$")
ax.set_xlim(1e10, 1e18)
ax.grid()
x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
ax.xaxis.set_major_locator(x_major)
x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 5)
ax.xaxis.set_minor_locator(x_minor)
ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
fig.tight_layout()

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(masses_poss_unmodified, fM_zero_spin_normalised_unmodified, label="BlackHawk")
ax.plot(m_CKSY, fM_CKSY, label="Carr et al. (2016) Fig. 1 [1604.05349]") 
ax.plot(masses_poss, fM_zero_spin_normalised, linestyle="dotted", linewidth=3, label="BlackHawk ($m_u = m_d = m_s = 300~\mathrm{MeV}$)", color="k")
ax.set_xscale("log")
ax.set_xlabel("$M~[\mathrm{g}]$")
ax.set_ylabel("$f(M)$")
ax.set_xlim(1e10, 1e18)
ax.legend(fontsize="small")
fig.tight_layout()