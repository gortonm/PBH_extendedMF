#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 10:42:31 2023

@author: ppxmg2
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Specify the plot style
mpl.rcParams.update({'font.size': 16,'font.family':'serif'})
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

def log_triangle(m_values, m_p, n):
    psi_values = []
    
    b = np.power(m_p, -n-1) / (1/(n+1) + (1/(n-1)))
    print((1/(n+1) + (1/(n-1))))
    d = b * np.power(m_p, 2*n)
    
    for m in m_values:
        if m < m_p:
            psi_values.append(b * np.power(m, n))
        else:
            psi_values.append(d * np.power(m, -n))
            
    return psi_values

def f_max(m):
    return 1e-25 * np.power(m, 2)

n = 0.5

mp_values = np.logspace(18, 22, 50)
m_values = np.logspace(16, 24, 1000)
f_PBH = [1 / np.trapz(log_triangle(m_values, m_p, n) / f_max(m_values), m_values) for m_p in mp_values]

fig, ax = plt.subplots(figsize=(6,6))
ax.plot(mp_values, f_max(mp_values), color="tab:grey", label="Delta func.")
ax.plot(mp_values, f_PBH, color="tab:blue", marker="x", linestyle="None", label="Triangle MF")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("$m~[\mathrm{g}]$")
ax.set_ylabel("$f_\mathrm{PBH}$")
ax.set_xlim(min(mp_values), max(mp_values))
fig.tight_layout()

print(f_max(mp_values) / f_PBH)

#%%
n=3
m_p = 1e20
print("Integral of psi_N = {:.5e}".format(np.trapz(log_triangle(m_values, m_p, n), m_values)))
fig1, ax1 = plt.subplots(figsize=(6,6))
ax1.plot(m_values, log_triangle(m_values, m_p, n), color="tab:grey", label="Delta func.")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("$m~[\mathrm{g}]$")
ax1.set_ylabel("$\psi_\mathrm{N}~[\mathrm{g}]^{-1}$")
ax1.set_xlim(min(m_values), max(m_values))
fig1.tight_layout()
