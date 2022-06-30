#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 16:33:37 2022

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

# Script to reproduce constraints on PBHs from (extra)galactic gamma rays, 
# using results from BlackHawk 

# Unit conversion factors
s_to_yr = 1 / (365.25 * 86400)   # convert 1 s to yr
cm_to_kpc = 3.2408e-22    # convert 1 cm to kpc
g_to_GeV = 5.61e23    # convert 1 gram to GeV / c^2



# Parameters 
r_0 = 8.12    # galactocentric radius of Sun, in kpc (Table I caption: Coogan, Morrison & Profumo '20 2010.04797)

# Parameters (NFW density profile)
rho_0 = 0.376   # local DM density [GeV / c^2 cm^{-3}] (Table 3 de Salas+ '19 1906.06133)
r_s = 11   # scale radius [kpc] (Table 3 de Salas+ '19 1906.06133)
gamma = 1   # inner slope

def rho_NFW(r, r_s):
    return rho_0 * (r_s/r)**gamma * (1 + (r/r_s))**(gamma-3)

def rho_Einasto(r, r_s):
    return rho_0 * np.exp( - (2/alpha) * ( (r/r_s)**alpha - 1))

def r(l, psi):
    return np.sqrt(l**2 + r_0**2 - 2*l*r_0*np.cos(psi))

def j_psi(psi, r_s, n_steps=10000, NFW=True):
    l_values = np.linspace(0, r_0, n_steps)
    if NFW:
        return np.trapz(rho_NFW(r(l_values, psi), r_s), l_values)
    else:
        return np.trapz(rho_Einasto(r(l_values, psi), r_s), l_values)

# unit conversion factor from units of integral output to those used in 2010.04797 [MeV cm^{-2} sr^{-1}]
unit_conversion_Coogan = 3.0857e24

delta_omega = 2.39246e-2    # 5 square degree observing region around the galactic centre


def j(delta_omega, r_s, n_steps=10000, NFW=True):
    lim = np.sqrt(delta_omega / np.pi) / 2   # integration limit on psi (in radians)
    psi_values = np.linspace(1e-6, lim, n_steps)
    integrand_values = [psi * j_psi(psi, r_s, NFW) for psi in psi_values]
    return (4 * np.pi / (delta_omega)) * np.trapz(integrand_values, psi_values)

print("NFW")
print(j(delta_omega, r_s) * unit_conversion_Coogan)


# Parameters (generalised NFW)
rho_0 = 0.387   # local DM density [GeV / c^2 cm^{-3}] (Table 3 de Salas+ '19 1906.06133)
r_s = 8.1   # scale radius [kpc] (Table 3 de Salas+ '19 1906.06133)
gamma = 1.2   # inner slope

print("Generalised NFW")
print(j(delta_omega, r_s) * unit_conversion_Coogan)


# Parameters (Einasto)
print("Einasto (varying r_s and alpha)")
rho_0 = 0.388   # local DM density [GeV / c^2 cm^{-3}] (Table 3 de Salas+ '19 1906.06133)
for r_s in [6.5, 9.2, 14.5]:
    for alpha in [0.09, 0.18, 0.39]:
        print(j(delta_omega, r_s, NFW=False) * unit_conversion_Coogan)
