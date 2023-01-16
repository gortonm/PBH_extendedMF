#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 11:38:16 2023

@author: ppxmg2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from GC_plots_extended_MF import *

# Produce plots of the Subaru-HSC microlensing constraints on PBHs, for 
# extended mass functions, using the method from 1705.05567.

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

filepath = './Extracted_files/'

#%%
# Test: calculate constraints from Fig. 20 of 2002.12778.
# The monochromatic MF constraints used match Fig. 6 of 1910.01285.

mc_subaru = 10**np.linspace(-15, -4, 100)

# Load data files
m_subaru_mono, f_max_subaru_mono = load_data("Subaru-HSC_mono.csv")
m_subaru_mono_Smyth, f_max_subaru_mono_Smyth = load_data("Subaru-HSC_mono.csv")
mc_subaru_LN, f_pbh_subaru_LN = load_data("Subaru-HSC_LN.csv")

sigma = 2

if "__main__" == __name__:
    
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(m_subaru_mono, f_max_subaru_mono, label='Extracted (2002.12778)')
    ax.plot(m_subaru_mono_Smyth, f_max_subaru_mono_Smyth, linewidth=7, alpha=0.5, linestyle='dotted', label='Extracted (1910.01285)')
    ax.plot(m_subaru_mono, f_max(m_subaru_mono, m_subaru_mono, f_max_subaru_mono), linestyle='dotted', linewidth=4, label='Interpolated')
    ax.set_xlabel('$M_\mathrm{PBH}~[M_\odot]$')
    ax.set_ylabel('$f_\mathrm{PBH}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title("Monochromatic MF")
    ax.legend()
    fig.tight_layout()
   
    # Calculate constraints for extended MF from microlensing.
    f_pbh_subaru = []
    
    for m_c in mc_subaru:
        
        f_pbh_subaru.append(1/np.trapz(integrand(1, m_subaru_mono, m_c, sigma, m_subaru_mono, f_max_subaru_mono), m_subaru_mono))
        
    fig, ax = plt.subplots(figsize=(6,6))
    plt.plot(mc_subaru_LN, f_pbh_subaru_LN, label='Extracted (2002.12778)')
    plt.plot(mc_subaru, f_pbh_subaru, label='Calculated', linestyle='dashed')

    ax.set_xlabel('$M_\mathrm{c}~[M_\odot]$')
    ax.set_ylabel('$f_\mathrm{PBH}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.set_xlim(10**(-12), 1e-4)
    ax.set_ylim(1e-3, 1)
    ax.set_title("Log-normal MF ($\sigma = {:.1f}$)".format(sigma))
    fig.tight_layout()


#%%
# Calculate Subaru-HSC constraint for an extended MF, using the monochromatic
# MF constraint from 2007.12697.
# Sanity check for a very small sigma to compare to the monochromatic MF
# constraint.
# Also compare to the PBHbounds constraint

mc_subaru = 10**np.linspace(20, 29, 1000)

# Load data files
m_subaru_mono, f_max_subaru_mono = load_data("Croon2020_R90_0.csv")

# Load comparison constraint from PBHbounds
PBHbounds = np.transpose(np.genfromtxt("./../PBHbounds/PBHbounds/bounds/HSC.txt", delimiter=" ", skip_header=1))
print(PBHbounds)
m_subaru_mono_PBHbounds, f_max_subaru_mono_PBHbounds = np.transpose(np.genfromtxt("./../PBHbounds/PBHbounds/bounds/HSC.txt", delimiter=" ", skip_header=1))

sigma = 0.2

if "__main__" == __name__:
    
   
    # Calculate constraints for extended MF from microlensing.
    f_pbh_subaru = []
    
    for m_c in mc_subaru:

        f_pbh_subaru.append(1/np.trapz(integrand(1, m_subaru_mono, m_c, sigma, m_subaru_mono, f_max_subaru_mono), m_subaru_mono))
        
    fig, ax = plt.subplots(figsize=(6,6.5))
        
    ax.plot(mc_subaru, f_pbh_subaru, label="Calculated", )
    ax.plot(m_subaru_mono, f_max_subaru_mono, linestyle='dotted', label="Monochromatic (Croon et al. (2020))")
    ax.plot(m_subaru_mono_PBHbounds * 1.989e33, f_max_subaru_mono_PBHbounds, linestyle='dotted', linewidth=1.5, label="Monochromatic (PBHbounds)")
    ax.set_xlabel('$M_\mathrm{PBH}~[\mathrm{g}]$')
    ax.set_ylabel('$f_\mathrm{PBH}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.set_xlim(1e20, 1e29)
    ax.set_ylim(1e-3, 1)
    ax.set_title("Log-normal MF ($\sigma = {:.2f}$)".format(sigma) + ", $R_{90} = 0$")
    fig.tight_layout()


#%%
# Calculate Subaru-HSC constraint for an extended MF, using the monochromatic
# MF constraint from 2007.12697.

mc_subaru = 10**np.linspace(20, 29, 100)

# Load data files
m_subaru_mono, f_max_subaru_mono = load_data("Croon2020_R90_0.csv")

sigma = .5

if "__main__" == __name__:
   
    # Calculate constraints for extended MF from microlensing.
    f_pbh_subaru = []
    
    for m_c in mc_subaru:
        
        f_pbh_subaru.append(1/np.trapz(integrand(1, m_subaru_mono, m_c, sigma, m_subaru_mono, f_max_subaru_mono), m_subaru_mono))
        
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(mc_subaru, f_pbh_subaru, label="$R_{90} = 0$", )
    ax.set_xlabel('$M_\mathrm{c}~[\mathrm{g}]$')
    ax.set_ylabel('$f_\mathrm{PBH}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.set_xlim(1e20, 1e29)
    ax.set_ylim(1e-3, 1)
    ax.set_title("Log-normal MF ($\sigma = {:.1f}$)".format(sigma))
    fig.tight_layout()


#%%
# Calculate Subaru-HSC constraint for an extended MF, using the monochromatic
# MF constraint from 2007.12697. Plot sigma=0.5 and sigma=1.0 results on the 
# same figure.

mc_subaru = 10**np.linspace(20, 29, 100)

# Load data files
m_subaru_mono, f_max_subaru_mono = load_data("Croon2020_R90_0.csv")

if "__main__" == __name__:
    
    # Calculate constraints for extended MF from microlensing.
    f_pbh_subaru_sigma05 = []
    f_pbh_subaru_sigma10 = []
    
    for m_c in mc_subaru:
        
        f_pbh_subaru_sigma05.append(1/np.trapz(integrand(1, m_subaru_mono, m_c, 0.5, m_subaru_mono, f_max_subaru_mono), m_subaru_mono))
        f_pbh_subaru_sigma10.append(1/np.trapz(integrand(1, m_subaru_mono, m_c, 1.0, m_subaru_mono, f_max_subaru_mono), m_subaru_mono))
       
    fig, ax = plt.subplots(figsize=(5.5,5.5))
    ax.plot(mc_subaru, f_pbh_subaru_sigma05, label="$\sigma = 0.5$")
    ax.plot(mc_subaru, f_pbh_subaru_sigma10, label="$\sigma = 1.0$")
    ax.plot(m_subaru_mono, f_max_subaru_mono, color='k', linestyle='dotted', linewidth=3, alpha=0.5, label="Monochromatic")
    ax.set_xlabel('$M_\mathrm{c} / M_\mathrm{PBH} ~[\mathrm{g}]$')
    ax.set_ylabel('$f_\mathrm{PBH}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.set_xlim(1e21, 1e29)
    ax.set_ylim(1e-3, 1)
    ax.set_title("Log-normal MF")
    fig.tight_layout()
