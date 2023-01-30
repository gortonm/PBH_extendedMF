#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 12:01:35 2023

@author: ppxmg2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import erf

# Check certain assumptions in the derivations in Niemeyer & Jedamzik (1998)
# (astro-ph/9709072)

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

delta_c = 1/3
gamma = 0.356
k = 1

# choose range of sigma between 0.1 delta_c and 0.2 delta_c, corresponding
# to the comment in the paper that this range of sigma generates a
# cosmologically interesting number of PBHs, for Gaussian statistics.
sigma_min = 0.1*delta_c
sigma_max = 0.2*delta_c

#%%

def delta_m(sigma):
    return 0.5 * (delta_c + np.sqrt(4*gamma*sigma**2 + delta_c**2))
    
def delta_m_approx(sigma):
    return delta_c + gamma*sigma**2 / delta_c

def integrand(delta, sigma):
    return k * np.power(delta-delta_c, gamma) * np.exp(-delta**2 / (2*sigma**2)) / (sigma * np.sqrt(2*np.pi))

def omega_PBH_new(sigma):
    return 0.5 * k * np.power(sigma, 2*gamma) * (erf(1/(np.sqrt(2)*sigma)) - erf(delta_c/(np.sqrt(2)*sigma)))
    
def omega_PBH_new_approx(sigma):
    return (np.sqrt(2)/4) * k * np.power(sigma, 1 + 2*gamma) * ((np.exp(-delta_c**2 / (2*sigma**2))/delta_c) - (np.exp(- 1 / (2*sigma**2))))
    
def omega_PBH_new_approx_v2(sigma):
    return k * np.power(sigma, 1 + 2*gamma) * np.exp(-delta_c**2 / (2*sigma**2))

def omega_PBH_new_numeric(sigma):
    deltas_integral = np.logspace(np.log10(delta_c), 0, 10000)
    integrand_values = [integrand(delta, sigma) for delta in deltas_integral]
    return np.trapz(integrand_values, deltas_integral)

sigmas_integrand = np.logspace(np.log10(sigma_min), np.log10(sigma_max), 5)
deltas = np.logspace(np.log10(delta_c), 0, 1000)

for sigma in sigmas_integrand:
    plt.figure()
    plt.plot(deltas, integrand(deltas, sigma)/max(integrand(deltas, sigma)), label="${:.3f}$".format(sigma))
    plt.gca().vlines(delta_m(sigma), ymin=0, ymax=1, color="k", linestyle="dashed", alpha=0.5)
    plt.gca().vlines(delta_m_approx(sigma), ymin=0, ymax=1, color="k", linestyle="dotted", alpha=0.5)
    plt.xlabel("$\delta$")
    plt.ylabel("Integrand (normalised to peak value)")
    plt.xscale("log")
    #plt.yscale("log")
    plt.legend(title="$\sigma$")
    plt.tight_layout()

sigmas = np.logspace(np.log10(sigma_min), np.log10(sigma_max), 1000)
omega_PBH_trapz = [omega_PBH_new_numeric(sigma) for sigma in sigmas]

plt.figure()
plt.plot(sigmas, omega_PBH_trapz, label="Trapezium rule")
plt.plot(sigmas, omega_PBH_new_approx_v2(sigmas), label="Approximate expression from Eq. (7)")
plt.xlabel("$\sigma$")
plt.ylabel("$\hat{\Omega}_\mathrm{PBH, new} / k$")
plt.yscale("log")
plt.legend()
plt.tight_layout()


plt.figure()
plt.plot(sigmas, omega_PBH_trapz/omega_PBH_new_approx_v2(sigmas), label="Trapezium rule \n / Approximate analytic expression")
plt.xlabel("$\sigma$")
plt.ylabel("$\hat{\Omega}_\mathrm{PBH, new}$")
plt.legend()
plt.tight_layout()

