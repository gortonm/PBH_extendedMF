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
    plt.savefig("./Figures/Critical_collapse/30-1_integrand_sigma={:.3f}.png".format(sigma), doi=1200)

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
plt.savefig("./Figures/Critical_collapse/30-1_trapezium_approx_comparison.png", doi=1200)

plt.figure()
plt.plot(sigmas, omega_PBH_trapz/omega_PBH_new_approx_v2(sigmas), label="Trapezium rule \n / Approximate analytic expression")
plt.xlabel("$\sigma$")
plt.ylabel("$\hat{\Omega}_\mathrm{PBH, new} / k$")
plt.legend()
plt.tight_layout()
plt.savefig("./Figures/Critical_collapse/30-1_trapezium_approx_ratio.png", doi=1200)

#%% Estimate the variance of the mass function
# Note that the mass function given in Eq. (10) is the fraction of PBH number
# per logarithmic interval, while the mass function definition in 1705.05567
# is the density per linear interval. The density per unit linear interval
# is proportional to the mass function in Eq. (10)

from scipy.special import loggamma

def var(mf, m_min=1e-5, m_max=1e5, n_steps=10000):
    m_values = np.logspace(np.log10(m_min), np.log10(m_max), n_steps)
    
    ln_term_integrand = np.log(m_values) * mf(m_values)
    ln_square_term_integrand = np.log(m_values)**2 * mf(m_values)
    
    ln_term = np.trapz(m_values, ln_term_integrand)
    ln_square_term = np.trapz(m_values, ln_square_term_integrand)
    
    var = ln_square_term - ln_term**2
    return var


m_c = 1
sigma = 0.2   # parameter for checking log-normal mass function


# For the Yokoyama et al. mass function, these values are adequate to obtain
# the standard deviation to 3 SF.
m_min = 1e-5
m_max = 1e5
n_steps = 10000
m_values = np.logspace(np.log10(m_min), np.log10(m_max), n_steps)

# quantities appearing in the Yokoyama (1998) mass function
gamma = 0.3558019   # using the precise value from gr-qc/9503007
m_p = 25   # this can be chosen freely

# quantities appearing in the Niemeyer & Jedamzik (1998) mass function
delta_c = 1/3
sigma_PS = 0.15   # power spectrum standard deviation
K = 1   # this can be chosen freely

# quantities appearing in the skew-LN mass function
sigma_SLN = 0.55
alpha_SLN = -2.27
m_c = 10

# quantities appearing in the critical collapse mass function
alpha_CC = 3.06
beta = 2.12
m_f = 10

def mf_LN(m):
    # Lognormal mass function
    return np.exp(-np.log(m/m_c)**2 / (2*sigma**2)) / (np.sqrt(2*np.pi) * sigma * m)

def mf_Yokoyama_shape(m, m_p=m_p, gamma=gamma):   
    return np.power(m/m_p, 1/gamma) * np.exp(-np.power(m/m_p, 1/gamma))

mf_Yokoyama_normalisation = 1/np.trapz(m_values, mf_Yokoyama_shape(m_values, m_p, gamma))

def mf_Yokoyama(m, m_p=m_p, gamma=gamma):
    return mf_Yokoyama_normalisation * mf_Yokoyama_shape(m, m_p, gamma)

def mf_NJ98_shape(m, K=K, gamma=gamma, delta_c=delta_c, sigma_PS=sigma_PS):
    m_bh = m / K
    return np.power(m_bh, 1/gamma) * np.exp(- np.power(delta_c + np.power(m_bh, 1/gamma), 2) / (2*sigma_PS**2) )

mf_NJ98_normalisation = 1/np.trapz(m_values, mf_NJ98_shape(m_values, K, gamma, delta_c, sigma_PS))

def mf_NJ98(m, K=K, gamma=gamma, delta_c=delta_c, sigma_PS=sigma_PS):
    return mf_NJ98_normalisation * mf_NJ98_shape(m)

def skew_LN(m, m_c=m_c, sigma=sigma_SLN, alpha=alpha_SLN):
    # Skew-lognormal mass function, as defined in Eq. (8) of 2009.03204.
    return np.exp(-np.log(m/m_c)**2 / (2*sigma**2)) * (1 + erf( alpha_SLN * np.log(m/m_c) / (np.sqrt(2) * sigma_SLN))) / (np.sqrt(2*np.pi) * sigma_SLN * m)

def CC_v2(m, m_f=m_f, alpha_CC=alpha_CC, beta=beta):
    log_psi = np.log(beta/m_f) - loggamma((alpha_CC+1) / beta) + (alpha_CC * np.log(m/m_f)) - np.power(m/m_f, beta)
    return np.exp(log_psi)


print(var(mf_LN))
print(np.sqrt(abs(var(mf_LN))))

print(var(mf_Yokoyama))
print(np.sqrt(abs(var(mf_Yokoyama))))

print(var(mf_NJ98))
print(np.sqrt(abs(var(mf_NJ98))))

print(var(skew_LN))
print(np.sqrt(abs(var(skew_LN))))

print(var(CC_v2))
print(np.sqrt(abs(var(CC_v2))))

# check if the skew-LN and critical collapse MFs are normalised to 1
print(np.trapz(m_values, skew_LN(m_values)))
print(np.trapz(m_values, CC_v2(m_values)))


