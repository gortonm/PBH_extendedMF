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
gamma = 0.35
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
plt.savefig("./Figures/Critical_collapse/30-1_trapezium_approx_comparison.pdf")

plt.figure()
plt.plot(sigmas, omega_PBH_trapz/omega_PBH_new_approx_v2(sigmas), label="Trapezium rule \n / Approximate analytic expression")
plt.xlabel("$\sigma$")
plt.ylabel("$\hat{\Omega}_\mathrm{PBH, new} / k$")
plt.legend()
plt.tight_layout()
plt.savefig("./Figures/Critical_collapse/30-1_trapezium_approx_ratio.pdf")

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
    
    ln_term = np.trapz(ln_term_integrand, m_values)
    ln_square_term = np.trapz(ln_square_term_integrand, m_values)
    
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
#gamma = 0.3558019   # using the precise value from gr-qc/9503007
gamma = 0.36   # using value from Niemeyer & Jedamzik (1998) and Gow et al. (2021)
m_p = 25   # this can be chosen freely

# quantities appearing in the Niemeyer & Jedamzik (1998) mass function
delta_c = 0.5
sigma_PS = 0.1 * delta_c  # power spectrum standard deviation
K = 1   # this can be chosen freely

# quantities appearing in the skew-LN mass function
sigma_SLN = 0.55
alpha_SLN = -2.27
m_c = 100   # this can be chosen freely

# quantities appearing in the critical collapse mass function
alpha_CC = 3.06
beta = 2.12
m_f = 100   # this can be chosen freely

def mf_LN(m):
    # Lognormal mass function
    return np.exp(-np.log(m/m_c)**2 / (2*sigma**2)) / (np.sqrt(2*np.pi) * sigma * m)

# shape of mass function differs from Eq. (9) in Yokoyama (1998) since it is in
# terms of the peak mass, and the MF shown is d\Omega / dM rather than
# d\Omega / d\ln(M)
def mf_Yokoyama_shape(m, m_p=m_p, gamma=gamma):   
    return np.power(m/m_p, 1/gamma) * np.exp(-np.power(m/m_p, 1/gamma))

def mf_Yokoyama(m, m_p=m_p, gamma=gamma):
    mf_Yokoyama_normalisation = 1/np.trapz(mf_Yokoyama_shape(m_values, m_p, gamma), m_values)
    return mf_Yokoyama_normalisation * mf_Yokoyama_shape(m, m_p, gamma)

# mass function for d\Omega / dM has the same shape as d\phi / d\ln M, since
# d\phi / d\ln M = M * d\phi / dM \propto M * dn / dM \propto d\Omega / dM
def mf_NJ98_shape(m, K=K, gamma=gamma, delta_c=delta_c, sigma_PS=sigma_PS):
    m_bh = m / K
    #print("term 1 = ", np.power(delta_c/sigma, 2))
    #print("term 2 = ", np.power( np.power(m/K, 1/gamma) / sigma , 2))
    #print("term 3 = ", (2/sigma) * np.power(m/K, 1/gamma) * delta_c/sigma )
    return np.power(m_bh, 1/gamma) * np.exp(- np.power(delta_c + np.power(m_bh, 1/gamma), 2) / (2*sigma_PS**2) )

def mf_NJ98(m, K=K, gamma=gamma, delta_c=delta_c, sigma_PS=sigma_PS):
    print(sigma_PS)
    mf_NJ98_normalisation = 1/np.trapz(mf_NJ98_shape(m_values, K, gamma, delta_c, sigma_PS), m_values)
    return mf_NJ98_normalisation * mf_NJ98_shape(m, K, gamma, delta_c, sigma_PS)

def mf_NJ98_approx(m, K=K, gamma=gamma, delta_c=delta_c, sigma_PS=sigma_PS):
    mf_NJ98_normalisation = 1/np.trapz(mf_NJ98_shape(m_values, K, gamma, delta_c, sigma_PS), m_values)
    m_bh = m / K
    return mf_NJ98_normalisation * np.power(m_bh, 1/gamma) * np.exp(-np.power(delta_c / sigma_PS, 2) / 2)

def skew_LN(m, m_c=m_c, sigma=sigma_SLN, alpha=alpha_SLN):
    # Skew-lognormal mass function, as defined in Eq. (8) of 2009.03204.
    return np.exp(-np.log(m/m_c)**2 / (2*sigma**2)) * (1 + erf( alpha_SLN * np.log(m/m_c) / (np.sqrt(2) * sigma_SLN))) / (np.sqrt(2*np.pi) * sigma_SLN * m)

def CC_v2(m, m_f=m_f, alpha_CC=alpha_CC, beta=beta):
    # Critical collapse mass function, as defined in Eq. (9) of 2009.03204.
    log_psi = np.log(beta/m_f) - loggamma((alpha_CC+1) / beta) + (alpha_CC * np.log(m/m_f)) - np.power(m/m_f, beta)
    return np.exp(log_psi)

def Gaussian(x, mu, sigma):
    return np.exp(-(x-mu)**2 / (2*sigma**2)) / (sigma * np.sqrt(2*np.pi))

#print("Log-normal (sigma = {:.2f})".format(sigma))
#print("sd = " + str(np.sqrt(var(mf_LN))) + "\n")

#print("Yokoyama (1998) MF")
#print("sd = " + str(np.sqrt(var(mf_Yokoyama))) + "\n")

print("Niemeyer & Jedamzik (1998) MF")
print("sd = " + str(np.sqrt(var(mf_NJ98))) + "\n")
print("sd^2 = " + str(var(mf_NJ98)) + "\n")
print("sd~^2 = " + str(var(mf_NJ98_approx)) + "\n")

#print("Skew LN")
#print("sd = "+ str(np.sqrt(var(skew_LN))) + "\n")

#print("GCC3")
#print("sd = " + str(np.sqrt(var(CC_v2))) + "\n")


#%%

def Mmax_NJ(m, K, gamma, delta_c, sigma_PS):
    return K * np.power(0.5*delta_c*(np.sqrt(4*(sigma_PS/delta_c)**2 + 1) - 1), gamma)

def Mmax_NJ_approx(m, K, gamma, delta_c, sigma_PS):
    return K * np.power(sigma_PS**2 / delta_c, gamma)


# Plot the mass functions
m_H = 1
m_pbh_plotting = np.linspace(0.05*m_H, 2.5*m_H, 100)
plt.figure(figsize=(6, 6))
plt.plot(m_pbh_plotting/m_H, mf_Yokoyama(m_pbh_plotting, m_p=m_H, gamma=0.36), label="Yokoyama")
plt.plot(m_pbh_plotting/m_H, mf_NJ98(m_pbh_plotting, K=3.3*m_H, gamma=0.36, delta_c=1/3, sigma_PS=0.1*1/3), label="NJ $(\sigma/\delta_c = 0.1$, $\delta_c = 1/3$)")
plt.plot(m_pbh_plotting/m_H, mf_NJ98(m_pbh_plotting, K=3.3*m_H, gamma=0.36, delta_c=0.5, sigma_PS=0.2*0.5), label="NJ $(\sigma/\delta_c = 0.2$, $\delta_c = 0.5$)")
#plt.plot(m_pbh_plotting/m_H, mf_NJ98(m_pbh_plotting, K=3.3*m_H, gamma=0.36, delta_c=0.5, sigma_PS=0.1*0.5), label="NJ $(\sigma/\delta_c = 0.1$, $\delta_c = 0.5$)")
#plt.plot(m_pbh_plotting/m_H, mf_NJ98(m_pbh_plotting, K=3.3*m_H, gamma=0.36, delta_c=1/3, sigma_PS=0.2*1/3), label="NJ $(\sigma/\delta_c = 0.2$, $\delta_c = 1/3$)")
plt.plot(m_pbh_plotting/m_H, Gaussian(m_pbh_plotting, m_H, sigma=0.374), label="Gaussian", color='k', linestyle="dashed", alpha=0.75)

ax = plt.gca()
ax.vlines(Mmax_NJ(m_pbh_plotting, K=3.3*m_H, gamma=0.36, delta_c=1/3, sigma_PS=0.1*1/3)/m_H, ymin=0.1, ymax=3, linestyle='dotted', color='tab:orange')
ax.vlines(Mmax_NJ(m_pbh_plotting, K=3.3*m_H, gamma=0.36, delta_c=0.5, sigma_PS=0.2*0.5)/m_H, ymin=0.1, ymax=3, linestyle='dotted', color='tab:green')
ax.vlines(Mmax_NJ_approx(m_pbh_plotting, K=3.3*m_H, gamma=0.36, delta_c=1/3, sigma_PS=0.1*1/3)/m_H, ymin=0.1, ymax=3, linestyle='dashed', color='tab:orange')
ax.vlines(Mmax_NJ_approx(m_pbh_plotting, K=3.3*m_H, gamma=0.36, delta_c=0.5, sigma_PS=0.2*0.5)/m_H, ymin=0.1, ymax=3, linestyle='dashed', color='tab:green')

plt.xlabel("$M_\mathrm{PBH} / M_\mathrm{H}$")
plt.ylabel("$\mathrm{d}\phi / \mathrm{d}\ln M_\mathrm{PBH}$")
plt.yscale("log")
plt.ylim(0.1, 3)
plt.legend(fontsize="small")
plt.tight_layout()
plt.savefig("./Figures/Critical_collapse/MF_comparison_NJ_Yokoyama.png")


# Plot the mass functions against the maximum value
m_H = 1
m_pbh_plotting = np.linspace(0.05*m_H, 2.5*m_H, 100)
plt.figure(figsize=(6, 6))
plt.plot(m_pbh_plotting/m_H, mf_Yokoyama(m_pbh_plotting, m_p=m_H, gamma=0.36), label="Yokoyama")
plt.plot(m_pbh_plotting/Mmax_NJ(m_pbh_plotting, K=3.3*m_H, gamma=0.36, delta_c=1/3, sigma_PS=0.1*1/3)/m_H, mf_NJ98(m_pbh_plotting, K=3.3*m_H, gamma=0.36, delta_c=1/3, sigma_PS=0.1*1/3), label="NJ $(\sigma/\delta_c = 0.1$, $\delta_c = 1/3$)")
plt.plot(m_pbh_plotting/Mmax_NJ(m_pbh_plotting, K=3.3*m_H, gamma=0.36, delta_c=0.5, sigma_PS=0.2*0.5)/m_H, mf_NJ98(m_pbh_plotting, K=3.3*m_H, gamma=0.36, delta_c=0.5, sigma_PS=0.2*0.5), label="NJ $(\sigma/\delta_c = 0.2$, $\delta_c = 0.5$)")
#plt.plot(m_pbh_plotting/m_H, mf_NJ98(m_pbh_plotting, K=3.3*m_H, gamma=0.36, delta_c=0.5, sigma_PS=0.1*0.5), label="NJ $(\sigma/\delta_c = 0.1$, $\delta_c = 0.5$)")
#plt.plot(m_pbh_plotting/m_H, mf_NJ98(m_pbh_plotting, K=3.3*m_H, gamma=0.36, delta_c=1/3, sigma_PS=0.2*1/3), label="NJ $(\sigma/\delta_c = 0.2$, $\delta_c = 1/3$)")
plt.plot(m_pbh_plotting/m_H, Gaussian(m_pbh_plotting, m_H, sigma=0.374), label="Gaussian", color='k', linestyle="dashed", alpha=0.75)

plt.xlabel("$M_\mathrm{PBH} / M_\mathrm{max}$")
plt.ylabel("$\mathrm{d}\phi / \mathrm{d}\ln M_\mathrm{PBH}$")
plt.yscale("log")
plt.ylim(0.1, 3)
plt.xlim(0, 2)
plt.legend(fontsize="small")
plt.tight_layout()
plt.savefig("./Figures/Critical_collapse/MF_comparison_NJ_Yokoyama_Mmax.png")



#%% Calculations from Gow et al. (2021) mass function with a delta-function
# power spectrum peak

top_hat = True
Gaussian = False

g_star = 10.75   # number of relativistic degrees of freedom at PBH formation
M_H = 1.  # horizon mass at PBH formation, in solar masses

A = 1e-2  # power spectrum peak amplitude
gamma = 0.36   # critical exponent


def k_p(M_H):
    # peak scale in terms of the horizon mass it corresponds to (in Mpc^{-1})
    return np.sqrt(17) * 1e6 * np.power(g_star / 10.75, -1/12) * np.power(M_H, -1/2)

if top_hat:
    K = 4
    C_c = 0.55

if Gaussian:
    K = 10
    C_c = 0.25

def window_top_hat(k, R):
    if k * R < 4.49 / R:
        return 3 * (np.sin(k*R) - (k*R)*np.cos(k*R)) / (k*R)**3
    else:
        return 0

def window_Gaussian(k, R):
    return np.exp(-(k*R)**2 / 4)

def window(k, R):
    if top_hat:
        return window_top_hat(k, R)
    elif Gaussian:
        return window_Gaussian(k, R)

k_p = k_p(M_H)


# Hubble radius at formation (in Mpc), in terms of the horizon mass it 
# corresponds to
R = 3.1e-7 * np.sqrt(M_H)

sigma0 = (16/81) * A * (k_p * R)**4 * window(k_p, R)**2 / k_p
print(sigma0)


def mf_Gow21_delta_shape(m, K=K, gamma=gamma, C_c=C_c, sigma0=sigma0):
    x = m / (M_H * K)
    return np.power(x, 1/gamma) * np.exp(- np.power(C_c + np.power(x, 1/gamma), 2) / (2*sigma0**2) )

print("Gow et al. (2021) MF (delta-function power spectrum)")
print("sd = " + str(np.sqrt(var(mf_Gow21_delta_shape))) + "\n")
