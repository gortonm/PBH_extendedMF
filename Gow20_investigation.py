#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 15:44:36 2023
@author: ppxmg2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from scipy.special import erf, loggamma
from scipy.special import gamma as Gamma
from loadBH import load_data
import warnings


# Code for studying 2009.03204, on extended mass functions. 

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

# maximum float value in Python:
import sys
print(sys.float_info.max)


# Include this line to catch warnings in the same way as errors
warnings.filterwarnings("error")

def skew_LN(m, m_c, sigma, alpha):
    # Skew-lognormal mass function, as defined in Eq. (8) of 2009.03204.
    return np.exp(-np.log(m/m_c)**2 / (2*sigma**2)) * (1 + erf( alpha * np.log(m/m_c) / (np.sqrt(2) * sigma))) / (np.sqrt(2*np.pi) * sigma * m)

def loc_param_CC(m_p, alpha, beta):
    # Location parameter for critical collapse mass function, from Table I of 2009.03204.
    return m_p * np.power(beta/alpha, 1/beta)

def CC(m, m_f, alpha, beta, gamma=0.36):
    # Critical collapse mass function, as defined in Eq. (9) of 2009.03204.
    
    try:
        Gamma((alpha+1)/beta)
        np.power((m / m_f), alpha)
        np.exp(-np.power(m/m_f, beta))
    except RuntimeWarning:
        print("m_f = ", m_f)
        print("alpha = ", alpha)
        print("beta = ", beta)
        print("Gamma ((alpha+1)/beta) = ", Gamma((alpha+1)/beta))
        print("(alpha+1)/beta) = ", (alpha+1)/beta)
        print("np.exp(-(m/m_f)^beta) = ", np.exp(-np.power(m/m_f, beta)))
        print("(m / m_f)^alpha = ", np.power((m / m_f), alpha))

    return (beta / m_f) * np.power(Gamma((alpha+1)/beta), -1) * np.power((m / m_f), alpha) * np.exp(-np.power(m/m_f, beta))

def CC_normalised(m, m_p, alpha, beta):
    return np.power(m/m_p, alpha) * np.exp((alpha/beta)*(1 - np.power(m/m_p, beta)))

def CC_v2(m, m_f, alpha, beta):
    log_psi = np.log(beta/m_f) - loggamma((alpha+1) / beta) + (alpha * np.log(m/m_f)) - np.power(m/m_f, beta)
    print("ln(beta/m_f) = ", np.log(beta/m_f))
    print("ln(Gamma ((alpha+1)/beta)) = ", loggamma((alpha+1)/beta))
    print("alpha ln(m/m_f) = ", alpha * np.log(m/m_f))
    print("(m/m_f)^beta = ", np.power(m/m_f, beta))
    print("log_psi = ", log_psi)
    return np.exp(log_psi)

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 22))

deltas = np.array([0., 0.1, 0.3, 0.5, 1.0, 2.0, 5.0])
ln_mcs = np.array([4.13, 4.13, 4.15, 4.21, 4.40, 4.88, 5.41])
sigmas = np.array([0.55, 0.55, 0.57, 0.60, 0.71, 0.97, 2.77])
alphas_SL = np.array([-2.27, -2.24, -2.07, -1.82, -1.31, -0.66, 1.39])
mps_SL = np.array([40.9, 40.9, 40.9, 40.8, 40.8, 40.6, 32.9])
mps_CC = np.array([40.8, 40.8, 40.7, 40.7, 40.8, 40.6, 35.1])
alphas_CC = np.array([3.06, 3.09, 3.34, 3.82, 5.76, 18.9, 13.9])
betas = np.array([2.12, 2.08, 1.72, 1.27, 0.51, 0.0669, 0.0206])
# maximum values of the mass function
psis_SL_max = skew_LN(mps_SL, np.exp(ln_mcs), sigmas, alphas_SL)
psis_CC_max = CC_v2(mps_CC, loc_param_CC(mps_CC, alphas_CC, betas), alphas_CC, betas)

# PBH masses (in solar masses)
m_pbh_values = np.logspace(0, 3.5, 1000)

for i in range(len(deltas)):
    
    psi_SL = skew_LN(m_pbh_values, m_c=np.exp(ln_mcs[i]), sigma=sigmas[i], alpha=alphas_SL[i]) / psis_SL_max[i]
    psi_CC = CC_v2(m_pbh_values, loc_param_CC(mps_CC[i], alphas_CC[i], betas[i]), alphas_CC[i], betas[i]) / psis_CC_max[i]
    
    # find x-axis limits
    xmin = m_pbh_values[min(min(np.where(psi_SL > 0.1)))]
    xmax = m_pbh_values[max(max(np.where(psi_SL > 0.1)))]
    
    ax = axes[int((i+1)/2)][(i+1)%2]
    ax.plot(m_pbh_values, psi_SL, color='b', linestyle=(0, (5, 7)))
    ax.plot(m_pbh_values, psi_CC, color='g', linestyle="dashed")
    if (i+1)%2 == 0:
        ax.set_ylabel(r"$\psi(m) / \psi_\mathrm{max}$")
    if int((i+1)/2) == 3:
        ax.set_xlabel(r"$m~[M_\odot]$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.1, 2)
    ax.legend(title=r"$\Delta = {:.1f}$".format(deltas[i]))
    
fig.tight_layout(h_pad=2)



# Compare mass functions to those plotted in Fig. 5 of 2009.03204.
fig_SL, axes_SL = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
fig_CC, axes_CC = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

for j in range(2):
    ax_comp = axes_SL[j]
    i = j+3
    
    for axis in [ax_comp.xaxis, ax_comp.yaxis]:
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        axis.set_major_formatter(formatter)
        axis.set_minor_formatter(formatter)
    
    psi_SL = skew_LN(m_pbh_values, m_c=np.exp(ln_mcs[i]), sigma=sigmas[i], alpha=alphas_SL[i]) / psis_SL_max[i]
    
    # find x-axis limits
    xmin = m_pbh_values[min(min(np.where(psi_SL > 0.05)))]
    xmax = m_pbh_values[max(max(np.where(psi_SL > 0.05)))]

    m_SL_comp, SL_comp = load_data("Delta_{:.1f}_SL.csv".format(deltas[i]))
    ax_comp.plot(m_SL_comp, SL_comp, color='b', marker='x', linestyle='None', label="Extracted")
    ax_comp.plot(m_pbh_values, psi_SL, color='tab:blue', linestyle="dashed", alpha=0.5, label="Calculated")
    ax_comp.set_ylabel(r"$\psi(m) / \psi_\mathrm{max}$")
    ax_comp.set_xlabel(r"$m~[M_\odot]$")
    ax_comp.set_xscale("log")
    ax_comp.set_yscale("log")
    ax_comp.set_xlim(xmin, xmax)
    ax_comp.set_ylim(0.1, 2)
    for axis in [ax_comp.xaxis, ax_comp.yaxis]:
        axis.set_major_formatter(ScalarFormatter())
    ax_comp.xaxis.set_minor_formatter(ScalarFormatter())
    ax_comp.set_xticks([20, 40, 60, 80])
    ax_comp.legend()
    ax_comp.set_title(r"$\Delta = {:.1f}$".format(deltas[i]))

fig_CC.suptitle("Generalised Critical Collapse")
fig_SL.tight_layout(h_pad=2)


for j in range(2):
    ax_comp = axes_CC[j]
    i = j+3
    
    psi_CC = CC_v2(m_pbh_values, loc_param_CC(mps_CC[i], alphas_CC[i], betas[i]), alphas_CC[i], betas[i]) / psis_CC_max[i]
    
    # find x-axis limits
    xmin = m_pbh_values[min(min(np.where(psi_CC > 0.05)))]
    xmax = m_pbh_values[max(max(np.where(psi_CC > 0.05)))]

    m_CC_comp, CC_comp = load_data("Delta_{:.1f}_CC.csv".format(deltas[i]))
    ax_comp.plot(m_CC_comp, CC_comp, color='g', marker='x', linestyle='None', label="Extracted")
    ax_comp.plot(m_pbh_values, psi_CC, color='tab:green', linestyle="dashed", alpha=0.5, label="Calculated")
    ax_comp.set_ylabel(r"$\psi(m) / \psi_\mathrm{max}$")
    ax_comp.set_xlabel(r"$m~[M_\odot]$")
    ax_comp.set_xscale("log")
    ax_comp.set_yscale("log")
    ax_comp.set_xlim(xmin, xmax)
    ax_comp.set_ylim(0.1, 2)
    for axis in [ax_comp.xaxis, ax_comp.yaxis]:
        axis.set_major_formatter(ScalarFormatter())
    ax_comp.xaxis.set_minor_formatter(ScalarFormatter())
    ax_comp.set_xticks([10, 20, 50, 100])
    ax_comp.legend()
    ax_comp.set_title(r"$\Delta = {:.1f}$".format(deltas[i]))

fig_CC.suptitle("Generalised critical collapse")
fig_CC.tight_layout(h_pad=2)

#%%
# Reproduce Fig. 3 curve for the CC3 mass function, Delta = 5
fig, ax = plt.subplots(figsize=(6,6))

psi_CC_normalised = CC_v2(m_pbh_values, mps_CC[i], alphas_CC[i], betas[i]) / psis_CC_max[i]

# find x-axis limits
xmin = m_pbh_values[min(min(np.where(psi_CC_normalised > 0.05)))]
xmax = m_pbh_values[max(max(np.where(psi_CC_normalised > 0.05)))]

i = 6
m_CC_comp, CC_comp = load_data("Gow20_Fig3_Delta_{:.1f}_CC.csv".format(deltas[i]))
ax.plot(m_pbh_values, psi_CC_normalised, color='k', label="Calculated")
ax.plot(m_CC_comp, CC_comp, color='g', marker='x', linestyle='None', alpha=0.5,  label="Extracted")
ax.set_ylabel(r"$\psi(m) / \psi_\mathrm{max}$")
ax.set_xlabel(r"$m~[M_\odot]$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.9, 2000)
ax.set_ylim(0.1, 2)
ax.legend()
ax.set_title(r"$\Delta = {:.1f}$".format(deltas[i]))

fig.suptitle("Generalised critical collapse")
fig.tight_layout(h_pad=2)

# There is a difference in the results here, but I think they can be 
# explained sensibly. For the reasoning, see the 18/1 entry of 
# https://www.evernote.com/shard/s463/nl/233603521/4c05ae9d-c9cc-76b2-a4fb-bb0783b7246d?title=Gow%20et%20al.%20(2020)%20%5B2009.03204%5D:%20reproducing%20Fig.%205.


#%% Check individual sections of the GCC mass function.

m =  40
m_p = 40
alphas = [5.76, 18.9]
betas = [0.51, 0.0669]

"""
for i in range(len(alphas)):
    alpha = alphas[i]
    beta = betas[i]
    m_f = loc_param_CC(m_p, alpha, beta)
    
    
    print("alpha = ", alpha)
    print("beta = ", beta)
    print("m_f = ", m_f)
    print("(alpha+1)/beta) = ", (alpha+1)/beta)
    print("Gamma ((alpha+1)/beta) = ", Gamma((alpha+1)/beta))
    print("(m/m_f)^beta = ", np.power(m/m_f, beta))
    print("np.exp(-(m/m_f)^beta) = ", np.exp(-np.power(m/m_f, beta)))
    print("(m / m_f)^alpha = ", np.power((m / m_f), alpha))
    print("psi_CC = ", CC(m, m_f, alpha, beta))
"""

# check logarithms of quantities
for i in range(len(alphas)):
    alpha = alphas[i]
    beta = betas[i]
    m_f = loc_param_CC(m_p, alpha, beta)
    
    """
    print("alpha = ", alpha)
    print("beta = ", beta)
    print("m_f = ", m_f)
    print("ln(beta/m_f) = ", np.log(beta/m_f))
    print("ln(1 / Gamma ((alpha+1)/beta)) = ", loggamma((alpha+1)/beta))
    print("alpha ln(m/m_f) = ", alpha * np.log(m/m_f))
    print("(m/m_f)^beta = ", np.power(m/m_f, beta))
    """
    #print("(m / m_f)^alpha = ", np.power((m / m_f), alpha))
    print("log(psi_CC) = ", np.log(CC_v2(m, m_f, alpha, beta)))
