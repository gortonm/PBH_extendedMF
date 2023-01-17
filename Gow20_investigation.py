#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 15:44:36 2023
@author: ppxmg2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import erf
from scipy.special import gamma as Gamma
from loadBH import load_data


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


def skew_LN(m, m_c, sigma, alpha):
    # Skew-lognormal mass function, as defined in Eq. (8) of 2009.03204.
    return np.exp(-np.log(m/m_c)**2 / (2*sigma**2)) * (1 + erf( alpha * np.log(m/m_c) / (np.sqrt(2) * sigma))) / (np.sqrt(2*np.pi) * sigma * m)

def loc_param_CC(m_p, alpha, beta):
    # Location parameter for critical collapse mass function, from Table I of 2009.03204.
    return m_p * np.power(beta/alpha, 1/beta)

def CC(m, m_f, alpha, beta, gamma=0.36):
    # Critical collapse mass function, as defined in Eq. (9) of 2009.03204.
    return (beta / m_f) * np.power(Gamma((alpha+1)/beta), -1) * np.power((m / m_f), alpha) * np.exp(-np.power(m/m_f, beta))


fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 22))

deltas = np.array([0., 0.1, 0.3, 0.5, 1.0, 2.0, 5.0])
ln_mcs = np.array([4.13, 4.13, 4.15, 4.21, 4.40, 4.88, 5.41])
sigmas = np.array([0.55, 0.55, 0.57, 0.60, 0.71, 0.97, 2.77])
alphas_SL = np.array([-2.27, -2.24, -2.07, -1.82, -1.31, -0.66, 1.39])
mps_SL = np.array([40.9, 40.9, 40.9, 40.8, 40.8, 40.6, 32.9])
mps_CC = np.array([40.8, 40.8, 40.7, 40.7, 40.8, 40.6, 35.1])
alphas_CC = np.array([3.06, 3.09, 3.34, 3.92, 5.76, 18.9, 13.9])
betas = np.array([2.12, 2.08, 1.72, 1.27, 0.51, 0.0669, 0.0206])
# maximum values of the mass function
psis_SL_max = skew_LN(mps_SL, np.exp(ln_mcs), sigmas, alphas_SL)
psis_CC_max = CC(mps_CC, loc_param_CC(mps_CC, alphas_CC, betas), alphas_CC, betas)

# PBH masses (in solar masses)
m_pbh_values = np.logspace(0, 3.5, 1000)

for i in range(len(deltas)):
    psi_SL = skew_LN(m_pbh_values, m_c=np.exp(ln_mcs[i]), sigma=sigmas[i], alpha=alphas_SL[i]) / psis_SL_max[i]
    psi_CC = CC(m_pbh_values, loc_param_CC(mps_CC[i], alphas_CC[i], betas[i]), alphas_CC[i], betas[i]) / psis_CC_max[i]
    
    # find x-axis limits
    xmin = m_pbh_values[min(min(np.where(psi_SL > 0.1)))]
    xmax = m_pbh_values[max(max(np.where(psi_SL > 0.1)))]
    
    ax = axes[int((i+1)/2)][(i+1)%2]
    ax.plot(m_pbh_values, psi_SL, color='b', linestyle=(0, (5, 7)))
    ax.plot(m_pbh_values, psi_CC, color='g', linestyle="dashed")
    if (i+1)%2 == 0:
        ax.set_ylabel("$\psi(m) / \psi_\mathrm{max}$")
    if int((i+1)/2) == 3:
        ax.set_xlabel("$m~[M_\odot]$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.1, 2)
    ax.legend(title="$\Delta = {:.1f}$".format(deltas[i]))
    
fig.tight_layout(h_pad=2)



# Compare mass functions to those plotted in Fig. 5 of 2009.03204.
fig_comp, axes_comp = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

for j in range(2):
    ax_comp = axes_comp[j]
    i = j+3
    
    psi_SL = skew_LN(m_pbh_values, m_c=np.exp(ln_mcs[i]), sigma=sigmas[i], alpha=alphas_SL[i]) / psis_SL_max[i]
    psi_CC = CC(m_pbh_values, loc_param_CC(mps_CC[i], alphas_CC[i], betas[i]), alphas_CC[i], betas[i]) / psis_CC_max[i]
    
    # find x-axis limits
    xmin = m_pbh_values[min(min(np.where(psi_SL > 0.05)))]
    xmax = m_pbh_values[max(max(np.where(psi_SL > 0.05)))]

    m_SL_comp, SL_comp = load_data("Delta_{:.1f}_SL.csv".format(deltas[i]))
    m_CC_comp, CC_comp = load_data("Delta_{:.1f}_CC.csv".format(deltas[i]))
    ax_comp.plot(m_SL_comp, SL_comp, color='cyan', marker='x', linestyle='None')
    ax_comp.plot(m_CC_comp, CC_comp, color='greenyellow', marker='x', linestyle='None')
    ax_comp.plot(m_pbh_values, psi_SL, color='tab:blue', linestyle=(0, (5, 7)), label="Skew-LN")
    ax_comp.plot(m_pbh_values, psi_CC, color='tab:green', linestyle="dashed", label="Generalised CC")
    ax_comp.set_xscale("log")
    ax_comp.set_yscale("log")
    ax_comp.set_xlim(xmin, xmax)
    ax_comp.set_ylim(0.1, 2)
    ax_comp.legend()
    ax_comp.set_title(r"$\Delta = {:.1f}$".format(deltas[i]))

fig_comp.tight_layout(h_pad=2)