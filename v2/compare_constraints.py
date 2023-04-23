#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 18:25:20 2022

@author: ppxmg2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from preliminaries import load_data
from extended_MF_checks import envelope, load_results_Isatis

# Script to compare different evaporation constraints.

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

def find_f_PBH(beta_prime, m):
    """
    Convert from beta prime to f_PBH, using Eq. 8 of Carr, Kühnel & Sandstad 
    (2016) [1607.06077].

    Parameters
    ----------
    beta_prime : Array-like
        Scaled initial fraction of Universe in PBHs.
    m : Array-like
        PBH mass, in grams.

    Returns
    -------
    Array-like.
        Fraction of dark matter in PBHs.

    """
    return 4.11e8 * beta_prime * (m / 1.989e33)**(-1/2)


def find_beta_prime(f_PBH, m):
    """
    Convert from beta prime to f_PBH, using the inverse of Eq. 8 of Carr, 
    Kühnel & Sandstad (2016) [1607.06077].   

    Parameters
    ----------
    f_PBH : Array-like
        Fraction of dark matter in PBHs.
    m : Array-like
        PBH mass, in grams.

    Returns
    -------
    Array-like
        Scaled initial fraction of Universe in PBHs.

    """
    return ((1/4.11e8) * f_PBH * (m / 1.989e33)**(1/2))


#%%

m_GC, beta_GC = load_data('2206.02672/2206.02672_Fig14_GC.csv')
m_GC_v2 = np.logspace(11, 21, 101)
constraints_names_unmodified, f_PBH_Isatis_unmodified = load_results_Isatis(modified=False)
fPBH_GC = envelope(f_PBH_Isatis_unmodified)

# Compare constraints on PBHs from Galactic photons shown in Fig. 14 of
# the review by Auffinger (2206.02672) and Auffinger (2022) (2201.01265) Fig. 3
# Plot beta prime.

#beta_prime_DLR20 = find_beta_prime(fPBH_D20, m_D20_v2)
beta_prime_GC = find_beta_prime(fPBH_GC, m_GC_v2)
#beta_prime_AAS20 = find_beta_prime(fPBH_EX, m_EX)

fig, ax = plt.subplots(figsize=(6,3))
ax.plot(m_GC, beta_GC, color='tab:orange', label = r"Auffinger review Fig. 14")
ax.plot(m_GC_v2, beta_prime_GC, linewidth=2, linestyle='dotted', color='r', label = r"2201.01265")

ax.set_xlabel(r"$m$ [g]")
ax.set_ylabel(r"$\beta '$")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(1e-28, 1e-16)
ax.set_xlim(1e14, 1e17)
ax.legend(fontsize='x-small')
plt.tight_layout()

# Compare constraints on PBHs from Galactic photons shown in Fig. 14 of
# the review by Auffinger (2206.02672) and Auffinger (2022) (2201.01265) Fig. 3
# Plot f_PBH.

fPBH_GC_review = find_f_PBH(beta_GC, m_GC)
fig, ax = plt.subplots(figsize=(6,3))
ax = plt.gca()
ax.plot(m_GC, fPBH_GC_review, color='tab:orange', label = r"Auffinger review Fig. 14")
ax.plot(m_GC_v2, fPBH_GC, linewidth=2, linestyle='dotted', color='r', label = r"2201.01265 Fig. 3")

ax.set_xlabel(r"$m$ [g]")
ax.set_ylabel(r"$f_\mathrm{PBH}$")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(1e-10, 1)
ax.set_xlim(1e14, 1e17)
ax.legend(fontsize='x-small')
plt.tight_layout()

#%%
# Plot constraints from DLR '20, Boudaud & Cirelli (2019), including
# uncertainty range, and compare to those from galactic photons.

m_GC_v2 = np.logspace(11, 21, 1000)
constraints_names_unmodified, f_PBH_Isatis = load_results_Isatis(modified=True)
fPBH_GC = envelope(f_PBH_Isatis)
fPBH_GC_strongest = fPBH_GC * np.power(10, -2.5)
fPBH_GC_weakest = fPBH_GC * np.power(10, 2.5)

m_D20_v2, fPBH_D20_strongest = load_data('1912.01014/1912.01014_Fig2_a__0_newaxes_2.csv')
#m_EX, fPBH_EX = load_data('AAS20_Fig4_mono.csv')

fPBH_GC_Auffinger = find_f_PBH(beta_GC, m_GC)

m_BC19_v2_weakest, fPBH_BC19_weakest = load_data("1807.03075/1807.03075_prop_B_nobkg.csv")
m_BC19_v2_strongest, fPBH_BC19_strongest = load_data("1807.03075/1807.03075_prop_A_bkg.csv")
fPBH_BC19_strongest_interp = np.interp(m_BC19_v2_weakest, m_BC19_v2_strongest, fPBH_BC19_strongest)
fPBH_BC19_mean = 10 ** (0.5*(np.log10(fPBH_BC19_weakest) + np.log10(fPBH_BC19_strongest_interp)))

m_KP23, fPBH_KP23 = load_data("2302.04408/2302.04408_MW_diffuse_SPI.csv")

# calculate weakest constraint (within the uncertainties) from Dasgupta, Laha & Ray (2020)
r_s = 20    # scale radius, in kpc
R_min = 1.5    # minimum positron propagation distance, in kpc
R_max = 3.5    # maximim positron propagation distance, in kpc
annihilation_fraction = 0.8     # fraction of positrons produced in Galactic centre which annihilate, when R = 3.5 kpc
fPBH_D20_weakest = fPBH_D20_strongest * (((np.log(1 + (R_max / r_s)) - R_max / (R_max + r_s))) / (np.log(1 + (R_min / r_s)) - R_min / (R_min + r_s))) / annihilation_fraction
fPBH_D20_mean = 10 ** (0.5*(np.log10(fPBH_D20_weakest) + np.log10(fPBH_D20_strongest)))

plt.figure(figsize=(7,7))
ax = plt.gca()
ax2 = ax.twinx()
ax3 = ax.twinx()
ax.plot(m_GC_v2, fPBH_GC, label = "Isatis")
ax.fill_between(m_GC_v2, fPBH_GC_weakest, fPBH_GC_strongest, alpha=0.25)

ax2.fill_between(m_D20_v2, fPBH_D20_strongest, fPBH_D20_weakest, color="tab:orange", alpha=0.25)
ax2.plot(m_D20_v2, fPBH_D20_mean, color="tab:orange", label = "DLR '20")
ax2.plot(m_KP23, fPBH_KP23, linestyle="dashed", color="purple", label="KP '23 MW diffuse (SPI)")

ax2.plot(fPBH_BC19_mean, fPBH_BC19_mean, color="tab:red", label="Voyager-1 (BC '19)")
ax2.fill_between(m_BC19_v2_weakest, fPBH_BC19_strongest_interp, m_BC19_v2_weakest, color="tab:red", alpha=0.25)

ax.set_xlabel(r"$M_\mathrm{PBH}$ [g]")
ax.set_ylabel(r"$f_\mathrm{PBH}$")
ax.set_xscale('log')
ax.set_yscale('log')
ax2.set_xscale("log")
ax2.set_yscale("log")
#ax3.set_xscale("log")
#ax3.set_yscale("log")
ax.set_ylim(1e-6, 1)
ax2.set_ylim(1e-6, 1)
#ax3.set_ylim(1e-6, 1)
ax2.get_yaxis().set_visible(False)
#ax3.get_yaxis().set_visible(False)

ax.set_xlim(1e16, 1e18)
ax.legend(fontsize='small', title=r"Galactic centre photons", loc=[0.53, 0.28])
ax2.legend(fontsize='small', title=r"Electrons/positrons", loc=[0.48, 0.07])
#ax3.legend(fontsize='small', title=r"Voyager-1 (Boudaud \& Cirelli 2019)", loc=[0.35, 0.025])

plt.tight_layout()
plt.savefig("./Existing_constraints/evap_tightest_est_uncertainties.pdf")