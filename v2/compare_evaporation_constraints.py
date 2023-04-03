#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 18:25:20 2022

@author: ppxmg2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from extended_MF_checks import load_results_Isatis, envelope

# Script to reproduce plots from Auffinger '22 (2206.02672)

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

def load_data(filename):
    
    """
    Load data from a file located in the folder './Extracted_files/'.

    Parameters
    ----------
    fname : String
        File name.

    Returns
    -------
    Array-like.
        Contents of file.
    """
    
    return np.genfromtxt(filepath+filename, delimiter=',', unpack=True)


#%%
# Plot constraints from DLR '20, Boudaud & Cirelli (2019), including
# uncertainty range, and compare to those from galactic photons.

m_GC = np.logspace(11, 21, 1000)
constraints_names, f_PBH_Isatis = load_results_Isatis(modified=True)
fPBH_GC_Auffinger = envelope(f_PBH_Isatis)

m_BC19_v2_weakest, fPBH_BC19_weakest = load_data("./1807.03075/1807.03075_prop_B_nobkg.csv")
m_BC19_v2_strongest, fPBH_BC19_strongest = load_data("./1807.03075/1807.03075_prop_A_bkg.csv")

m_Berteaud22, fPBH_Berteaud22 = load_data("./2202.07483/2202.07483_Fig3.csv")
m_KP23_NFW, fPBH_KP23_NFW = load_data("./2302.04408/2302.04408_MW_NFW.csv")
m_KP23_tot, fPBH_KP23_tot = load_data("./2302.04408/2302.04408_MW_total.csv")

# calculate weakest constraint (within the uncertainties) from Dasgupta, Laha & Ray (2020)
m_D20_v2, fPBH_D20 = load_data("./1912.01014/1912.01014_Fig2_a__0_newaxes_2.csv")
r_s = 20    # scale radius, in kpc
R_min = 1.5    # minimum positron propagation distance, in kpc
R_max = 3.5    # maximim positron propagation distance, in kpc
annihilation_fraction = 0.8     # fraction of positrons produced in Galactic centre which annihilate, when R = 3.5 kpc
fPBH_D20_weakest = fPBH_D20 * (((np.log(1 + (R_max / r_s)) - R_max / (R_max + r_s))) / (np.log(1 + (R_min / r_s)) - R_min / (R_min + r_s))) / annihilation_fraction

plt.figure(figsize=(7,7))
ax = plt.gca()
ax2 = ax.twinx()
ax3 = ax.twinx()
ax.plot(m_GC, fPBH_GC_Auffinger, label = "Isatis defaults")
ax.plot(m_Berteaud22, fPBH_Berteaud22, color="k", linestyle="dashed", label="Berteaud+ '22")
#ax2.plot(m_D20_v2, fPBH_D20_weakest, color="tab:orange", linestyle='dotted', label = "DLR '20 Iso 1.5 kpc")
ax2.plot(m_D20_v2, fPBH_D20, color="tab:orange", label = "DLR '20 NFW, 3.5 kpc")
ax.plot(m_KP23_NFW, fPBH_KP23_NFW, color="purple", linestyle="dashed", label="KP '23 MW diffuse \n (SPI, NFW template)")
ax.plot(m_KP23_tot, fPBH_KP23_tot, color="purple", label="KP '23 MW diffuse (SPI)")

#ax3.plot(m_BC19_v2_weakest, fPBH_BC19_weakest, color="tab:red", linestyle='dotted', label = "Prop B, w/o background")
#ax3.plot(m_BC19_v2_strongest, fPBH_BC19_strongest, color="tab:red", label = "Prop A, w/ background")

ax.set_xlabel(r"$M_\mathrm{PBH}$ [g]")
ax.set_ylabel(r"$f_\mathrm{PBH}$")
ax.set_xscale('log')
ax.set_yscale('log')
ax2.set_xscale("log")
ax2.set_yscale("log")
ax3.set_xscale("log")
ax3.set_yscale("log")
ax.set_ylim(1e-5, 1)
ax2.set_ylim(1e-5, 1)
ax3.set_ylim(1e-5, 1)
ax2.get_yaxis().set_visible(False)
#ax3.get_yaxis().set_visible(False)

ax.set_xlim(1e16, 1e18)
ax.legend(fontsize='small', title=r"Galactic centre photons", loc=[0.45, 0.2])
ax2.legend(fontsize='small', title=r"511 keV", loc=[0.48, 0.07])
#ax3.legend(fontsize='small', title=r"Voyager-1 (Boudaud \& Cirelli 2019)", loc=[0.35, 0.025])

plt.tight_layout()
plt.savefig("./Results/Figures/Existing_constraints/evap_tightest.pdf")