#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 18:25:20 2022

@author: ppxmg2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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



# Compare constraints on PBHs from Galactic photons shown in Fig. 14 of
# the review by Auffinger (2206.02672) and their sources.
# Note that the constraints shown from Auffinger (2022) (2201.01265) Fig. 3
# are only for COMPTEL, and this does not match the constraints shown in Fig.
# 14 of the review by Auffinger (2206.02672) and their sources.

m_D20, beta_D20 = load_data('Auffinger21_Fig10_D20.csv')
m_GC, beta_GC = load_data('Auffinger21_Fig14_GC.csv')
m_EX_XG, beta_XG = load_data('Auffinger22_Fig14_XG.csv')


m_D20_v2, fPBH_D20 = load_data('DLR20_Fig2_a__0.csv')
m_GC_v2, fPBH_GC = load_data('A22_Fig3.csv')

m_EX, fPBH_EX = load_data('AAS20_Fig4_mono.csv')


beta_prime_DLR20 = find_beta_prime(fPBH_D20, m_D20_v2)
beta_prime_GC = find_beta_prime(fPBH_GC, m_GC_v2)
beta_prime_AAS20 = find_beta_prime(fPBH_EX, m_EX)

plt.figure(figsize=(5.5,5.5))
ax = plt.gca()
ax.plot(m_D20, beta_D20, color='tab:blue', label = "511 keV \n (Auffinger review Fig. 14)")
ax.plot(m_GC, beta_GC, color='tab:orange', label = "Galactic Centre \n (Auffinger review Fig. 14)")
ax.plot(m_EX_XG, beta_XG, color='grey', label = "EXGB \n (Auffinger review Fig. 14)")

ax.plot(m_D20_v2, beta_prime_DLR20, linewidth=2, linestyle='dotted', color='g', label = "Positron annihilation \n Dasgupta, Laha \& Ray (2020)")
ax.plot(m_GC_v2, beta_prime_GC, linewidth=2, linestyle='dotted', color='r', label = "Galactic Centre photons \n Auffinger (2022) ")
ax.plot(m_EX, beta_prime_AAS20, linewidth=2, color='k', linestyle='dotted', label = "EXGB \n (Arbey, Auffinger \& Silk (2020))")

ax.set_xlabel("$M_\mathrm{PBH}$ [g]")
ax.set_ylabel(r"$\beta '$")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(1e-28, 1e-12)
ax.set_xlim(1e8, 1e18)
ax.legend(fontsize='small')
plt.tight_layout()


# Plot constraints from DLR '20, Boudaud & Cirelli (2019), including
# uncertainty range, and compare to those from galactic photons.

fPBH_GC_Auffinger = find_f_PBH(beta_GC, m_GC)

m_BC19_v2_weakest, fPBH_BC19_weakest = load_data("BC19_prop_B_nobkg.csv")
m_BC19_v2_strongest, fPBH_BC19_strongest = load_data("BC19_prop_A_bkg.csv")

m_KP23_LMC_weakest, fPBH_KP23_LMC_weakest = load_data("KP23_LMC_weakest.csv")
m_KP23_LMC_strongest, fPBH_KP23_LMC_strongest = load_data("KP23_LMC_tightest.csv")

# calculate weakest constraint (within the uncertainties) from Dasgupta, Laha & Ray (2020)
r_s = 20    # scale radius, in kpc
R_min = 1.5    # minimum positron propagation distance, in kpc
R_max = 3.5    # maximim positron propagation distance, in kpc
annihilation_fraction = 0.8     # fraction of positrons produced in Galactic centre which annihilate, when R = 3.5 kpc
fPBH_D20_weakest = fPBH_D20 * (((np.log(1 + (R_max / r_s)) - R_max / (R_max + r_s))) / (np.log(1 + (R_min / r_s)) - R_min / (R_min + r_s))) / annihilation_fraction

plt.figure(figsize=(7,6))
ax = plt.gca()
ax2 = ax.twinx()
ax3 = ax.twinx()
ax.plot(m_GC, fPBH_GC_Auffinger, label = "Auffinger review Fig. 14")
ax2.plot(m_D20_v2, fPBH_D20_weakest, color="tab:orange", linestyle='dotted', label = "DLR Iso 1.5 kpc")
ax2.plot(m_D20_v2, fPBH_D20, color="tab:orange", label = "DLR '20 NFW, 3.5 kpc")
ax2.plot(m_KP23_LMC_weakest, fPBH_KP23_LMC_weakest, linestyle="dotted", color="k")
ax2.plot(m_KP23_LMC_strongest, fPBH_KP23_LMC_strongest, color="k", label = "KP23 (LMC)")

ax3.plot(m_BC19_v2_weakest, fPBH_BC19_weakest, color="tab:red", linestyle='dotted', label = "Prop B, w/o background")
ax3.plot(m_BC19_v2_strongest, fPBH_BC19_strongest, color="tab:red", label = "Prop A, w/ background")

ax.set_xlabel("$M_\mathrm{PBH}$ [g]")
ax.set_ylabel("$f_\mathrm{PBH}$")
ax.set_xscale('log')
ax.set_yscale('log')
ax2.set_xscale("log")
ax2.set_yscale("log")
ax3.set_xscale("log")
ax3.set_yscale("log")
ax.set_ylim(1e-8, 1)
ax2.set_ylim(1e-8, 1)
ax3.set_ylim(1e-8, 1)
ax2.get_yaxis().set_visible(False)
ax3.get_yaxis().set_visible(False)

ax.set_xlim(1e15, 1e18)
ax.legend(fontsize='small', title="Galactic centre photons", loc=[0.526, 0.45])
ax2.legend(fontsize='small', title="511 keV", loc=[0.45, 0.22])
ax3.legend(fontsize='small', title="Voyager-1 (Boudaud \& Cirelli 2019)", loc=[0.35, 0.025])

plt.tight_layout()
plt.savefig("./Figures/Existing_constraints/evap_tightest.png")


fPBH_GC_Auffinger = find_f_PBH(beta_GC, m_GC)


# Plot constraints on PBHs from Galactic Centre photons, recalculated from
# isatis_reproduction.py, including the highest-energy bin

m_GC_COMPTEL_all_bins, fPBH_GC_COMPTEL_all_bins = np.genfromtxt("./Data/fPBH_Auffinger22_all_bins_COMPTEL_1107.0200.txt", delimiter="\t", unpack=True)
m_GC_EGRET_all_bins, fPBH_GC_EGRET_all_bins = np.genfromtxt("./Data/fPBH_Auffinger22_all_bins_EGRET_9811211.txt", delimiter="\t", unpack=True)
m_GC_INTEGRAL_all_bins, fPBH_GC_INTEGRAL_all_bins = np.genfromtxt("./Data/fPBH_Auffinger22_all_bins_INTEGRAL_1107.0200.txt", delimiter="\t", unpack=True)
m_GC_FermiLAT_all_bins, fPBH_GC_FermiLAT_all_bins = np.genfromtxt("./Data/fPBH_Auffinger22_all_bins_Fermi-LAT_1101.1381.txt", delimiter="\t", unpack=True)

fPBH_GC_envelope = []
for i in range(len(fPBH_GC_COMPTEL_all_bins)):
    fPBH_GC_envelope.append(min([fPBH_GC_COMPTEL_all_bins[i], fPBH_GC_EGRET_all_bins[i], fPBH_GC_INTEGRAL_all_bins[i], fPBH_GC_FermiLAT_all_bins[i]]))

plt.figure(figsize=(6,6))
ax = plt.gca()
ax.plot(m_GC, fPBH_GC_Auffinger, color='k', label = "Galactic Centre \n (Auffinger review Fig. 14)")
#ax.plot(m_GC_v2, fPBH_GC, linewidth=2, linestyle='dotted', color='r', label = "Galactic Centre photons \n Auffinger (2022)")
ax.plot(m_GC_INTEGRAL_all_bins, fPBH_GC_INTEGRAL_all_bins, linewidth=1.5, marker='x', linestyle='dotted', label = "INTEGRAL")
ax.plot(m_GC_COMPTEL_all_bins, fPBH_GC_COMPTEL_all_bins, linewidth=1.5, marker='x', linestyle='dotted', label = "COMPTEL")
ax.plot(m_GC_EGRET_all_bins, fPBH_GC_EGRET_all_bins, linewidth=1.5, marker='x', linestyle='dotted', label = "EGRET")
ax.plot(m_GC_FermiLAT_all_bins, fPBH_GC_FermiLAT_all_bins, linewidth=1.5, marker='x', linestyle='dotted', label = "Fermi-LAT")
ax.plot(m_GC_INTEGRAL_all_bins, fPBH_GC_envelope, linewidth=1.5, marker='x', linestyle='dotted', label = "envelope")

ax.set_xlabel("$M_\mathrm{PBH}$ [g]")
ax.set_ylabel("$f_\mathrm{PBH}$")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(1e-10, 1)
ax.set_xlim(1e14, 1e18)
ax.legend(fontsize='small')
plt.title("Including highest-energy bin")
plt.tight_layout()


# Plot constraints on PBHs from Galactic Centre photons, recalculated from
# isatis_reproduction.py, excluding the highest-energy bin, as shown in
# Fig. 14 of the review by Auffinger (2206.02672)

m_GC_COMPTEL, fPBH_GC_COMPTEL = np.genfromtxt("./Data/fPBH_Auffinger22_COMPTEL_1107.0200.txt", delimiter="\t", unpack=True)
m_GC_EGRET, fPBH_GC_EGRET = np.genfromtxt("./Data/fPBH_Auffinger22_EGRET_9811211.txt", delimiter="\t", unpack=True)
m_GC_INTEGRAL, fPBH_GC_INTEGRAL = np.genfromtxt("./Data/fPBH_Auffinger22_INTEGRAL_1107.0200.txt", delimiter="\t", unpack=True)
m_GC_FermiLAT, fPBH_GC_FermiLAT = np.genfromtxt("./Data/fPBH_Auffinger22_Fermi-LAT_1101.1381.txt", delimiter="\t", unpack=True)

m_A22_COMPTEL, fPBH_A22_COMPTEL = load_data("A22_Fig3_COMPTEL.csv")
m_A22_EGRET, fPBH_A22_EGRET = load_data("A22_Fig3_EGRET.csv")
m_A22_INTEGRAL, fPBH_A22_INTEGRAL = load_data("A22_Fig3_INTEGRAL.csv")
m_A22_FermiLAT, fPBH_A22_FermiLAT = load_data("A22_Fig3_FermiLAT.csv")

plt.figure(figsize=(6,6))
ax = plt.gca()
ax.plot(m_GC, fPBH_GC_Auffinger, color='k', label = "Galactic Centre \n (Auffinger review Fig. 14)")
#ax.plot(m_GC_v2, fPBH_GC, linewidth=2, linestyle='dotted', color='k', alpha=0.5, label = "Galactic Centre photons \n Auffinger (2022)")
ax.plot(m_GC_INTEGRAL, fPBH_GC_INTEGRAL, linewidth=1.5, marker='x', linestyle='dotted', label = "INTEGRAL")
ax.plot(m_GC_COMPTEL, fPBH_GC_COMPTEL, linewidth=1.5, marker='x', linestyle='dotted', label = "COMPTEL")
ax.plot(m_GC_EGRET, fPBH_GC_EGRET, linewidth=1.5, marker='x', linestyle='dotted', label = "EGRET")
ax.plot(m_GC_FermiLAT, fPBH_GC_FermiLAT, linewidth=1.5, marker='x', linestyle='dotted', label = "Fermi-LAT")

ax.set_xlabel("$M_\mathrm{PBH}$ [g]")
ax.set_ylabel("$f_\mathrm{PBH}$")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(1e-10, 1)
ax.set_xlim(1e14, 1e18)
ax.legend(fontsize='small')
plt.title("Excluding highest-energy bin")
plt.tight_layout()



plt.figure(figsize=(6,6))
ax = plt.gca()
ax.plot(m_A22_COMPTEL, fPBH_A22_COMPTEL, color='k', alpha=0.5)
ax.plot(m_A22_EGRET, fPBH_A22_EGRET, color='k', alpha=0.5)
ax.plot(m_A22_INTEGRAL, fPBH_A22_INTEGRAL, color='k', alpha=0.5)
ax.plot(m_A22_FermiLAT, fPBH_A22_FermiLAT, color='k', alpha=0.5)
ax.plot(m_GC_INTEGRAL, fPBH_GC_INTEGRAL, linewidth=1.5, marker='x', linestyle='None', label = "INTEGRAL")
ax.plot(m_GC_COMPTEL, fPBH_GC_COMPTEL, linewidth=1.5, marker='x', linestyle='None', label = "COMPTEL")
ax.plot(m_GC_EGRET, fPBH_GC_EGRET, linewidth=1.5, marker='x', linestyle='None', label = "EGRET")
ax.plot(m_GC_FermiLAT, fPBH_GC_FermiLAT, linewidth=1.5, marker='x', linestyle='None', label = "Fermi-LAT")

ax.set_xlabel("$M_\mathrm{PBH}$ [g]")
ax.set_ylabel("$f_\mathrm{PBH}$")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(1e-10, 1)
ax.set_xlim(1e14, 1e18)
ax.legend(fontsize='small')
plt.title("Excluding highest-energy bin")
plt.tight_layout()


# Compare constraints on PBHs from photons (Galactic photons and extragalactic)
#, to those from Voyager-1 and 511 keV line constraints

plt.figure(figsize=(10,6))
ax = plt.gca()
ax2 = ax.twinx()
ax3 = ax.twinx()
ax.plot(m_GC_INTEGRAL_all_bins, fPBH_GC_envelope, label = "Galactic centre")
ax.plot(m_EX, fPBH_EX, color='cyan', label='Extragalactic \n (Arbey, Auffinger \& Silk (2020))')
ax2.plot(m_D20_v2, fPBH_D20_weakest, color="tab:orange", linestyle='dotted', label = "1.5 kpc")
ax2.plot(m_D20_v2, fPBH_D20, color="tab:orange", label = "NFW, 3.5 kpc")
ax3.plot(m_BC19_v2_weakest, fPBH_BC19_weakest, color="tab:red", linestyle='dotted', label = "Prop B, w/o background")
ax3.plot(m_BC19_v2_strongest, fPBH_BC19_strongest, color="tab:red", label = "Prop A, w/ background")

ax.set_xlabel("$M_\mathrm{PBH}$ [g]")
ax.set_ylabel("$f_\mathrm{PBH}$")
ax.set_xscale('log')
ax.set_yscale('log')
ax2.set_xscale("log")
ax2.set_yscale("log")
ax3.set_xscale("log")
ax3.set_yscale("log")
ax.set_ylim(1e-8, 1)
ax2.set_ylim(1e-8, 1)
ax3.set_ylim(1e-8, 1)
ax2.get_yaxis().set_visible(False)
ax3.get_yaxis().set_visible(False)

ax.set_xlim(1e15, 1e18)
ax.legend(fontsize='small', title="Photons", loc=[0.57, 0.45])
ax2.legend(fontsize='small', title="511 keV \n (Dasgupta, Laha \& Ray 2020)", loc=[0.57, 0.21])
ax3.legend(fontsize='small', title="Voyager-1 (Boudaud \& Cirelli 2019)", loc=[0.57, 0.025])

plt.tight_layout()


#%% Compare results obtained using my version of Isatis to the values 
# computed using Isatis directly

# Load result from Isatis
Isatis_path = './../Downloads/version_finale/scripts/Isatis/'
results_name = "A22Fig3"

constraints_file = np.genfromtxt("%s/results_photons_%s.txt"%(Isatis_path,results_name),dtype = "str")
constraints_names_bis = constraints_file[0,1:]
constraints = np.zeros([len(constraints_file)-1,len(constraints_file[0])-1])
for i in range(len(constraints)):
    for j in range(len(constraints[0])):
        constraints[i,j] = float(constraints_file[i+1,j+1])

masses = 10**np.arange(14, 18.05, 0.1)
Mmin = min(masses)  # minimum PBH mass in your runs
Mmax = max(masses)  # maximum PBH mass in your runs

# creating labels
constraints_names = []
for i in range(len(constraints_names_bis)):
    temp = constraints_names_bis[i].split("_")
    temp2 = ""
    for i in range(len(temp)-1):
        temp2 = "".join([temp2,temp[i],'\,\,'])
    temp2 = "".join([temp2,'\,\,[arXiv:',temp[-1],']'])
    constraints_names.append(temp2)


plt.figure(figsize=(6,6))
ax = plt.gca()
for i in range(len(constraints_names)):
    ax.plot(masses,constraints[:,i], color='k', alpha=0.5)
ax.plot(m_GC_INTEGRAL, fPBH_GC_INTEGRAL, linewidth=1.5, marker='x', linestyle='None', label = "INTEGRAL")
ax.plot(m_GC_COMPTEL, fPBH_GC_COMPTEL, linewidth=1.5, marker='x', linestyle='None', label = "COMPTEL")
ax.plot(m_GC_EGRET, fPBH_GC_EGRET, linewidth=1.5, marker='x', linestyle='None', label = "EGRET")
ax.plot(m_GC_FermiLAT, fPBH_GC_FermiLAT, linewidth=1.5, marker='x', linestyle='None', label = "Fermi-LAT")
ax.set_xlabel("$M_\mathrm{PBH}$ [g]")
ax.set_ylabel("$f_\mathrm{PBH}$")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(1e-10, 1)
ax.set_xlim(Mmin, Mmax)
ax.legend(fontsize='small')
plt.title("Excluding highest-energy bin")
plt.tight_layout()





#%% Compare strength of constraint from Galactic Centre photons, the 511 keV
# line constraints from 1912.01014 and constraints from the X-ray emission
# from galaxy clusters (from 2103.12354).

m_LC21, fPBH_LC21 = load_data('LC21_A262_NFW.csv')

plt.figure(figsize=(9, 6))
plt.plot(m_D20, fPBH_D20, linewidth=2, label = "Positron annihilation \n Dasgupta, Laha \& Ray (2020)")
plt.plot(m_D20, 10*fPBH_D20, label = "\t Upper bound")
plt.plot(m_GC, fPBH_GC, linewidth=2, label = "Galactic Centre photons \n Auffinger (2022) ")
plt.plot(m_LC21, fPBH_LC21, linewidth=2, label = "Galaxy clusters \nLee \& Chan (2021) [A262]")
plt.xlabel("$M_\mathrm{PBH}$ [g]")
plt.ylabel("$f_\mathrm{PBH}$")
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.legend(fontsize='small')
plt.xlim(1e15, 2e17)
plt.ylim(1e-4, 1)