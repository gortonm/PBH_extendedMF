#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 12:27:34 2023

@author: ppxmg2
"""

import numpy as np
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
from reproduce_COMPTEL_constraints_v2 import load_data, read_blackhawk_spectra

# Compare flux data from Isatis flux file to that in the sources cited.

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


file_path_extracted = './Extracted_files/'

# Load flux data extracted from sources cited in Isatis code

E_COMPTEL_lower, flux_COMPTEL_lower = load_data("Bouchet_Fig7_COMPTEL_lower_xy.csv") * 1e-3
E_COMPTEL_upper, flux_COMPTEL_upper = load_data("Bouchet_Fig7_COMPTEL_upper_xy.csv") * 1e-3

E_COMPTEL = 10 ** (0.5 * (np.log10(E_COMPTEL_lower) + np.log10(E_COMPTEL_upper)))
flux_COMPTEL = 10 ** (0.5 * (np.log10(flux_COMPTEL_lower) + np.log10(flux_COMPTEL_upper)))
yerr_COMPTEL = [flux_COMPTEL_upper - flux_COMPTEL, flux_COMPTEL - flux_COMPTEL_lower]
xerr_COMPTEL = [E_COMPTEL_upper - E_COMPTEL, E_COMPTEL - E_COMPTEL_lower]

_, flux_INTEGRAL_lower = load_data("Bouchet_Fig7_INTEGRAL_lower_y.csv") * 1e-3
E_INTEGRAL, flux_INTEGRAL_upper = load_data("Bouchet_Fig7_INTEGRAL_upper_y.csv") * 1e-3

flux_INTEGRAL = 10 ** (0.5 * (np.log10(flux_INTEGRAL_lower) + np.log10(flux_INTEGRAL_upper)))
yerr_INTEGRAL = [flux_INTEGRAL_upper - flux_INTEGRAL, flux_INTEGRAL - flux_INTEGRAL_lower]

E_EGRET_lower, flux_EGRET_lower = load_data("EGRET_lower_xy.csv") * 1e-3
E_EGRET_upper, flux_EGRET_upper = load_data("EGRET_upper_xy.csv") * 1e-3

E_EGRET = 10 ** (0.5 * (np.log10(E_EGRET_lower) + np.log10(E_EGRET_upper)))
flux_EGRET = 10 ** (0.5 * (np.log10(flux_EGRET_lower) + np.log10(flux_EGRET_upper)))
yerr_EGRET = [flux_EGRET_upper - flux_EGRET, flux_EGRET - flux_EGRET_lower]
xerr_EGRET = [E_EGRET_upper - E_EGRET, E_EGRET - E_EGRET_lower]

E_FermiLAT_lower, flux_FermiLAT_lower = load_data("FermiLAT_lower_y.csv") * 1e-3
E_FermiLAT_upper, flux_FermiLAT_upper = load_data("FermiLAT_upper_y.csv") * 1e-3

E_FermiLAT = 10 ** (0.5 * (np.log10(E_FermiLAT_lower) + np.log10(E_FermiLAT_upper)))
flux_FermiLAT = 10 ** (0.5 * (np.log10(flux_FermiLAT_lower) + np.log10(flux_FermiLAT_upper)))
yerr_FermiLAT = [flux_FermiLAT_upper - flux_FermiLAT, flux_FermiLAT - flux_FermiLAT_lower]
xerr_FermiLAT = [E_FermiLAT_upper - E_FermiLAT, E_FermiLAT - E_FermiLAT_lower]


#%%

# Load flux data from Isatis (Galactic background)
Isatis_path = "./../Downloads/version_finale/scripts/Isatis/"
INTEGRAL = np.genfromtxt("%s/constraints/photons/flux_INTEGRAL_1107.0200.txt"%(Isatis_path),skip_header = 6)
COMPTEL = np.genfromtxt("%s/constraints/photons/flux_COMPTEL_1107.0200.txt"%(Isatis_path),skip_header = 6)
EGRET = np.genfromtxt("%s/constraints/photons/flux_EGRET_9811211.txt"%(Isatis_path),skip_header = 6)
FermiLAT = np.genfromtxt("%s/constraints/photons/flux_Fermi-LAT_1101.1381.txt"%(Isatis_path),skip_header = 6)

# Plot E^2 * flux (code from plotting.py in Isatis)
f = pylab.figure(3)
f.clf()
ax = f.add_subplot(111)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e-5,1e+2)
ax.set_ylim(1e-6,1e-4)
ax.errorbar(INTEGRAL[:,0],INTEGRAL[:,3]*INTEGRAL[:,0]**2,xerr = [INTEGRAL[:,1],INTEGRAL[:,2]],yerr = [INTEGRAL[:,4]*INTEGRAL[:,0]**2,INTEGRAL[:,5]*INTEGRAL[:,0]**2],fmt = "o",markersize = 2,fillstyle = None,label = "${\\rm INTEGRAL}$")
ax.errorbar(COMPTEL[:,0],COMPTEL[:,3]*COMPTEL[:,0]**2,xerr = [COMPTEL[:,1],COMPTEL[:,2]],yerr = [COMPTEL[:,4]*COMPTEL[:,0]**2,COMPTEL[:,5]*COMPTEL[:,0]**2],fmt = "o",markersize = 2,fillstyle = None,label = "${\\rm COMPTEL}$")
ax.errorbar(EGRET[:,0],EGRET[:,3]*EGRET[:,0]**2,xerr = [EGRET[:,1],EGRET[:,2]],yerr = [EGRET[:,4]*EGRET[:,0]**2,EGRET[:,5]*EGRET[:,0]**2],fmt = "o",markersize = 2,fillstyle = None,label = "${\\rm EGRET}$")
ax.errorbar(FermiLAT[:,0],FermiLAT[:,3]*FermiLAT[:,0]**2,xerr = [FermiLAT[:,1],FermiLAT[:,2]],yerr = [FermiLAT[:,4]*FermiLAT[:,0]**2,FermiLAT[:,5]*FermiLAT[:,0]**2],fmt = "o",markersize = 2,fillstyle = None,label = "${\\rm Fermi-LAT}$")
ax.grid(True)
ax.legend(loc="best", fontsize="small")

# additions to plotting.py in Isatis

ax.errorbar(E_INTEGRAL, flux_INTEGRAL, yerr=yerr_INTEGRAL, fmt = "x", color="k", alpha=0.7)
ax.errorbar(E_COMPTEL, flux_COMPTEL, xerr=xerr_COMPTEL, yerr=yerr_COMPTEL, fmt = "x", color="k", alpha=0.7)
ax.errorbar(E_EGRET, flux_EGRET, xerr=xerr_EGRET, yerr=yerr_EGRET, fmt = "x", color="k", alpha=0.7)
ax.errorbar(E_FermiLAT, flux_FermiLAT, xerr=xerr_FermiLAT, yerr=yerr_FermiLAT, fmt = "x", color="k", alpha=0.7)

ax.set_xlabel("$E$ [GeV]")
ax.set_ylabel("$E^2 \mathrm{d}\Phi / \mathrm{d}E$ $[ \mathrm{GeV} \cdot \mathrm{s}^{-1} \cdot \mathrm{sr}^{-1}]$")

f.show()
f.tight_layout(rect = [0,0,1,1])
