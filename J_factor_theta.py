#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 11:51:13 2022

@author: ppxmg2
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Script to compare the fluxes used in the calculations of Auffinger '22
# Fig. 3 (2201.01265), to the constraints from Coogan, Morrison & Profumo '21
# (2010.04797)

# Specify the plot style
mpl.rcParams.update({'font.size': 24,'font.family':'serif'})
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

pc_to_cm = 3.09e18    # convert pc to cm
MeV_to_Modot = 8.96260432e-61    # convert MeV / c^2 to grams

# Astrophysical parameters

# CMP '21 (values from de Salas et al. '19)
rho_0_CMP = 0.376     # DM density at the Sun, in GeV / cm^3
r_s_CMP = 11 * 1e3   # scale radius, in pc
r_0_CMP = 8.12 * 1e3    # galactocentric distance of Sun, in pc
gamma = 1

"""
# CMP '21 (generalised NFW values from de Salas et al. '19)
rho_0_CMP = 0.387     # DM density at the Sun, in GeV / cm^3
r_s_CMP = 8.1 * 1e3   # scale radius, in pc
r_0_CMP = 8.12 * 1e3    # galactocentric distance of Sun, in pc
gamma = 1.3
"""
# Attempt to calculate J-factor for CMP '21

MeV_to_Modot = 8.96260432e-61

def rho_NFW(r):
    rho_0 = rho_0_CMP
    r_s = r_s_CMP
    return rho_0 * ((r_s/r)**gamma * (1 + (r/r_s)))**(gamma-3)

def r(los, theta):
    return np.sqrt(los**2 + r_0_CMP**2 - 2*r_0_CMP*los*np.cos(theta))

def j_psi(psi, n_steps=10000):
    los_values = np.linspace(0, r_0_CMP, n_steps)
    return np.trapz(rho_NFW(r(los_values, psi)), los_values)

def j_theta(theta_max):
    delta_omega = 2 * np.pi * (1-np.cos(theta_max))
    print(delta_omega)
    theta_values = np.linspace(1e-5, theta_max, 10000)
    
    # try calculating value sinfor integrand = theta * j_psi(theta)
    integrand_values = [theta * j_psi(theta) for theta in theta_values]
    integral = (2 * np.pi / (delta_omega)) * np.trapz(integrand_values, theta_values) * (1/MeV_to_Modot) * (1/pc_to_cm)**2
    print(integral / 1.597e26)
    
    # try calculating value for integrand = theta * j_psi(theta)
    integrand_values = [np.sin(theta) * j_psi(theta) for theta in theta_values]
    integral = (2 * np.pi / (delta_omega)) * np.trapz(integrand_values, theta_values) * 1000 * (1/pc_to_cm)
    print(integral / 1.597e26)
    
    # Units:
        # [j_psi] = [GeV pc cm^{-3}]
        # [integrand_values] = [GeV pc cm^{-3}]
        # [integral] = [1000 * GeV pc cm^{-3} sr^{-1} cm/pc] = [MeV cm^{-2} sr^{-1}]
    
    integrand_values = [np.sin(theta) * j_psi(theta) for theta in theta_values]
    return (2 * np.pi / (delta_omega)) * np.trapz(integrand_values, theta_values) * (1/MeV_to_Modot) * (1/pc_to_cm)**2

CMP21 = True
print(j_theta(np.radians(5)) / 1.597e26)
    