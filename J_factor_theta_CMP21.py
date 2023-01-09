#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 13:54:40 2022

@author: ppxmg2
"""

import numpy as np

# Reproduce results from Coogan, Morrison & Profumo (2021)

r_s = 11    # scale radius, in kpc
r_odot = 8.33    # galactocentric solar radius, in kpc
rho_DM_odot = 0.376 * 1000   # DM density at the solar radius, in MeV cm^{-3}
rho_0 = rho_DM_odot * (1 + (r_odot/r_s))**2 * (r_odot/r_s)   # normaliseation for NFW profile, in MeV cm^{-3}

pc_to_cm = 3.0857e18    # conversion factor from pc to cm

n_steps = 1000

#%% For a disk-shaped region (of angular width theta_max)

def find_r(los, theta):
    return np.sqrt(r_odot**2 + los**2 - 2*r_odot*los*np.cos(theta))

def rho_NFW(los, theta):
    r = find_r(los, theta)
    return rho_0 * (r_s / r) * (1 + (r/r_s))**(-2)

def j(theta):
    los_values = np.linspace(0, 0.9999*r_odot, n_steps)
    j_integrand = [rho_NFW(los, theta) for los in los_values]
    return np.trapz(j_integrand, los_values)
    #return np.sum(j_integrand) * (los_values[1] - los_values[0])

def j_avg(theta_max):
    delta_omega = 2*np.pi*(1-np.cos(theta_max))
    print(delta_omega)
    theta_values = np.linspace(0, theta_max, n_steps)
    
    integrand = [j(theta) * np.sin(theta) for theta in theta_values]
    return 2 * np.pi * np.trapz(integrand, theta_values) * 1000 * pc_to_cm / delta_omega

theta_max = np.radians(5)
print(j_avg(theta_max))


#%% For a range of galactic coordinates (b, l)
from scipy.integrate import tplquad

def find_r(los, b, l):
    return np.sqrt(r_odot**2 + los**2 - 2*r_odot*los*np.cos(b)*np.cos(l))

def rho_NFW(los, b, l):
    r = find_r(los, b, l)
    return rho_0 * (r_s / r) * (1 + (r/r_s))**(-2)

def j_integrand(los, b, l):
    return rho_NFW(los, b, l) * np.cos(b)

def j_avg(b_max, l_max):
    delta_omega = 4*np.sin(b_max)*l_max
    print(delta_omega)
    
    return 4 * np.array(tplquad(j_integrand, 0, l_max, 0, b_max, 0, 0.99999*r_odot)) * 1000 * pc_to_cm / delta_omega

b_max, l_max = np.radians(20), np.radians(60)
print(j_avg(b_max, l_max))

