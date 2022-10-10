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


#%% Calculate J-factor for the parameter values used in Auffinger '22

r_s = 17 * 1000    # scale radius, in pc
r_odot = 8.5 * 1000   # galactocentric solar radius, in pc
rho_0 = 0.0125   # normaliseation for NFW profile, in M_\odot pc^{-3}
r_max = 200 * 1000    # maximum halo radius, in pc

# Isatis reproduction values
pc_to_cm = 3.0857e18    # conversion factor from pc to cm
r_s = 17 * 1000 * pc_to_cm    # scale radius, in cm
r_odot = 8.5 * 1000 * pc_to_cm   # galactocentric solar radius, in cm
rho_0 = 8.5e-25	# characteristic halo density in g/cm^3
r_max = 200 * 1000 * pc_to_cm    # maximum halo radius, in cm

from scipy.integrate import tplquad

def find_r(los, b, l):
    return np.sqrt(r_odot**2 + los**2 - 2*r_odot*los*np.cos(b)*np.cos(l))

def rho_NFW(los, b, l):
    r = find_r(los, b, l)
    return rho_0 * (r_s / r) * (1 + (r/r_s))**(-2)

def j_integrand(los, b, l):
    return rho_NFW(los, b, l) * np.cos(b)

def s_max(l, b):
    return (r_odot * np.cos(b) * np.cos(l)) + np.sqrt(r_max**2 - r_odot**2 * (1- (np.cos(b)*np.cos(l))**2))

def j_avg(b_max, l_max):
    delta_omega = 4*np.sin(b_max)*l_max        
    return 4 * np.array(tplquad(j_integrand, 0, l_max, 0, b_max, 0, s_max)) / delta_omega

if "__main__" == __name__:
    b_max, l_max = np.radians(15), np.radians(30)
    print(j_avg(b_max, l_max))
