#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 13:54:40 2022

@author: ppxmg2
"""

import numpy as np

# Reproduce results from Cirelli et al. (2011) (arxiv:1012.4515)

r_s = 24.42
rho_s = 0.184

rho_odot = 0.3
r_0 = 8.33

n_steps = 1000

def find_r(los, theta):
    return np.sqrt(r_0**2 + los**2 - 2*r_0*los*np.cos(theta))

def rho_NFW(los, theta):
    r = find_r(los, theta)
    return rho_s * (r_s / r) * (1 + (r/r_s))**(-2)


def j(theta):
    los_values = np.linspace(0, 0.9999*r_0, n_steps)
    j_integrand = [(rho_NFW(los, theta) / rho_odot)**2 / (r_0) for los in los_values]
    return np.trapz(j_integrand, los_values)
    #return np.sum(j_integrand) * (los_values[1] - los_values[0])

def j_avg(theta_max):
    delta_omega = 2*np.pi*(1-np.cos(theta_max))
    print(delta_omega)
    theta_values = np.linspace(0, theta_max, n_steps)
    
    integrand = [j(theta) * np.sin(theta) for theta in theta_values]
    return 2 * np.pi * np.trapz(integrand, theta_values) / delta_omega

theta_max = np.radians(2)
print(j_avg(theta_max))