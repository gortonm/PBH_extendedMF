#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 12:11:42 2022
@author: ppxmg2
"""

from reproduce_extended_MF import *
import numpy as np
import matplotlib.pyplot as plt

f_pbh = 1
u_0 = 0
u_T = 1  # threshold impact parameter, chosen to reproduce Fig. 8 of Niikura+ '19
n_steps = 1000

x_min = 0
x_max = 1

tol = 1e-12  # tolerance for comparing whether u_min = u_1.34

hours_to_years = 1 / (365.25 * 24)  # conversion factor from a time in hours to a time in years
minutes_to_years = hours_to_years / 60

# Subaru-HSC survey parameters
tE_min = 2 * minutes_to_years # minimum event duration observable by Subaru-HSC (in yr)
tE_max = 7 * hours_to_years # maximum event duration observable by Subaru-HSC (in yr)
exposure = 8.7e7 * tE_max # exposure for observing M31, in star * yr

n_exp = 4.74 # 95% confidence limit on the number of expected events, for a single candidate event

f_pbh = 1
def dgamma_integrand_MW(x, u_min, m_pbh, t_hat):
    if abs(u_min - u_T) < tol:
        return 0
    return rho_MW(x) * einstein_radius(x, m_pbh)**4 * (u_T**2 - u_min**2)**1.5 * np.exp(-(4 * einstein_radius(x, m_pbh)**2 * (u_T**2 - u_min**2) / (t_hat * v_0_MW)**2)) / v_0_MW**2
    
def dgamma_integrand_M31(x, u_min, m_pbh, t_hat):
    if x == 1:
        return 0
    if abs(u_min - u_T) < tol:
        return 0
    return rho_M31(x) * einstein_radius(x, m_pbh)**4 * (u_T**2 - u_min**2)**1.5 * np.exp(-(4 * einstein_radius(x, m_pbh)**2 * (u_T**2 - u_min**2) / (t_hat * v_0_M31)**2)) / v_0_M31**2

def dgamma_MW(m_pbh, t_hat):
    prefactor = 32 * f_pbh * d_s / (m_pbh * t_hat ** 4)
    return prefactor * double_integral(dgamma_integrand_MW, x_min, x_max, u_0, u_T - u_0, n_steps, m_pbh, t_hat)

def dgamma_M31(m_pbh, t_hat):
    prefactor = 32 * f_pbh * d_s / (m_pbh * t_hat ** 4)
    return prefactor * double_integral(dgamma_integrand_M31, x_min, x_max, u_0, u_T - u_0, n_steps, m_pbh, t_hat)

def calc_f_pbh(m_pbh):
    t_hat_values = 2 * np.linspace(2*minutes_to_years, 7*hours_to_years, n_steps)
    
    # factor of 0.5 to account for the efficiency function, approximated as a constant between t_min and t_max
    return n_exp / (0.5 * exposure * np.trapz(dgamma_MW(m_pbh, t_hat_values) + dgamma_M31(m_pbh, t_hat_values), t_hat_values))

plot_dgamma = True

if plot_dgamma:
    m_pbh_values = 10.**np.arange(-12, -6, 1.)
    t_hat_values = np.linspace(2*tE_min, 2*tE_max, 50)
    
    for m_pbh in m_pbh_values:
        print(m_pbh)
        dgamma_values = []
        
        for t_hat in t_hat_values:
            dgamma_values.append(dgamma_MW(m_pbh, t_hat) + dgamma_M31(m_pbh, t_hat))
        
        plt.plot(np.array(t_hat_values) / hours_to_years, np.array(dgamma_values) / hours_to_years**2, label='$10^{:.0f}M_\odot$'.format(np.log10(m_pbh)))
        
    plt.xlabel(r'$\hat{t}$ [hr]')
    plt.ylabel('Event rate [hr]$^{-2}$')
    plt.xscale('log')
    plt.yscale('log')


if "__main__" == __name__:
    t_hat_values = 10.**np.arange(-2.5, 1, 0.1) * hours_to_years
    plt.figure()
    
    f_pbh_values = []
    m_pbh_values = 10.**np.arange(-15, -10, 1)
    for m_pbh in m_pbh_values:
        print(m_pbh)
        f_pbh_values.append(calc_f_pbh(m_pbh))
    
    plt.plot(m_pbh_values, f_pbh_values)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e-12, 1e-5)
    plt.ylim(1e-3, 1)
    plt.legend()
    plt.xlabel(r'$M_\mathrm{PBH}~[M_\odot]$')
    #plt.ylabel('f_\mathrm{PBH}$')
    plt.tight_layout()