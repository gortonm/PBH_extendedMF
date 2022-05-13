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
u_0 = 1e-5
u_T = 1  # threshold impact parameter, chosen to reproduce Fig. 8 of Niikura+ '19
n_steps = 1000

def v_r(x, u_min, m_pbh, t_hat):
    return 2 * einstein_radius(x, m_pbh) * np.sqrt(u_T**2 - u_min**2) / t_hat

def dgamma_integrand_MW(x, u_min, m_pbh, t_hat):
    prefactor = 2 * f_pbh * d_s / (m_pbh)
    first_term = rho_MW(x) * v_r(x, u_min, m_pbh, t_hat)**4 * np.exp(-(v_r(x, u_min, m_pbh, t_hat) / v_0_MW)**2) / (v_0_MW **2 * np.sqrt(u_T**2 - u_min**2))
    return prefactor * first_term

def dgamma_integrand_M31(x, u_min, m_pbh, t_hat):
    prefactor = 2 * f_pbh * d_s / (m_pbh)
    first_term = rho_M31(x) * v_r(x, u_min, m_pbh, t_hat)**4 * np.exp(-(v_r(x, u_min, m_pbh, t_hat) / v_0_M31)**2) / (v_0_M31 **2 * np.sqrt(u_T**2 - u_min**2))
    return prefactor * first_term

def dgamma_MW(m_pbh, t_hat):
    return double_integral(dgamma_integrand_MW, x_min, x_max, u_0, u_T - u_0, n_steps, m_pbh, t_hat)

def dgamma_M31(m_pbh, t_hat):
    return double_integral(dgamma_integrand_M31, x_min, x_max, u_0, u_T - u_0, n_steps, m_pbh, t_hat)


hours_to_years = 1 / (365.25 * 24)
t_hat_values = 10.**np.arange(-2, 1, 0.1) * hours_to_years
plt.figure()

for m_pbh in 10.**np.arange(-8, -7.9, 1):
    dgamma_MW_values = []
    dgamma_M31_values = []
    print(m_pbh)
    
    for t_hat in t_hat_values:
        dgamma_MW_values.append(dgamma_MW(m_pbh, t_hat))
        dgamma_M31_values.append(dgamma_M31(m_pbh, t_hat))
        
    plt.plot(np.array(t_hat_values) / hours_to_years, np.array(dgamma_MW_values) / hours_to_years**2, label='MW')
    plt.plot(np.array(t_hat_values) / hours_to_years, np.array(dgamma_M31_values) / hours_to_years**2, label='M31')
    plt.plot(np.array(t_hat_values) / hours_to_years, (np.array(dgamma_M31_values) + np.array(dgamma_MW_values)) / hours_to_years**2, label='Total')

plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-8, 1e-5)
plt.xlim(1e-2, 10)
plt.legend()
plt.xlabel('$\hat{t}$ [hr]')
plt.ylabel('Event rate [hr]$^{-2}$')

