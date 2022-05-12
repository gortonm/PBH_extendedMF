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
u_T = 1  # threshold impact parameter, chosen to reproduce Fig. 8

def v_r(x, u_min, m_pbh, t_hat):
    return 2 * einstein_radius(x, m_pbh) * np.sqrt(u_T - u_min**2)

def dgamma_integrand(x, u_min, args=(m_pbh, t_hat)):
    prefactor = 2 * f_pbh * d_s / (m_pbh)
    first_term = rho_M31(x) * v_r(x, u_min, m_pbh, t_hat)**4 * np.exp(-(v_r(x, u_min, m_pbh, t_hat) / v_0_M31)**2) / (v_0_M31 **2 * np.sqrt(u_T - u_min**2))
    second_term = rho_MW(x) * v_r(x, u_min, m_pbh, t_hat)**4 * np.exp(-(v_r(x, u_min, m_pbh, t_hat) / v_0_M31)**2) / (v_0_MW **2 * np.sqrt(u_T - u_min**2))
    return prefactor * (first_term + second_term)

def dgamma(m_pbh, t_hat):
    return double_integral(dgamma_integrand, x_min, x_max, 0, u_T, args=(m_pbh, t_hat)) / t_hat**4

t_hat_values = 10.**np.arange(-2, 1) / (365.25 * 24)
plt.figure()

for m_pbh in 10.**np.arange(-11, -7.9, 1):
    dgamma_values = []
    
    for t_hat in t_hat_values:
        dgamma_values.append(dgamma(m_pbh, t_hat))
        
    plt.plot(t_hat_values * 365.25 * 24, dgamma_values * (365.25 * 24)**2, label='1e{:.0f} $M_\odot$'.format(np.log10(m_pbh)))

plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel('$\hat{t}$ [hr]')
plt.ylabel('Event rate [hr]$^{-2}$')