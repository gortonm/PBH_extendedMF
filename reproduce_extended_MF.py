#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 11:46:30 2022

@author: ppxmg2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Specify the plot style
mpl.rcParams.update({'font.size': 30,'font.family':'serif'})
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


sigma = 2
filepath = './Extracted_files/'

""" Methods for Subaru-HSC constraints """
speed_conversion = 1.022704735e-6  # conversion factor from km/s to pc/yr
density_conversion = 0.026339714 # conversion factor from GeV / cm^3 to solar masses / pc^3

# Astrophysical parameters
d_s = 770e3  # M31 distance, in pc
v_0 = 250 * speed_conversion  # Circular speed in M31, in pc / yr
c, G = 2.99792458e5 * speed_conversion, 4.30091e-3 * speed_conversion ** 2  # convert to units with [distance] = pc, [time] = yr
r_s = 25e3 # scale radius for M31, in pc
rho_s = 0.19 * density_conversion # characteristic density in M31, in solar masses / pc^3

# Subaru-HSC survey parameters
tE_min = 2 / (365.25 * 24 * 60) # minimum event duration observable by Subaru-HSC (in yr)
tE_max = 7 / (365.25 * 24) # maximum event duration observable by Subaru-HSC (in yr)
exposure = 8.7e7 * tE_max # exposure for observing M31, in star * yr

n_exp = 4.74 # 95% confidence limit on the number of expected events, for a single candidate event

def u_134(x):
    """ Temporary, needs updating """
    return 1

def rho_DM(x): # DM density in M31
    r = d_s * (1-x)
    return rho_s / ((r/r_s) * (1 + r/r_s)**2)

def efficiency(t_E):
    if tE_min < t_E < tE_max:
        return 0.5
    else:
        return 0

def einstein_radius(x, m_pbh):
    """
    Calculate Einstein radius of a lens.

    Parameters
    ----------
    x : Float
        Fractional line-of-sight distance to M31.
    m_pbh : Float
        Lens mass, in solar masses.

    Returns
    -------
    Float
        Einstein radius of a lens at line-of-sight distance d_L, in pc.

    """
    return 2 * np.sqrt(G * m_pbh * d_s * x * (1-x) / c ** 2)

def v_E(x, m_pbh, t_E):
    return 2 * u_134(x) * einstein_radius(x, m_pbh) / t_E

def kernel_integrand(x, m_pbh, t_E):
    return (2 * exposure * d_s / v_0**2) * efficiency(t_E) * rho_DM(x) * v_E(x, m_pbh, t_E)**4 * np.exp(-( v_E(x, m_pbh, t_E) / v_0)**2)
    

def log_normal_MF(f_pbh, m, m_c):
    return f_pbh * np.exp(-np.log(m/m_c)**2 / (2*sigma**2)) / (np.sqrt(2*np.pi) * sigma * m)

def left_riemann_sum(y, x):
    return np.sum(np.diff(x) * y[:-1])

def kernel(m_pbh, n_steps = 10000):    
    tE_values = np.linspace(tE_min, tE_max, n_steps)
    x_values = np.linspace(0, 1, n_steps)
    
    integral = 0
    
    # Double integration, needs checking
    for t_E in np.linspace(tE_min, tE_max, n_steps):
        
        internal_integral = 0
        
        for x in x_values:
            internal_integral += np.sum(kernel_integrand(x, m_pbh, t_E)) * np.diff(x_values)
    
        integral += np.sum(internal_integral * np.diff(tE_values))
        
    return integral

""" General methods, applicable to any constraint """

def f_max(a_exp, m_pbh):
    return a_exp / kernel(m_pbh)

def integrand(m, m_c, f_max, f_pbh):
    integrand = []
    for i in range(len(m)):
        integrand.append(log_normal_MF(f_pbh, m[i], m_c) / f_max(m[i]))
    return integrand



def load_data(filename):
    return np.genfromtxt(filepath+filename, delimiter=',', unpack=True)

def findroot(f, a, b, tolerance, n_max):
    n = 1
    while n <= n_max:
        c = (a + b) / 2
        #print(c)
        #print(f(c))
        
        if f(c) == 0 or abs((b - a) / 2) < tolerance:
            return c
            break
        n += 1
        
        # set new interval
        if np.sign(f(c)) == np.sign(f(a)):
            a = c
        else:
            b = c
    print("Method failed")


if "__main__" == __name__:    

    # Evaporation constraints (from gamma-rays)
    mc_evaporation = 10**np.linspace(-18, -15, 100)
    m_evaporation_mono, f_max_evaporation_mono = load_data('Gamma-ray_mono.csv')
    mc_evaporation_LN, f_pbh_evaporation_LN = load_data('Gamma-ray_LN.csv')
    
    
    # approximate the evaporation constraints using a power law
    power = (np.log10(f_max_evaporation_mono[-1]) - np.log10(f_max_evaporation_mono[0])) / (np.log10(m_evaporation_mono[-1]) - np.log10(m_evaporation_mono[0]))
    prefactor = f_max_evaporation_mono[0] * (m_evaporation_mono[0]) ** (-power)
    prefactor2 = f_max_evaporation_mono[-1] * (m_evaporation_mono[-1]) ** (-power)
    print(power)
    
    m_range = 10**np.arange(-20, -15.9, 0.05)
    f_max_evaporation_mono_extrapolated = prefactor * np.power(m_range, power)
    
    # Range of characteristic masses in log-normal mass function for the power-law approximation
    mc_evaporation_extrapolated = 10**np.linspace(-15, -13.5, 100)
    
    # Subaru-HSC constraints
    mc_subaru = 10**np.linspace(-11.5, -4.5, 100)
    m_subaru_mono, f_max_subaru_mono = load_data('Subaru-HSC_mono.csv')
    mc_subaru_LN, f_pbh_subaru_LN = load_data('Subaru-HSC_LN.csv')
    
    
    # calculate constraints for extended MF from evaporation
    f_pbh_evap = []
    f_pbh_evap_extrapolated = []
    n_max = 1000
    
    for m_c in mc_evaporation:
        
        def f_constraint_function_evap(f_pbh):
            return left_riemann_sum(integrand(m_evaporation_mono, m_c, f_max_evaporation_mono, f_pbh), m_evaporation_mono) - 1
        
        f_pbh_evap.append(findroot(f_constraint_function_evap, 1, 1e-4, tolerance = 1e-8, n_max = n_max))
    
    for m_c in mc_evaporation_extrapolated:
        
        def f_constraint_function_evap_extrapolated(f_pbh):
            return left_riemann_sum(integrand(m_range, m_c, f_max_evaporation_mono_extrapolated, f_pbh), m_range) - 1
    
        f_pbh_evap_extrapolated.append(findroot(f_constraint_function_evap_extrapolated, 1, 1e-4, tolerance = 1e-8, n_max = n_max))
    
    # calculate constraints for extended MF from Subaru-HSC
    f_pbh_subaru = []
    for m_c in mc_subaru:
        
        def f_constraint_function_subaru(f_pbh):
            return left_riemann_sum(integrand(m_subaru_mono, m_c, f_max_subaru_mono, f_pbh), m_subaru_mono) - 1
       
        f_pbh_subaru.append(findroot(f_constraint_function_subaru, 1, 1e-4, tolerance = 1e-8, n_max = n_max))
    
    
    # Test plot
    plt.figure(figsize=(12,8))
    plt.plot(m_evaporation_mono, f_max_evaporation_mono, linewidth = 3, label='Evaporation (extracted)', color='violet')
    plt.plot(m_subaru_mono, f_max_subaru_mono, linewidth = 3, label='Subaru-HSC (extracted)', color='tab:blue')
    plt.plot(m_range, f_max_evaporation_mono_extrapolated, linewidth = 3, label='Evaporation (power-law fit)', color='k', linestyle='dotted')
    
    plt.xlabel('$M_\mathrm{PBH}~[M_\odot]$')
    plt.ylabel('$f_\mathrm{PBH}$')
    plt.title('Monochromatic')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-4, 1)
    plt.legend()
    plt.savefig('./Figures/Extracted_constraints.png')
    
    # Test plot
    plt.figure(figsize=(12,8))
    plt.plot(mc_evaporation[:-1], f_pbh_evap[:-1], linewidth = 3, label='Computed')
    plt.plot(mc_evaporation_LN, f_pbh_evaporation_LN, linewidth = 3, label='Extracted')
    #plt.plot(mc_evaporation_extrapolated, f_pbh_evap_extrapolated, linewidth = 3, label='Computed \n (power-law fit)')
    plt.xlabel('$M_c ~[M_\odot]$')
    plt.ylabel('$f_\mathrm{PBH}$')
    plt.title('Log-normal $(\sigma = 2)$')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-4, 1)
    plt.legend()
    plt.savefig('./Figures/evaporation_initial.png')
    
    plt.figure(figsize=(12,8))
    plt.plot(mc_subaru, f_pbh_subaru, linewidth = 3, label='Computed', color='tab:orange')
    plt.plot(mc_subaru_LN[2:-5], f_pbh_subaru_LN[2:-5], linewidth = 3, label='Extracted', color='tab:blue')
    plt.xlabel('$M_c ~[M_\odot]$')
    plt.ylabel('$f_\mathrm{PBH}$')
    plt.title('Log-normal $(\sigma = 2)$')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-3, 1)
    plt.legend()
    plt.savefig('./Figures/HSC_initial.png')
