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
density_conversion = 37.97560265 # conversion factor from GeV / cm^3 to solar masses / pc^3
c, G = 2.99792458e5 * speed_conversion, 4.30091e-3 * speed_conversion ** 2  # convert to units with [distance] = pc, [time] = yr

# Astrophysical parameters
d_s = 770e3  # M31 distance, in pc
v_0_M31 = 250 * speed_conversion  # Circular speed in M31, in pc / yr
v_0_MW = 220 * speed_conversion # Circular speed in the Milky Way, in pc / yr
r_s_M31 = 25e3 # scale radius for M31, in pc
r_s_MW = 21.5e3 # scale radius for the Milky Way, in pc
rho_s_M31 = 0.19 * density_conversion # characteristic density in M31, in solar masses / pc^3
rho_s_MW = 0.184 * density_conversion # characteristic density of Milky Way, in solar masses / pc^3
b, l = np.radians(-21.6), np.radians(121.2) # M31 galactic coordinates, in radians
sun_distance = 8.5e3 # distance of Sun from centre of MW
r_sol = 2.25461e-8  # solar radius, in pc

# Subaru-HSC survey parameters
tE_min = 2 / (365.25 * 24 * 60) # minimum event duration observable by Subaru-HSC (in yr)
tE_max = 7 / (365.25 * 24) # maximum event duration observable by Subaru-HSC (in yr)
exposure = 8.7e7 * tE_max # exposure for observing M31, in star * yr

n_exp = 4.74 # 95% confidence limit on the number of expected events, for a single candidate event


# minimum and maximum values of x to integrate over
x_min = 1e-5
x_max = 1 - x_min
# can't use x=0, 1 since this gives r=0 in the DM density

def load_data(filename):
    return np.genfromtxt(filepath+filename, delimiter=',', unpack=True)

r_source_Rsol, pdf_load = load_data('mean_R_pdf.csv')
# convert units of source radius from solar radii to parsecs
r_source_pc = r_source_Rsol * r_sol

# normalise PDF
normalisation_factor = 1 / (np.trapz(pdf_load, r_source_Rsol))
pdf_normed = normalisation_factor * pdf_load

def rho_NFW(r, r_s, rho_s):
    return rho_s / ((r/r_s) * (1 + r/r_s)**2)

def rho_MW(x): # DM density in Milky Way
    r_MW = np.sqrt(sun_distance**2 - 2*x*sun_distance*np.cos(b)*np.cos(l) + (x*d_s**2))
    return rho_NFW(r_MW, r_s_MW, rho_s_MW)    

def rho_M31(x): # DM density in M31
    r_M31 = d_s * (1-x)
    return rho_NFW(r_M31, r_s_M31, rho_s_M31)

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

# Linearly interpolated version of the PDF of source radii
def pdf_source_radii(r_source):
    return np.interp(r_source, r_source_pc, pdf_normed, left=0, right=0)

# Function for u_134 with r_S:
filepath = './Data_files/'
r_values_load, u_134_values_load = load_data('u_134.csv')
def u_134(r_S):
    return np.interp(r_S, r_values_load, u_134_values_load, left=1, right=0)

filepath = './Extracted_files/'

# Scaled source radius
def r_S(x, r_source, m_pbh):
    return x * r_source / einstein_radius(x, m_pbh)

def v_E(x, t_E, r_source, m_pbh):
    #print('r_S = ', r_S(x, r_source, m_pbh))
    #print('u_1.34 = ', u_134(r_S(x, r_source, m_pbh)))
    #print('R_E = ', einstein_radius(x, m_pbh))
    #print('t_E =', t_E)
    #print('v_E = ', 2 * u_134(r_S(x, r_source, m_pbh)) * einstein_radius(x, m_pbh) / t_E)
    return 2 * u_134(r_S(x, r_source, m_pbh)) * einstein_radius(x, m_pbh) / t_E

def kernel_integrand(x, t_E, r_source, m_pbh):
    #print('new')
    #print(r_source) #=0
    #print(pdf_source_radii(r_source)) # =0
    #print(efficiency(t_E))
    #print(rho_DM(x))
    #print(v_E(x, t_E, r_source, m_pbh)**4) # =0
    #print(np.exp(-( v_E(x, t_E, r_source, m_pbh) / v_0)**2))
    return pdf_source_radii(r_source) * efficiency(t_E) * ( (rho_MW(x) * v_E(x, t_E, r_source, m_pbh)**4 * np.exp(-( v_E(x, t_E, r_source, m_pbh) / v_0_MW)**2) + (rho_M31(x) * v_E(x, t_E, r_source, m_pbh)**4 * np.exp(-( v_E(x, t_E, r_source, m_pbh) / v_0_M31)**2))))
    
def log_normal_MF(f_pbh, m, m_c):
    return f_pbh * np.exp(-np.log(m/m_c)**2 / (2*sigma**2)) / (np.sqrt(2*np.pi) * sigma * m)

def double_integral(f, x_a, x_b, y_a, y_b, args=(), n_steps=10000):
    
    # f: function to integrate
    x_values = np.linspace(x_a, x_b, n_steps)
    y_values = np.linspace(y_a, y_b, n_steps)

    first_integral_fixed_y = []
    for y in y_values:
        
        integrand_1 = []
        
        for x in x_values:
            integrand_1.append(f(x, y, args))
            
        first_integral_fixed_y.append(np.trapz(integrand_1, x_values))
                                          
    integrand = np.trapz(first_integral_fixed_y, y_values)
    return integrand

def triple_integral(f, x_a, x_b, y_a, y_b, z_a, z_b, args=(), n_steps=100):
    
    # Possible ways to speed up:
        # passing arrays of x_values, y_values, z_values as function arguments, so they are not redefined each time
    
    
    # f: function to integrate
    x_values = np.linspace(x_a, x_b, n_steps)
    y_values = np.linspace(y_a, y_b, n_steps)
    z_values = np.linspace(z_a, z_b, n_steps)
        
    second_integral_fixed_z = []
        
    for z in z_values:
        
        first_integral_fixed_y = []
          
        for y in y_values:
            
            integrand_1 = [f(x, y, z, args) for x in x_values]
            
            #print(integrand_1)
                            
            first_integral_fixed_y.append(np.trapz(integrand_1, x_values))
        
        second_integral_fixed_z.append(np.trapz(first_integral_fixed_y, y_values))
    
    integral = np.trapz(second_integral_fixed_z, z_values)

    return integral
    
def triple_integral_test_func(x, y, z, k=1):
    return k * (z*x**2 + 4*y*z**3)

def double_integral_test_func(x, y, k=1):
    return k * (x**2 + 4*y)

from scipy.integrate import tplquad
def kernel(m_pbh):
    # kernel_integrand: A Python function or method of at least three variables in the order (z, y, x).
    return (2 * exposure * d_s / v_0**2) * triple_integral(kernel_integrand, x_min, x_max, tE_min, tE_max, min(r_source_Rsol), max(r_source_Rsol), m_pbh)
    #return tplquad(kernel_integrand, r_source_pc[1], max(r_source_pc), lambda x: tE_min, lambda x: tE_max, lambda x, y: x_min, lambda x, y: x_max, args=([m_pbh]))
""" General methods, applicable to any constraint """

def f_max(a_exp, m_pbh):
    print(kernel(m_pbh))
    return a_exp / kernel(m_pbh)[0]

def integrand(m, m_c, f_pbh, a_exp):
    integrand = []
    for i in range(len(m)):
        print('LN MF', log_normal_MF(f_pbh, m[i], m_c))
        print('kernel = ', kernel(m[i]))
        integrand.append(log_normal_MF(f_pbh, m[i], m_c) * kernel(m[i]) / a_exp)
    return integrand



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
    n_max = 100

    # Evaporation constraints (from gamma-rays)
    """
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

    # calculate constraints for extended MF from evaporation
    f_pbh_evap = []
    f_pbh_evap_extrapolated = []
    
    for m_c in mc_evaporation:
        
        def f_constraint_function_evap(f_pbh):
            return np.trapz(integrand(m_evaporation_mono, m_c, f_max_evaporation_mono, f_pbh), m_evaporation_mono) - 1
        
        f_pbh_evap.append(findroot(f_constraint_function_evap, 1, 1e-4, tolerance = 1e-8, n_max = n_max))
    
    for m_c in mc_evaporation_extrapolated:
        
        def f_constraint_function_evap_extrapolated(f_pbh):
            return np.trapz(integrand(m_range, m_c, f_max_evaporation_mono_extrapolated, f_pbh), m_range) - 1
    
        f_pbh_evap_extrapolated.append(findroot(f_constraint_function_evap_extrapolated, 1, 1e-4, tolerance = 1e-8, n_max = n_max))
    """
    # Subaru-HSC constraints
    mc_subaru = 10**np.linspace(-11.5, -4.5, 100)
    m_subaru_mono, f_max_subaru_mono = load_data('Subaru-HSC_mono.csv')
    mc_subaru_LN, f_pbh_subaru_LN = load_data('Subaru-HSC_LN.csv')
    
    
    
    # try reproducing monochromatic Subaru-HSC constraints
    f_pbh_subaru_mono_calculated = []
    for m in mc_subaru:
        print(m)
        f_pbh_subaru_mono_calculated.append( kernel(m) / n_exp )
    
    
    # Test plot
    plt.figure(figsize=(12,8))
    #plt.plot(m_evaporation_mono, f_max_evaporation_mono, linewidth = 3, label='Evaporation (extracted)', color='violet')
    plt.plot(m_subaru_mono, f_max_subaru_mono, linewidth = 3, label='Subaru-HSC (extracted)')
    plt.plot(mc_subaru, f_pbh_subaru_mono_calculated, linewidth = 3, label='Subaru-HSC (calculated)')
    #plt.plot(m_range, f_max_evaporation_mono_extrapolated, linewidth = 3, label='Evaporation (power-law fit)', color='k', linestyle='dotted')
    
    plt.xlabel('$M_\mathrm{PBH}~[M_\odot]$')
    plt.ylabel('$f_\mathrm{PBH}$')
    plt.title('Monochromatic')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-4, 1)
    plt.legend()
    #plt.savefig('./Figures/Extracted_constraints.png')
    """
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
    """
    
    # calculate constraints for extended MF from Subaru-HSC
    f_pbh_subaru = []
    for m_c in mc_subaru:
        
        def f_constraint_function_subaru(f_pbh):
            return np.trapz(integrand(m_subaru_mono, m_c, f_pbh, a_exp=n_exp), m_subaru_mono) - 1
       
        f_pbh_subaru.append(findroot(f_constraint_function_subaru, 1, 1e-4, tolerance = 1e-4, n_max = n_max))
    
        print(np.log10(m_c))    
    
    
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
    #plt.savefig('./Figures/HSC_initial.png')
