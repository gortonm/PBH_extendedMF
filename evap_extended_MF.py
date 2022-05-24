#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 13:32:36 2022

@author: ppxmg2
"""

# Constraints for a log-normal MF, using various assumptions about the 
# normalisation and integration limits.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import erf

# Specify the plot style
mpl.rcParams.update({'font.size': 20,'font.family':'serif'})
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


filepath = './Extracted_files/'

m_star = 5e14 / 1.989e33    # 24/5: updated value of M_*, reflecting the better fit found to the monochromatic MF constraints
sigma = 2
epsilon = 0.4

m_2 = np.power(5e9, 1/(3+epsilon)) * m_star
#m_2 = 7e16 / 1.989e33

def log_normal_MF(f_pbh, m, m_c):
    return f_pbh * np.exp(-np.log(m/m_c)**2 / (2*sigma**2)) / (np.sqrt(2*np.pi) * sigma * m)

def load_data(filename):
    return np.genfromtxt(filepath+filename, delimiter=',', unpack=True)

def integrand(f_pbh, m, m_c):
    return log_normal_MF(f_pbh, m, m_c) / f_evap(m)

def findroot(f, a, b, tolerance_1, tolerance_2, n_max):
    n = 1
    while n <= n_max:
        c = (a + b) / 2
        #print('\n' + 'New:')
        #print('c = ', c)
        #print('f(c) = ', f(c))
        
        if abs(f(c)) < tolerance_1 or abs((b - a) / 2) < tolerance_2:
        #if abs(f(c)) < tolerance_1:
            return c
            break
        n += 1
        
        # set new interval
        if np.sign(f(c)) == np.sign(f(a)):
            a = c
        else:
            b = c
    print("Method failed")
    
    
def integral_analytic(m_c, m, epsilon):
    return erf((((epsilon + 3) * sigma**2) + np.log(m/m_c))/(np.sqrt(2) * sigma)) 

def constraint_analytic(m_range, m_c, epsilon):
    prefactor = 4e-8 * (m_c / m_star)**(3+epsilon) * np.exp(-0.5 * (epsilon + 3)**2 * sigma**2) 

    return prefactor / (integral_analytic(m_c, max(m_range), epsilon) - integral_analytic(m_c, min(m_range), epsilon))


def extended_constraint_analytic(m_c):
    return 2e-8 * np.power(m_c/m_star, 3+epsilon) * np.exp(-0.5*sigma**2 * (epsilon+3)**2)


mc_evaporation = 10**np.linspace(-17, -11, 100)
m_evaporation_mono, f_max_evaporation_mono = load_data('Gamma-ray_mono.csv')

def f_evap(m):
    return np.interp(m, m_evaporation_mono, f_max_evaporation_mono)

def f_evap_formula(m, epsilon=0.4):    # Eq. 62 of Carr+ '21
    return 2e-8 * np.array(m/m_star)**(3+epsilon)

def integrand_formula(f_pbh, m, m_c):
    return log_normal_MF(f_pbh, m, m_c) / f_evap_formula(m)

def constraint_function(f_pbh):
    return np.trapz(integrand(f_pbh, m_range, m_c), m_range) - 1

n_max = 100

if "__main__" == __name__:
    mc_evaporation_LN, f_pbh_evaporation_LN = load_data('Gamma-ray_LN.csv')
    mc_evaporation_LN_Carr17, f_pbh_evaporation_LN_Carr17 = load_data('Carr_17_LN_evap.csv')

    # calculate constraints for extended MF from evaporation
    f_pbh_evap = []
    f_pbh_evap_rootfinder = []
    
    for m_c in mc_evaporation:
        
        m_range = 10**np.linspace(np.log10(min(m_evaporation_mono)), np.log10(max(m_evaporation_mono)), 10000)
        #f_pbh_evap.append(1/np.trapz(integrand(m_range, m_c, f_pbh=1), m_range))
        f_pbh_evap.append(1/np.trapz(integrand(f_pbh=1, m=m_range, m_c=m_c), m_range))
        f_pbh_evap_rootfinder.append(findroot(constraint_function, 1e-5, 10, tolerance_1=1e-4, tolerance_2=1e-4, n_max=1000))

    plt.figure(figsize=(12,8))

    m1 = m_star
    m2 = 1e18/1.989e33
    f_pbh_evap = []
    f_pbh_evap_analytic = []
    f_pbh_evap_formula = []
    for m_c in mc_evaporation:
        
        normalisation_factor = 0.5 * (erf(np.log(m2*1000/m_c) / (np.sqrt(2) * sigma)) - erf(np.log(m1/m_c) / (np.sqrt(2) * sigma)))
        
        m_range = 10**np.linspace(np.log10(m1), np.log10(m2), 10000)
        f_pbh_evap.append(1/np.trapz(integrand(f_pbh=1, m=m_range, m_c=m_c), m_range))
        
        f_pbh_evap_analytic.append(constraint_analytic(m_range, m_c, epsilon))
        f_pbh_evap_formula.append(1/np.trapz(integrand_formula(f_pbh=1, m=m_range, m_c=m_c), m_range))
        
    plt.plot(mc_evaporation, np.array(f_pbh_evap)*normalisation_factor, label='Trapezium rule $M_1 = {:.0e}$g, $M_2 = {:.0e}$g'.format(m1*1.989e33, m2*1.989e33), linestyle = 'dotted', linewidth=6)
    plt.plot(mc_evaporation, np.array(f_pbh_evap_formula)*normalisation_factor, label='Trapezium rule (using Carr+ 21 Eq. 62) \n $M_1 = {:.0e}$g, $M_2 = {:.0e}$g'.format(m1*1.989e33, m2*1.989e33), linewidth=6)
    plt.plot(mc_evaporation, np.array(f_pbh_evap_analytic)*normalisation_factor, label='Analytic $M_1 = {:.0e}$g, $M_2 = {:.0e}$g'.format(m1*1.989e33, m2*1.989e33), linestyle = 'dotted', linewidth=6)

    plt.plot(mc_evaporation_LN, f_pbh_evaporation_LN, color='k', alpha=0.25, linewidth=4, label='Extracted (Carr+ 21)')

    plt.xlabel('$M_\mathrm{c}~[M_\odot]$')
    plt.ylabel('$f_\mathrm{PBH}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.xlim(10**(-15.5), 1e-12)
    plt.ylim(1e-4, 1)
    plt.title('Log-normal MF ($\sigma = 2$)')
    plt.tight_layout()
    #plt.savefig('Figures/evap_constraints_updated_analytic_numeric.pdf')
    
    
    
if "__main__" == __name__:
    mc_evaporation_LN, f_pbh_evaporation_LN = load_data('Gamma-ray_LN.csv')
    mc_evaporation_LN_Carr17, f_pbh_evaporation_LN_Carr17 = load_data('Carr_17_LN_evap.csv')

    # calculate constraints for extended MF from evaporation
    f_pbh_evap = []
    f_pbh_evap_rootfinder = []
    
    for m_c in mc_evaporation:
        
        m_range = 10**np.linspace(np.log10(min(m_evaporation_mono)), np.log10(max(m_evaporation_mono)), 10000)
        #f_pbh_evap.append(1/np.trapz(integrand(m_range, m_c, f_pbh=1), m_range))
        f_pbh_evap.append(1/np.trapz(integrand(f_pbh=1, m=m_range, m_c=m_c), m_range))
        f_pbh_evap_rootfinder.append(findroot(constraint_function, 1e-5, 10, tolerance_1=1e-4, tolerance_2=1e-4, n_max=1000))


    plt.figure(figsize=(12,8))

    m1 = m_star
    for i, m2 in enumerate([7e16/1.989e33, 1e18/1.989e33, np.power(5e9, 1/(3+epsilon)) * m_star]):
        f_pbh_evap = []

        for m_c in mc_evaporation:
            
            m_range = 10**np.linspace(np.log10(m1), np.log10(m2), 10000)
            #f_pbh_evap.append(1/np.trapz(integrand(m_range, m_c, f_pbh=1), m_range))
            f_pbh_evap.append(1/np.trapz(integrand(f_pbh=1, m=m_range, m_c=m_c), m_range))
            f_pbh_evap_rootfinder.append(findroot(constraint_function, 1e-5, 10, tolerance_1=1e-4, tolerance_2=1e-4, n_max=1000))
        
        plt.plot(mc_evaporation, f_pbh_evap, label=r'$M_1 = M_*$, $M_2 = {:.0e}$g'.format(m2*1.989e33), linestyle = 'dotted', linewidth=6+i/2)


    #plt.plot(mc_evaporation, f_pbh_evap, label='$M_1 = {:.0e}$g, $M_2 = {:.0e}$g'.format(min(m_evaporation_mono)*1.989e33, (max(m_evaporation_mono)*1.989e33)))
    plt.plot(mc_evaporation_LN, f_pbh_evaporation_LN, color='k', alpha=0.25, linewidth=5, label='Extracted (Carr+ 21)')
    #plt.plot(mc_evaporation, f_pbh_evap_rootfinder, label='Calculated (root-finder)', linestyle='dotted')

    plt.xlabel('$M_\mathrm{c}~[M_\odot]$')
    plt.ylabel('$f_\mathrm{PBH}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.xlim(10**(-15.5), 1e-12)
    plt.ylim(1e-4, 1)
    plt.title('Log-normal MF ($\sigma = 2$)')
    plt.tight_layout()
    #plt.savefig('Figures/evap_constraints_updated_fixedM1.pdf')



    plt.figure(figsize=(12,8))

    m2 = 7e16 / 1.989e33
    for i, m1 in enumerate([m_star, 2*m_star, 10*m_star]):
        f_pbh_evap = []

        for m_c in mc_evaporation:
            
            m_range = 10**np.linspace(np.log10(m1), np.log10(m2), 10000)
            #f_pbh_evap.append(1/np.trapz(integrand(m_range, m_c, f_pbh=1), m_range))
            f_pbh_evap.append(1/np.trapz(integrand(f_pbh=1, m=m_range, m_c=m_c), m_range))
            f_pbh_evap_rootfinder.append(findroot(constraint_function, 1e-5, 10, tolerance_1=1e-4, tolerance_2=1e-4, n_max=1000))
        
        plt.plot(mc_evaporation, f_pbh_evap, label=r'$M_1 = {:.0f}M_*$, $M_2 = {:.0e}$g'.format(m1/m_star, m2*1.989e33), linestyle = 'dotted', linewidth=6+i/2)


    #plt.plot(mc_evaporation, f_pbh_evap, label='$M_1 = {:.0e}$g, $M_2 = {:.0e}$g'.format(min(m_evaporation_mono)*1.989e33, (max(m_evaporation_mono)*1.989e33)))
    plt.plot(mc_evaporation_LN, f_pbh_evaporation_LN, color='k', alpha=0.25, linewidth=5, label='Extracted (Carr+ 21)')
    #plt.plot(mc_evaporation, f_pbh_evap_rootfinder, label='Calculated (root-finder)', linestyle='dotted')

    plt.xlabel('$M_\mathrm{c}~[M_\odot]$')
    plt.ylabel('$f_\mathrm{PBH}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.xlim(10**(-15.5), 1e-12)
    plt.ylim(1e-4, 1)
    plt.title('Log-normal MF ($\sigma = 2$)')
    plt.tight_layout()
    #plt.savefig('Figures/evap_constraints_updated_fixedM2.pdf')
    

    
    plt.figure(figsize=(12,8))
    f_pbh_evap = extended_constraint_analytic(mc_evaporation)
    plt.plot(mc_evaporation, f_pbh_evap, label=r'$M_1 = 0$, $M_2 = \infty$', linestyle = 'dotted', linewidth=6)

    #plt.plot(mc_evaporation, f_pbh_evap, label='$M_1 = {:.0e}$g, $M_2 = {:.0e}$g'.format(min(m_evaporation_mono)*1.989e33, (max(m_evaporation_mono)*1.989e33)))
    plt.plot(mc_evaporation_LN, f_pbh_evaporation_LN, color='k', alpha=0.25, linewidth=5, label='Extracted (Carr+ 21)')
    #plt.plot(mc_evaporation, f_pbh_evap_rootfinder, label='Calculated (root-finder)', linestyle='dotted')

    plt.xlabel('$M_\mathrm{c}~[M_\odot]$')
    plt.ylabel('$f_\mathrm{PBH}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.xlim(10**(-15.5), 1e-12)
    plt.ylim(1e-4, 1)
    plt.title('Log-normal MF ($\sigma = 2$)')
    plt.tight_layout()
    #plt.savefig('Figures/evap_constraints_updated_0_inf.pdf')

    
m_range = 10**np.linspace(-30, -15, 10000)
print(np.trapz(log_normal_MF(f_pbh=1, m=m_range, m_c=1e14 / 1.989e33), m_range))
print(np.trapz(log_normal_MF(f_pbh=1, m=m_range, m_c=1e16 / 1.989e33), m_range))
print(np.trapz(log_normal_MF(f_pbh=1, m=m_range, m_c=1e18 / 1.989e33), m_range))

plt.figure()
for m_c in np.array([10**(-16), 10**(-15), 10**(-14)]):
    plt.plot(m_range, log_normal_MF(f_pbh=1, m=m_range, m_c=m_c), label=r'$M_c = 1e{:.1f} M_\odot$'.format(np.log10(m_c)) )
    
plt.xlabel('$M~[M_\odot]$')
plt.ylabel('$\psi(M, f_\mathrm{PBH} = 1)$')
plt.vlines(m_star, ymin=1e10, ymax=5e14, linestyle='dotted', color='k', label=r'$M_* = {:3.1e} M_\odot$'.format(m_star))
plt.legend(fontsize='small')
#plt.xscale('log')
#plt.yscale('log')
plt.tight_layout()
plt.savefig('./Figures/LN_MF_cutoff_solmass_lower_mc.pdf')


plt.figure()
for m_c in np.array([10**(-16), 10**(-15), 10**(-14)]):
    plt.plot(1.989e33*np.array(m_range), log_normal_MF(f_pbh=1, m=m_range, m_c=m_c), label=r'$M_c = {:3.1e}$ g'.format(m_c*1.989e33) )
    
plt.xlabel('$M$ [g]')
plt.ylabel('$\psi(M, f_\mathrm{PBH} = 1)$')
plt.vlines(m_star, ymin=1e10, ymax=5e14, linestyle='dotted', color='k', label=r'$M_* = {:3.1e}$ g'.format(m_star*1.989e33))
plt.legend(fontsize='small')
#plt.xscale('log')
#plt.yscale('log')
plt.tight_layout()
plt.savefig('./Figures/LN_MF_cutoff_g_lower_mc.pdf')


# Find ratio of the area under a LN MF with different M_c values
def cdf(m_c):
    return 1 + erf(np.log(m_star/m_c) / (sigma * np.sqrt(2)))

print(cdf(m_c = 1e-15) / cdf(m_c = 1e-16))
print(cdf(m_c = 1e-16) / cdf(m_c = 1e-17))
print(cdf(m_c = 1e-17) / cdf(m_c = 1e-18))