#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 11:42:40 2022

@author: ppxmg2
"""

import numpy as np
from reproduce_COMPTEL_constraints_v2 import read_blackhawk_spectra, load_data
import matplotlib as mpl
import matplotlib.pyplot as plt
import cProfile, pstats
#from astropy import units as u
#from astropy.units import cds

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


# Reproduce Lee & Chan (2021) Fig. 1, expressing intermediate quantities
# in units of [GeV, cm, s, K, microGauss]

n_steps = 1000 # number of integration steps

# Parameters relating to clusters
A262 = True
A2199 = False
A85 = False
NGC5044 = False

# Booleans relating to choice of parameter values
Chan15 = False
upper_error = False

# Unit conversion factors
keV_to_GeV = 1e-3
solMass_to_g = 1.98847e33
Mpc_to_cm = 3.0857e24
kpc_to_cm = 3.0857e21
GeV_to_erg = 0.00160218

c = 2.99792458e11  # speed of light, in cm / s   # Wrong! should be e10 (21/8)
k_B = 8.617333262e-8  # Botlzmann constant, in keV/ K


m_e = 5.11e-4 / c**2 # electron/positron mass, in GeV / c^2
epsilon = 0.5 # paper says this varies between 0.5-1
Lambda_0 = 1.4e-27 # in erg cm^3 s^{-1} K^{-1/2}

E_min = m_e * c**2  # minimum electron/positron energy calculated from BlackHawk, in GeV
E_max = 5  # maximum electron/positron energy calculated from BlackHawk, in Gnp.diffeV


#%%

if A262:
    T_c_keV = 1  # maximum core temperature, in keV
    T_c_K = 1 / k_B  # maximum core temperature, in K
    
    rho_s = 14.1 * 1e14 * solMass_to_g / Mpc_to_cm**3 # scale density, in g / cm^3
    r_s = 172 * kpc_to_cm # scale radius, in cm
    R = 2 * kpc_to_cm # max radius to integrate out to, in cm
    z = 0.0161 # redshift
    #beta = 0.433 - 0.017
    beta = 0.433
    r_c = 30 * kpc_to_cm
    n_0 = 0.94 * 1e-2  # central number density of hot gas particles (thermal electrons/positrons), in cm^{-3}
    #n_0 = (0.94+0.15) * 1e-2  # central number density of hot gas particles (thermal electrons/positrons), in cm^{-3}
    
    extension='A262'

    B_0 = 2.9 # maximum central magnetic field, in microgauss
    L_0 = 5.6e38 # maximum observed luminosity, in erg s^{-1}

if A2199:
    T_c_keV = 2 # maximum core temperature, in keV
    T_c_K = T_c_keV / k_B # maximum core temperature, in K
    rho_s = 9.56 * 1e14 * solMass_to_g / Mpc_to_cm**3 # scale density, in g / cm^3
    r_s = 334 * kpc_to_cm # scale radius, in cm
    R = 3 * kpc_to_cm # max radius to integrate out to, in cm
    z = 0.0302 # redshift
    beta = 0.665-0.021
    r_c = 102 * kpc_to_cm
    n_0 = (0.97+0.03) * 1e-2 # central number density of hot gas particles (thermal electrons/positrons), in cm^{-3}

    extension = 'A2199'

    B_0 = 4.9 # maximum central magnetic field, in microgauss
    L_0 = 2.3e39 # maximum observed luminosity, in erg s^{-1}
    
if A85:
    T_c_keV = 3 # maximum core temperature, in keV
    T_c_K = T_c_keV / k_B # maximum core temperature, in K
    rho_s = 8.34 * 1e14 * solMass_to_g / Mpc_to_cm**3 # scale density, in g / cm^3
    r_s = 444 * kpc_to_cm # scale radius, in cm
    R = 3 * kpc_to_cm # max radius to integrate out to, in cm
    z = 0.0556 # redshift
    beta = 0.532 - 0.004
    r_c = 60 * kpc_to_cm
    n_0 = (3.00+0.15) * 1e-2 # central number density of hot gas particles (thermal electrons/positrons), in cm^{-3}
    
    extension = 'A85'
        
    B_0 = 11.6 # maximum central magnetic field, in microgauss
    L_0 = 2.7e40 # maximum observed luminosity, in erg s^{-1}
 
if NGC5044:
    T_c_keV = 0.8 # maximum core temperature, in keV
    T_c_K = T_c_keV / k_B # maximum core temperature, in K
    rho_s = 14.7 * 1e14 * solMass_to_g / Mpc_to_cm**3 # scale density, in g / cm^3
    r_s = 127 * kpc_to_cm # scale radius, in cm
    R = 1 * kpc_to_cm # max radius to integrate out to, in cm
    z = 0.009 # redshift
    beta = 0.524-0.003
    r_c = 8 * kpc_to_cm
    n_0 = (4.02+0.03) * 1e-2 # central number density of hot gas particles (thermal electrons/positrons), in cm^{-3}

    extension = 'NGC5044'

    B_0 = 5.0 # maximum central magnetic field, in microgauss
    L_0 = 6.3e38 # maximum observed luminosity, in erg s^{-1}

# DM density (NFW), in g cm^{-3}
def rho_NFW(r):
    return rho_s * (r_s / r) * (1 + (r/r_s))**(-2)

# Lorentz factor
def gamma(E):
    return E / (m_e * c**2)

# Number density of hot gas particles (thermal electrons/positrons), in cm^{-3}
def number_density(r):
    return n_0 * (1 + (r/r_c)**2)**(-3*beta/2)

# Cluster magnetic field, in microgauss
def magnetic_field(r):
    return 11 * epsilon**(-1/2) * np.sqrt((number_density(r)) / 0.1) * (T_c_keV / (2))**(3/4)

def b_C(E, r):
    return 6.13 * (number_density(r)) * (1 + np.log(gamma(E) / (number_density(r)))/75)

def b_T(E, r):
    b_1 = 0.0254 * (E)**2 * (magnetic_field(r))**2
    b_2 = 0.25 * (E)**2 * (1+z)**4
    b_3 = 1.51 * (number_density(r))  * (0.36 + np.log(gamma(E) / (number_density(r)))) 
    return (b_1 + b_2 + b_3) + b_C(E, r)

def luminosity_integrand_2(r, E, m_pbh, E_prime, spectrum_integrand):      
    return r**2 * np.trapz(spectrum_integrand, E_prime) * rho_NFW(r) * b_C(E, r) / (m_pbh * b_T(E, r))
    #return r**2 * np.sum(spectrum_integrand[:-1] * np.diff(E_prime)) * rho_NFW(r) * b_C(E, r) / (m_pbh * b_T(E, r))


def luminosity_predicted_2(i, m_pbh): # predicted luminosity, in erg s^{-1}
    r_values = 10**np.linspace(np.log10(1e-10 * kpc_to_cm), np.log10(R), n_steps)
    
    if numbered_mass_range == True:
        file_path_data = "../Downloads/version_finale/results/LC21_{:.0f}/".format(i+1)
           
    else:
        exponent = np.floor(np.log10(m_pbh))
        coefficient = m_pbh / 10**exponent
        file_path_data = "../blackhawk_v2.0/results/A22_Fig3_" + "{:.1f}e{:.0f}g/".format(coefficient, exponent)

    # Load electron secondary spectrum
    energies_ep_secondary, secondary_ep_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_secondary_spectra.txt", col=2)
    energies_photons_secondary, secondary_photon_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_secondary_spectra.txt", col=1)

    energies_secondary = energies_ep_secondary
    secondary_spectrum = secondary_ep_spectrum
    
    if i == 15:
        print(file_path_data)
        print(secondary_spectrum)    
     
    integrand_over_r = []
    for E in energies_ref:
        
        E_prime = 10**np.linspace(np.log10(E), np.log10(E_max), n_steps)
        spectrum_integrand = 10**np.interp(np.log10(E_prime), np.log10(energies_secondary), np.log10(secondary_spectrum))
        
        luminosity_integrand_terms = [luminosity_integrand_2(r, E, m_pbh, E_prime, spectrum_integrand) for r in r_values]
        integrand_over_r.append(np.trapz(luminosity_integrand_terms, r_values))

    integral = np.trapz(integrand_over_r, energies_ref)

    
    return 4 * np.pi * integral * GeV_to_erg

def luminosity_observed(): # observed luminosity
    r_values = np.linspace(0, R, n_steps)
   
    integrand_terms = [number_density(r)**2 * r**2 for r in r_values]
    return 4 * np.pi * Lambda_0 * np.sqrt(T_c_K) * np.trapz(integrand_terms, r_values)


from scipy.special import hyp2f1
def luminosity_observed_analytic(): # observed luminosity, in erg s^{-1} (analytic solution, in terms of hypergeometric function)
    return (4/3) * np.pi * n_0**2 * Lambda_0 * np.sqrt(T_c_K) * R**3 * hyp2f1(3/2, 3*beta, 5/2, -(R/r_c)**2)

print('Magnetic field (microgauss):')
print(magnetic_field(r=0))
print(B_0)

luminosity_calculated = luminosity_observed()
print('Observed luminosity (calculated) = {:.2e} '.format(luminosity_calculated))
print("Observed luminosity (from LC '21) = {:.2e}".format(L_0))
print('Ratio (calculated to LC21) = {:.5f}'.format(luminosity_calculated / L_0))



#%%


numbered_mass_range = True

#m_pbh_values = np.array([0.1, 0.12, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.5, 2, 3, 4, 6, 8]) * 10**16
m_pbh_values = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.5, 3, 6, 8]) * 10**16
#m_pbh_values = np.array([3, 6, 8]) * 10**16
#m_pbh_values = np.array([0.1, 0.3, 0.7, 1.5, 3, 6, 8]) * 10**16
#m_pbh_values = np.array([1e15])

if numbered_mass_range == True:
    m_pbh_values = 10**np.linspace(np.log10(5e14), 17, 25)
    
f_pbh_values = []
m_pbh_plotting = []

energies_ref = 10**np.linspace(np.log10(E_min), np.log10(E_max), n_steps)


def main():

    for i, m_pbh in enumerate(m_pbh_values):
        #print('M_PBH = {:.2e} g'.format(m_pbh))
        if i % 1 == 0:
            # Evaluate photon spectrum at a set of pre-defined energies                
            luminosity_predicted = luminosity_predicted_2(i, m_pbh)
            f_pbh_values.append(L_0 / luminosity_predicted)
            
            m_pbh_plotting.append(m_pbh)


if __name__ == '__main__':
    
    file_path_extracted = './Extracted_files/'
    m_pbh_LC21_extracted, f_PBH_LC21_extracted = load_data("LC21_" + extension + "_NFW.csv")
    """
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler)
    # Print to file
    stats.sort_stats('cumtime').dump_stats('./cProfiler/LC21_reproduction.txt')
    """
    f_pbh_values = []
    main()
        
    extracted_interpolated = 10**np.interp(np.log10(m_pbh_plotting), np.log10(m_pbh_LC21_extracted), np.log10(f_PBH_LC21_extracted))
    ratio = extracted_interpolated / np.array(f_pbh_values)
    frac_diff = ratio - 1
    
    #%%
    
    # linear dependence of f_PBH approximation
    index = 3
    f_pbh_PL = f_PBH_LC21_extracted[0] * (m_pbh_LC21_extracted / m_pbh_LC21_extracted[0])**index
    
    plt.figure(figsize=(7, 6))
    #plt.plot(m_pbh_plotting, 0.5*np.array(f_pbh_values), label='Reproduction')
    plt.plot(m_pbh_LC21_extracted, f_PBH_LC21_extracted, label='Extracted')
    plt.plot(m_pbh_LC21_extracted, np.array(f_pbh_PL), label='Power-law approx $(n={:.0f})$'.format(index))

    plt.plot()
    plt.xlabel('$M_\mathrm{PBH}$ [g]')
    plt.ylabel('$f_\mathrm{PBH}$')
    plt.title(extension)
    plt.tight_layout()
    plt.legend()
    plt.ylim(1e-8, 1)
    plt.xlim(4e14, 1e17)
    plt.yscale('log')
    plt.xscale('log')
    #plt.yticks(major_ticks)
    #plt.yticks(minor_ticks)
    """
    plt.figure()
    plt.plot(m_pbh_plotting, ratio, 'x')
    plt.xlabel('$M_\mathrm{PBH}$ [g]')
    plt.ylabel('$f_\mathrm{PBH, extracted} / f_\mathrm{PBH, calculated}$')
    plt.xscale('log')
    #plt.yscale('log')
    plt.xlim(4e14, 6e16)   # upper limit is where f_PBH = 1 in Fig. 1 of Lee & Chan (2021)
    plt.title(extension)
    plt.tight_layout()
    
    plt.figure()
    plt.plot(m_pbh_plotting, frac_diff, 'x')
    plt.xlabel('$M_\mathrm{PBH}$ [g]')
    plt.ylabel('$(f_\mathrm{PBH, extracted} / f_\mathrm{PBH, calculated}) - 1$')
    plt.xscale('log')
    #plt.yscale('log')
    plt.xlim(4e14, 6e16)   # upper limit is where f_PBH = 1 in Fig. 1 of Lee & Chan (2021)
    plt.title(extension)
    plt.tight_layout()
    
    
    print('f_PBH =', f_pbh_values)
    print('ratio =', ratio)
    print('fractional difference =', frac_diff)
    """
    
#%% Plot spectra
#m_pbh_values = np.array([0.1, 0.3, 0.7, 1.5, 3, 6, 8]) * 10**16
#m_pbh_values = np.array([0.1, 0.5, 1, 5, 7, 10]) * 10**16

plt.figure()
m_pbh_plotting = []

for i, m_pbh in enumerate(m_pbh_values):
    #exponent = np.floor(np.log10(m_pbh))
    #coefficient = m_pbh / 10**exponent
    #file_path_data = "../blackhawk_v2.0/results/A22_Fig3_" + "{:.1f}e{:.0f}g/".format(coefficient, exponent)
    #file_path_data = "../blackhawk_v2.0/results/Laha16_Fig1_" + "{:.0f}e{:.0f}g/".format(coefficient, exponent)
    file_path_data = "../Downloads/version_finale/results/LC21_{:.0f}/".format(i+1)
    
    # Load electron secondary spectrum
    energies_secondary, secondary_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_secondary_spectra.txt", col=2)
    #energies_secondary, secondary_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_primary_spectra.txt", col=7)
   
    # Evaluate photon spectrum at a set of pre-defined energies
    energies_ref = 10**np.linspace(np.log10(E_min), np.log10(E_max), n_steps)
    spectrum_ref = 10**np.interp(np.log10(energies_ref), np.log10(energies_secondary), np.log10(secondary_spectrum))
    
    plt.plot(energies_secondary, secondary_spectrum, 'x', label='$M_\mathrm{PBH} '+'= ${:.2e} g'.format(m_pbh))
    
plt.legend()
plt.xscale('log')
plt.yscale('log')


#%%
E = 0.1

if numbered_mass_range == True:
    m_pbh_values = 10**np.linspace(14.5, 17, 25)
    file_path_data = "../Downloads/version_finale/results/LC21_{:.0f}/".format(i+1)

r_values = 10**np.linspace(1, np.log10(R), 100)

plt.figure()
for i, m_pbh in enumerate(m_pbh_values):
    
    if i % 1 == 0:
    # Load electron secondary spectrum
        
        energies_secondary, secondary_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_secondary_spectra.txt", col=2)

        # Evaluate photon spectrum at a set of pre-defined energies
        spectrum_ref = 10**np.interp(np.log10(energies_ref), np.log10(energies_secondary), np.log10(secondary_spectrum))
                
        integral_over_r = [luminosity_integrand_2(r, E, m_pbh, spectrum_ref) for r in r_values]
        
        plt.plot(r_values, integral_over_r,  'x', label='{:.2e}'.format(m_pbh))
        
plt.xlabel('$r$ [kpc]')
plt.ylabel('Luminosity integrand, E={:.1f}GeV'.format(E))
plt.tight_layout()
plt.xscale('log')
plt.yscale('log')
plt.legend(title='$M_\mathrm{PBH} [g]$')