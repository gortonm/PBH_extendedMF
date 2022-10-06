#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 11:42:40 2022

@author: ppxmg2
"""

import numpy as np
from reproduce_COMPTEL_constraints_v2 import read_blackhawk_spectra, load_data
from scipy.integrate import dblquad
import matplotlib as mpl
import matplotlib.pyplot as plt
import cProfile, pstats


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


m_e = 5.11e-4 # electron/positron mass, in GeV / c^2
epsilon = 0.5 # paper says this varies between 0.5-1
Lambda_0 = 1.4e-27 # in erg cm^{-3} s^{-1} K^{-1/2}

n_steps = 1000 # number of integration steps
E_min = m_e # minimum electron/positron energy calculated from BlackHawk, in GeV
E_max = 5 # maximum electron/positron energy calculated from BlackHawk, in GeV

# Parameters relating to clusters
A262 = True
A2199 = False
A85 = False
NGC5044 = False

# Booleans relating to choice of parameter values
Chan15 = False
upper_error = False

# unit conversions
erg_to_GeV = 624.15    # erg to GeV
g_to_solar_mass = 1 / 1.989e33    # g to solar masses
kpc_to_cm = 3.09e21    # kpc to cm
keV_to_K = 11604525    # keV to Kelvin

#%%

if A262:
    T_c = 1 # maximum core temperature, in keV
    rho_s = 14.1 * 1e14 * 1e-9 # scale density, in solar masses * kpc^{-3}
    r_s = 172 # scale radius, in kpc
    R = 2 # max radius to integrate out to
    z = 0.0161 # redshift
    beta = 0.433
    r_c = 30
    n_0 = 0.94 * 1e-2 # central number density of hot gas particles (thermal electrons/positrons), in cm^{-3}
    
    extension='A262'

    if Chan15:
        # values that differ in Chan 2015
        beta = 0.443
        r_c = 41
        n_0 = 0.81 * 1e-2
        
    if upper_error:
        n_0 = (0.94 + 0.15) * 1e-2

    B_0 = 2.9 # maximum central magnetic field, in microgauss
    L_0 = 5.6e38 # maximum observed luminosity, in erg s^{-1}
    
if A2199:
    T_c = 2 # maximum core temperature, in keV
    rho_s = 9.56 * 1e14 * 1e-9 # scale density, in solar masses * Mpc^{-3}
    r_s = 334 # scale radius, in kpc
    R = 3 # max radius to integrate out to, in kpc
    z = 0.0302 # redshift
    beta = 0.665
    r_c = 102
    n_0 = 0.97 * 1e-2 # central number density of hot gas particles (thermal electrons/positrons), in cm^{-3}

    extension = 'A2199'

    if Chan15:
        # values that differ in Chan 2015
        r_c = 139
        n_0 = 0.83 * 1e-2
        
    if upper_error:
        n_0 = (0.97 + 0.03) * 1e-2

    B_0 = 4.9 # maximum central magnetic field, in microgauss
    L_0 = 2.3e39 # maximum observed luminosity, in erg s^{-1}
    
if A85:
    T_c = 3 # maximum core temperature, in keV
    rho_s = 8.34 * 1e14 * 1e-9 # scale density, in solar masses * Mpc^{-3}
    r_s = 444 # scale radius, in kpc
    R = 3 # max radius to integrate out to, in kpc
    z = 0.0556 # redshift
    beta = 0.532
    r_c = 60
    n_0 = 3.00 * 1e-2 # central number density of hot gas particles (thermal electrons/positrons), in cm^{-3}
    
    extension = 'A85'
    
    if Chan15:
        # values that differ in Chan 2015
        r_c = 82
        n_0 = 2.57 * 1e-2
        
    if upper_error:
        n_0 = (3.00 + 0.12) * 1e-2
    
    B_0 = 11.6 # maximum central magnetic field, in microgauss
    L_0 = 2.7e40 # maximum observed luminosity, in erg s^{-1}
 
if NGC5044:
    T_c = 0.8 # maximum core temperature, in keV
    rho_s = 14.7 * 1e14 * 1e-9 # scale density, in solar masses * kpc^{-3}
    r_s = 127 # scale radius, in kpc
    R = 1 # max radius to integrate out to, in kpc
    z = 0.009 # redshift
    beta = 0.524
    r_c = 8
    n_0 = 4.02 * 1e-2 # central number density of hot gas particles (thermal electrons/positrons), in cm^{-3}

    extension = 'NGC5044'

    if Chan15:
        # values that differ in Chan 2015
        r_c = 11
        n_0 = 3.45 * 1e-2
        
    if upper_error:
        n_0 = (4.02 + 0.03) * 1e-2

    B_0 = 5.0 # maximum central magnetic field, in microgauss
    L_0 = 6.3e38 # maximum observed luminosity, in erg s^{-1}



# DM density (NFW), in solar masses kpc^{-3}
def rho_NFW(r):
    return rho_s * (r_s / r) * (1 + (r/r_s))**(-2)

# Lorentz factor
def gamma(E):
    return E / m_e

# Number density of hot gas particles (thermal electrons/positrons), in cm^{-3}
def number_density(r):
    return n_0 * (1 + (r/r_c)**2)**(-3*beta/2)

# Cluster magnetic field, in microgauss
def magnetic_field(r):
    return 11 * epsilon**(-1/2) * np.sqrt(number_density(r) / 0.1) * (T_c / 2)**(3/4)

def b_C(E, r):
    return 6.13 * number_density(r) * (1 + np.log(gamma(E)/number_density(r))/75)

def b_T(E, r):
    b_1 = 0.0254 * E**2 * magnetic_field(r)**2
    b_2 = 0.25 * E**2 * (1+z)**4
    b_3 = 1.51 * number_density(r) * (0.36 + np.log(gamma(E) / number_density(r)))
    return (b_1 + b_2 + b_3) + b_C(E, r)

def luminosity_integrand(E, r):  
    E_prime = energies_ref[energies_ref > E]
    spectrum_integrand = spectrum_ref[energies_ref > E]
    #print(r**2 * np.trapz(spectrum_integrand, E_prime) * rho_NFW(r) * b_C(E, r))
    # 30/9: difference between Riemann sum and np.trapz() is small, on order of 1 part in 10^3
    return r**2 * np.sum(spectrum_integrand[:-1] * np.diff(E_prime)) * rho_NFW(r) * b_C(E, r) / (m_pbh * b_T(E, r))
    #return r**2 * np.trapz(spectrum_integrand, E_prime) * rho_NFW(r) * b_C(E, r) / (m_pbh * b_T(E, r))

def luminosity_predicted(): # predicted luminosity, in erg s^{-1}
    print(4 * np.pi * np.array(dblquad(luminosity_integrand, 0, R, m_e, E_max)) * g_to_solar_mass * erg_to_GeV)
    return 4 * np.pi * np.array(dblquad(luminosity_integrand, 0, R, m_e, E_max)) * g_to_solar_mass * erg_to_GeV


def luminosity_integrand_2(r, E):  
    E_prime = energies_ref[energies_ref > E]
    spectrum_integrand = spectrum_ref[energies_ref > E]
    #print('luminosity integrand', r**2 * np.trapz(spectrum_integrand, E_prime) * rho_NFW(r) * b_C(E, r) / (m_pbh * b_T(E, r)))    # typically quite large
    # 30/9: difference between Riemann sum and np.trapz() is small, on order of 1 part in 10^3
    return r**2 * np.sum(spectrum_integrand[:-1] * np.diff(E_prime)) * rho_NFW(r) * b_C(E, r) / (m_pbh * b_T(E, r))
    #return r**2 * np.trapz(spectrum_integrand, E_prime) * rho_NFW(r) * b_C(E, r) / (m_pbh * b_T(E, r))


def luminosity_predicted_2(): # predicted luminosity, in erg s^{-1}
    E_values = 10**np.linspace(np.log10(m_e), np.log10(E_max), n_steps)
    r_values = 10**np.linspace(np.log10(1e-7), np.log10(R), n_steps)
    
    integrand_over_r = []
    for E in E_values:
        #integrand_over_r.append(np.trapz(luminosity_integrand_2(r_values, E), r_values))
        integrand_over_r.append(np.sum(luminosity_integrand_2(r_values, E)[:-1] * np.diff(r_values)))
        
    #integral = np.trapz(integrand_over_r, E_values)
    integral = np.sum(integrand_over_r[:-1] * np.diff(E_values))
    #print(4 * np.pi * integral * g_to_solar_mass * erg_to_GeV)
    
    """
    integrand_over_E = []
    for r in r_values:
        integrand_over_E.append(np.trapz(luminosity_integrand_2(r, E_values)))
    integral = np.trapz(integrand_over_E, r_values)
    print(4 * np.pi * integral * g_to_solar_mass * erg_to_GeV)
    """
    return 4 * np.pi * integral * g_to_solar_mass * erg_to_GeV


def luminosity_observed(): # observed luminosity, in erg s^{-1}
    r_values = np.linspace(0, R, n_steps)
    integrand_values = []
    
    for i in range(n_steps):
        integrand_values.append(number_density(r_values[i])**2 * r_values[i]**2)
    
    return 4 * np.pi * Lambda_0 * np.sqrt(T_c * keV_to_K) * np.trapz(integrand_values, r_values) * kpc_to_cm**3

from scipy.special import hyp2f1
def luminosity_observed_analytic(): # observed luminosity, in erg s^{-1} (analytic solution, in terms of hypergeometric function)
    return (4/3) * np.pi * n_0**2 * Lambda_0 * np.sqrt(T_c * keV_to_K) * (R/r_c)**3 * hyp2f1(3/2, 3*beta, 5/2, -(R/r_c)**2) * r_c**3 * kpc_to_cm**3

print('Magnetic field (microgauss):')
print(magnetic_field(r=0))
print(B_0)

print('Observed luminosity [erg s^{-1}]')
print(luminosity_observed())
print(luminosity_observed_analytic())
print(L_0)
print('Ratio of calculated to given luminosities')
print(luminosity_observed_analytic() / L_0)


#%%
#m_pbh_values = np.array([0.1, 0.12, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.5, 2, 3, 4, 6, 8]) * 10**16
#m_pbh_values = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.5, 3, 6, 8]) * 10**16
#m_pbh_values = np.array([1e15])
m_pbh_values = 10**np.linspace(14.5, 17, 25)
f_pbh_values = []

def main():
    global m_pbh
    for i, m_pbh_val in enumerate(m_pbh_values):
        m_pbh = m_pbh_val
        #exponent = np.floor(np.log10(m_pbh))
        #coefficient = m_pbh / 10**exponent
        #file_path_data = "../blackhawk_v2.0/results/A22_Fig3_" + "{:.1f}e{:.0f}g/".format(coefficient, exponent)
        #file_path_data = "../blackhawk_v2.0/results/Laha16_Fig1_" + "{:.0f}e{:.0f}g/".format(coefficient, exponent)
        file_path_data = "../Downloads/version_finale/results/LC21_{:.0f}/".format(i+1)
        
        # Load electron secondary spectrum
        global energies_secondary
        global secondary_spectrum
        #energies_secondary, secondary_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_secondary_spectra.txt", col=2)
        energies_secondary, secondary_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_primary_spectra.txt", col=7)
       
        # Evaluate photon spectrum at a set of pre-defined energies
        global energies_ref
        global spectrum_ref
        energies_ref = 10**np.linspace(np.log10(E_min), np.log10(E_max), n_steps)
        spectrum_ref = np.interp(energies_ref, energies_secondary, secondary_spectrum)
        
        f_pbh_values.append(L_0 / luminosity_predicted_2())


print(f_pbh_values)

if __name__ == '__main__':
    
    file_path_extracted = './Extracted_files/'
    m_pbh_LC21_extracted, f_PBH_LC21_extracted = load_data("LC21_" + extension + "_NFW.csv")

    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler)
    # Print to file
    stats.sort_stats('cumtime').dump_stats('./cProfiler/LC21_reproduction.txt')
    
    f_pbh_values = []
    main()
    plt.plot(m_pbh_values, f_pbh_values)
    plt.xlabel('$M_\mathrm{PBH}$ [g]')
    plt.ylabel('$f_\mathrm{PBH}$')
    plt.tight_layout()
    plt.yscale('log')
    plt.xscale('log')
    
    extracted_interpolated = 10**np.interp(np.log10(m_pbh_values), np.log10(m_pbh_LC21_extracted), np.log10(f_PBH_LC21_extracted))
    ratio = extracted_interpolated / f_pbh_values
    frac_diff = (extracted_interpolated - f_pbh_values) / f_pbh_values
    plt.figure()
    plt.plot(m_pbh_values, ratio, 'x')
    plt.xlabel('$M_\mathrm{PBH}$ [g]')
    plt.ylabel('$f_\mathrm{PBH, extracted} / f_\mathrm{PBH, calculated}$')
    plt.title(extension)
    plt.tight_layout()

    
#%% Plot spectrum
for i, m_pbh_val in enumerate(m_pbh_values):
    exponent = np.floor(np.log10(m_pbh_val))
    coefficient = m_pbh_val / 10**exponent

    if i % 5 == 0:
        file_path_data = "../Downloads/version_finale/results/LC21_{:.0f}/".format(i+1)
        energies_primary, primary_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_primary_spectra.txt", col=7)
        energies_secondary, secondary_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_secondary_spectra.txt", col=2)
        
        plt.figure()
        plt.plot(energies_primary, primary_spectrum, 'x')
        plt.plot(energies_secondary, secondary_spectrum, 'x')
        plt.xlabel('$E$ [GeV]')
        plt.ylabel('$\mathrm{d}^2 N_e^{\pm} / (\mathrm{d}t\mathrm{d}E)$ [s$^{-1}$ GeV$^{-1}$]')
        plt.xscale('log')
        plt.yscale('log')
        plt.title('$M_\mathrm{PBH}$ ' + '= {:.0f}e{:.0f}g'.format(coefficient, exponent))
        plt.tight_layout()
    
#%% Investigate integrand for luminosity
"""
# plot integrand at fixed radius
energies = 10**np.linspace(np.log10(E_min), np.log10(E_max), n_steps)
energies_ref = 10**np.linspace(np.log10(E_min), np.log10(E_max), n_steps)
spectrum_ref = np.interp(energies_ref, energies_secondary, secondary_spectrum)


r = 1e-6
integrand_fixed_r = 4*np.pi*np.array(luminosity_integrand(energies, r))
print(integrand_fixed_r)
plt.plot(energies, 4*np.pi*np.array(luminosity_integrand(energies, r)))
plt.ylabel('Luminosity integrand [$\mathrm{kpc}^{-1} \cdot \mathrm{s}^{-1}$]')
plt.xlabel('$E$ [GeV]')
plt.xscale('log')
plt.tight_layout()
"""
# integrate over E, from E_min to E_max 
energies = 10**np.linspace(np.log10(E_min), np.log10(E_max), n_steps)
radii = 10**np.linspace(np.log10(1e-6), np.log10(R), n_steps)
    
energies_ref = 10**np.linspace(np.log10(E_min), np.log10(E_max), n_steps)
spectrum_ref = np.interp(energies_ref, energies_secondary, secondary_spectrum)

lum_int_over_E = []
for r in radii:
    lum_int_over_E.append(np.trapz(luminosity_integrand_2(energies, r), energies))
    
lum_int_over_r = []
for E in energies:
    lum_int_over_r.append(np.trapz(luminosity_integrand_2(E, radii), radii))
    
lum_int = np.trapz(lum_int_over_r, energies)


print(lum_int * g_to_solar_mass * erg_to_GeV)
print(luminosity_predicted_2())
                   
plt.figure(figsize=(9, 5))
plt.plot(radii, lum_int_over_E)
plt.xlabel('$r$ [kpc]')
plt.ylabel('Luminosity integrand \n (integrated over $E$)')
plt.tight_layout()
plt.xscale('log')
plt.yscale('log')

plt.figure(figsize=(9, 5))
plt.plot(energies, lum_int_over_r)
plt.xlabel('$E$ [GeV]')
plt.ylabel('Luminosity integrand \n (integrated over $r$)')
plt.tight_layout()

[energies_mg, radii_mg] = np.meshgrid(energies, radii)

luminosity_grid = np.zeros(shape=(n_steps, n_steps))
for i in range(len(energies)):
    for j in range(len(radii)):
        luminosity_grid[i][j] = luminosity_integrand_2(energies[i], radii[j])

fig = plt.figure()
ax = fig.gca(projection='3d')

# make 3D plot of integrand
surf = ax.plot_surface(energies_mg, radii_mg, 4*np.pi*luminosity_grid)
ax.set_xlabel('$E$ [GeV]', fontsize=14)
ax.set_ylabel('$r$ [kpc]', fontsize=14)
ax.set_zlabel('Luminosity integrand [$\mathrm{kpc}^{-1} \cdot \mathrm{s}^{-1}$]', fontsize=14)
plt.title('Luminosity integrand', fontsize=14)

# make heat map
heatmap = plt.figure()
ax1 = heatmap.gca()
plt.pcolormesh(energies_mg, radii_mg, np.log10(1+4*np.pi*(luminosity_grid)), cmap='jet')
plt.xlabel('$E$ [GeV]')
plt.ylabel('$r$ [kpc]')
plt.title(r'$\log_{10}$(1 + Luminosity integrand' + ' [$\mathrm{kpc}^{-1} \cdot \mathrm{s}^{-1}$])', fontsize=16)
plt.colorbar()
plt.tight_layout()