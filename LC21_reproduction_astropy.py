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
from astropy import units as u
from astropy.units import cds

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
Lambda_0 = 1.4e-27 * u.erg * u.cm**(3) * u.s**(-1) * u.K**(-0.5) # in erg cm^{-3} s^{-1} K^{-1/2}

n_steps = 1000 # number of integration steps
E_min = m_e * 1e9 * u.eV  # minimum electron/positron energy calculated from BlackHawk, in GeV
E_max = 5 * 1e9 * u.eV # maximum electron/positron energy calculated from BlackHawk, in GeV

# Parameters relating to clusters
A262 = True
A2199 = False
A85 = False
NGC5044 = False

# Booleans relating to choice of parameter values
Chan15 = False
upper_error = False

#%%

if A262:
    T_c = 1 * 1e3 * u.eV / cds.k # maximum core temperature, in keV
    rho_s = 14.1 * u.solMass / (1e3 * u.pc)**3 # scale density, in solar masses * kpc^{-3}
    r_s = 172 * 1e3 * u.pc # scale radius, in kpc
    R = 2 * 1e3 * u.pc # max radius to integrate out to
    z = 0.0161 # redshift
    beta = 0.433
    r_c = 30 * 1e3 * u.pc
    n_0 = 0.94 * 1e-2 * (u.cm)**(-3) # central number density of hot gas particles (thermal electrons/positrons), in cm^{-3}
    
    extension='A262'

    if Chan15:
        # values that differ in Chan 2015
        beta = 0.443
        r_c = 41
        n_0 = 0.81 * 1e-2
        
    if upper_error:
        n_0 = (0.94 + 0.15) * 1e-2

    B_0 = 2.9 * 1e-6 * u.G # maximum central magnetic field, in microgauss
    L_0 = 5.6e38 * u.erg * (u.s)**(-1) # maximum observed luminosity, in erg s^{-1}


# DM density (NFW), in solar masses kpc^{-3}
def rho_NFW(r):
    return rho_s * (r_s / r) * (1 + (r/r_s))**(-2)

# Lorentz factor
def gamma(E):
    #print( E / (m_e * 1e9 * u.eV))
    return E / (m_e * 1e9 * u.eV)

# Number density of hot gas particles (thermal electrons/positrons), in cm^{-3}
def number_density(r):
    return n_0 * (1 + (r/r_c)**2)**(-3*beta/2)

# Cluster magnetic field, in microgauss
def magnetic_field(r):
    return 11 * epsilon**(-1/2) * np.sqrt((number_density(r) / u.cm**(-3)) / 0.1) * (T_c / (2 * 1e3 * u.eV / cds.k))**(3/4) * 1e-6 * u.G

def b_C(E, r):
    return 6.13 * (number_density(r)/u.cm**(-3)) * (1 + np.log(gamma(E)/(number_density(r)/u.cm**(-3)))/75)

def b_T(E, r):
    b_1 = 0.0254 * (E/(1e9 * u.eV))**2 * (magnetic_field(r)/(1e-6 * u.G))**2
    b_2 = 0.25 * (E/(1e9 * u.eV))**2 * (1+z)**4
    b_3 = 1.51 * (number_density(r)/u.cm**(-3))  * (0.36 + np.log(gamma(E) / (number_density(r)/u.cm**(-3)))) 
    return (b_1 + b_2 + b_3) + b_C(E, r)

def luminosity_integrand_2(r, E):  
    E_prime = energies_ref[energies_ref > E]
    spectrum_integrand = spectrum_ref[energies_ref > E]
    #print('E_prime = ', E_prime[0])
   
    #print('spectrum = ', spectrum_integrand[0])
    #print('integral of spectrum = ', np.sum(spectrum_integrand[:-1] * np.diff(E_prime)))
    
    return r**2 * np.sum(spectrum_integrand[:-1] * np.diff(E_prime)) * rho_NFW(r) * b_C(E, r) / (m_pbh * b_T(E, r))


def luminosity_predicted_2(): # predicted luminosity
    E_values = 10**np.linspace(np.log10(E_min / (1e9 * u.eV)), np.log10(E_max / (1e9 * u.eV)), n_steps) * 1e9 * u.eV
    r_values = 10**np.linspace(np.log10(1e-7), np.log10(R/u.pc), n_steps) * u.pc
    
    #print('E_values[0] : ', E_values[0])
    
    integrand_over_r = []
    for E in E_values:
        
        luminosity_integrand_terms = []
        
        for r in r_values:
            luminosity_integrand_terms.append(luminosity_integrand_2(r, E))
        
        #print(luminosity_integrand_terms[0])    # units: solMass / (g pc s)
        #print(r_values[0])   # units: pc
        
        luminosity_integrand_terms_val = []
        for l in luminosity_integrand_terms:
            luminosity_integrand_terms_val.append(l.value)
        luminosity_integrand_terms_unit = luminosity_integrand_terms[0].unit
        """
        print('sum of terms = ', np.sum(luminosity_integrand_terms_val[:-1]))
        print('Riemann sum', np.sum(luminosity_integrand_terms_val[:-1] * np.diff(r_values)))
        print('Riemann sum (correct units)', np.sum(luminosity_integrand_terms_val[:-1] * np.diff(r_values)) * luminosity_integrand_terms_unit )
        """
        integrand_over_r.append(np.sum(luminosity_integrand_terms_val[:-1] * np.diff(r_values)) * luminosity_integrand_terms_unit)    # units: solMass / (g s)
    
        integrand_over_r_terms_val = []
        for a in integrand_over_r:
            integrand_over_r_terms_val.append(a.value)
        integrand_over_r_terms_unit = integrand_over_r[0].unit

    integral = np.sum(integrand_over_r_terms_val[:-1] * np.diff(E_values)) * integrand_over_r_terms_unit   # units: solMass * GeV / (g s)
    #print('integral = ', (4 * np.pi * integral))
    
    return 4 * np.pi * integral

def luminosity_observed(): # observed luminosity, in erg s^{-1}
    r_values = np.linspace(0 * u.pc, R, n_steps)
    integrand_values = []
    
    for i in range(n_steps):
        integrand_values.append(number_density(r_values[i])**2 * r_values[i]**2)
    
    integrand_values_dimensionless = integrand_values / (u.cm**6 / u.pc**2)
    #print(integrand_values_dimensionless)
    r_values_dimensionless = r_values / u.pc
    

    return 4 * np.pi * Lambda_0 * np.sqrt(T_c) * np.sum(integrand_values_dimensionless[:-1] * np.diff(r_values_dimensionless)) * (u.cm**6 / u.pc**2) * u.pc

print('Magnetic field (microgauss):')
print(magnetic_field(r=0*u.pc))
print(B_0)

print('Observed luminosity [erg s^{-1}]')
print(luminosity_observed().to(u.erg / u.s))
print(L_0)


#%%
#m_pbh_values = np.array([0.1, 0.12, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.5, 2, 3, 4, 6, 8]) * 10**16
#m_pbh_values = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.5, 3, 6, 8]) * 10**16
#m_pbh_values = np.array([1e15])
m_pbh_values = 10**np.linspace(14.5, 17, 25)
f_pbh_values = []

def main():
    global m_pbh
    for i, m_pbh_val in enumerate(m_pbh_values):
        m_pbh = m_pbh_val * u.g
        #exponent = np.floor(np.log10(m_pbh))
        #coefficient = m_pbh / 10**exponent
        #file_path_data = "../blackhawk_v2.0/results/A22_Fig3_" + "{:.1f}e{:.0f}g/".format(coefficient, exponent)
        #file_path_data = "../blackhawk_v2.0/results/Laha16_Fig1_" + "{:.0f}e{:.0f}g/".format(coefficient, exponent)
        file_path_data = "../Downloads/version_finale/results/LC21_{:.0f}/".format(i+1)
        
        # Load electron secondary spectrum
        global energies_secondary
        global secondary_spectrum
        energies_secondary, secondary_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_secondary_spectra.txt", col=2)
        #energies_secondary, secondary_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_primary_spectra.txt", col=7)
       
        # Evaluate photon spectrum at a set of pre-defined energies
        global energies_ref
        global spectrum_ref
        energies_ref = 10**np.linspace(np.log10(E_min / (1e9 * u.eV)), np.log10(E_max / (1e9 * u.eV)), n_steps) * 1e9 * u.eV
        spectrum_ref = np.interp(energies_ref / (1e9 * u.eV), energies_secondary, secondary_spectrum) * (1e9 * u.eV * u.s)**(-1)
        
        f_pbh_values.append(L_0 / np.array(luminosity_predicted_2()))


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
    plt.figure()
    plt.plot(m_pbh_values, f_pbh_values)
    plt.xlabel('$M_\mathrm{PBH}$ [g]')
    plt.ylabel('$f_\mathrm{PBH}$')
    plt.title(extension)
    plt.tight_layout()
    plt.yscale('log')
    plt.xscale('log')
    
    extracted_interpolated = 10**np.interp(np.log10(m_pbh_values), np.log10(m_pbh_LC21_extracted), np.log10(f_PBH_LC21_extracted))
    ratio = (extracted_interpolated / (np.array(f_pbh_values)))**(-1)
    frac_diff = abs((extracted_interpolated - (f_pbh_values)) / ( f_pbh_values))
    plt.figure()
    plt.plot(m_pbh_values, ratio, 'x')
    plt.xlabel('$M_\mathrm{PBH}$ [g]')
    plt.ylabel('$f_\mathrm{PBH, extracted} / f_\mathrm{PBH, calculated}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(extension)
    plt.tight_layout()

    plt.figure()
    ratio_2 = (3e61 * np.array(extracted_interpolated) / (np.array(f_pbh_values)))**(-1)
    plt.plot(m_pbh_values, ratio_2, 'x')
    plt.xlabel('$M_\mathrm{PBH}$ [g]')
    plt.ylabel(r'$\frac{f_\mathrm{PBH, reproduced}}{3\times 10^{61} f_\mathrm{PBH, extracted}}$')
    plt.hlines(y=1, xmin=min(m_pbh_values), xmax=max(m_pbh_values), color='k', alpha=0.3)
    plt.xscale('log')
    #plt.yscale('log')
    plt.title(extension)
    plt.tight_layout()

    
#%% Plot spectrum
for i, m_pbh_val in enumerate(m_pbh_values):
    exponent = np.floor(np.log10(m_pbh_val))
    coefficient = m_pbh_val / 10**exponent

    if i % 4 == 0:
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
        plt.xlim(m_e, 5)
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
radii = 10**np.linspace(np.log10(1e-10), np.log10(R), n_steps)

energies_ref = 10**np.linspace(np.log10(E_min), np.log10(E_max), n_steps)
spectrum_ref = 10**np.interp(np.log10(energies_ref), np.log10(energies_secondary), np.log10(secondary_spectrum))

lum_int_over_E = []
for r in radii:
    #lum_int_over_E.append(np.trapz(luminosity_integrand_2(r, energies), energies))
    
    luminosity_integrand_terms = []
    for E in energies:
        luminosity_integrand_terms.append(luminosity_integrand_2(r, E))
        
    #print(luminosity_integrand_2(r, energies))
    lum_int_over_E.append(np.sum(luminosity_integrand_terms[:-1] * np.diff(energies)))
    
    
lum_int_over_r = []
for E in energies:
    #lum_int_over_r.append(np.trapz(luminosity_integrand_2(radii, E), radii))
    lum_int_over_r.append(np.sum(luminosity_integrand_2(radii, E)[:-1] * np.diff(radii)))
    
lum_int = 4 * np.pi * np.trapz(lum_int_over_r, energies)

print(lum_int * g_to_solar_mass * erg_to_GeV)
print(luminosity_predicted_2())
                   
plt.figure(figsize=(9, 5))
plt.plot(radii, lum_int_over_E)
plt.xlabel('$r$ [kpc]')
plt.ylabel('Luminosity integrand \n (integrated over $E$)')
plt.tight_layout()

plt.figure(figsize=(9, 5))
plt.plot(energies, lum_int_over_r)
plt.xlabel('$E$ [GeV]')
plt.ylabel('Luminosity integrand \n (integrated over $r$)')
plt.tight_layout()
plt.xscale('log')
plt.yscale('log')

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