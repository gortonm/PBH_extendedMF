#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 15:30:09 2022

@author: ppxmg2
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from reproduce_extended_MF import triple_integral
from scipy.integrate import tplquad

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


"""Test for tplquad"""
def func(z, y, x):
    return z*x**2 + 4*y*z**3

x_min, x_max = 11, 14
y_min, y_max = 7, 10
z_min, z_max = 1, 2

tplquad(func, x_min, x_max, y_min, y_max, z_min, z_max)   # returns 3267.0, as expected

# Script to reproduce constraints on PBHs from (extra)galactic gamma rays, 
# using results from BlackHawk 

# Unit conversion factors
s_to_yr = 1 / (365.25 * 86400)   # convert 1 s to yr
cm_to_kpc = 3.2408e-22    # convert 1 cm to kpc
g_to_GeV = 5.61e23    # convert 1 gram to GeV / c^2

# Parameters
r_0 = 8.12    # galactocentric radius of Sun, in kpc (Table I caption: Coogan, Morrison & Profumo '20 2010.04797)
#E_min = 5.11e-4    # minimum positron energy to consider, in GeV
E_min = 1e-4
E_max = 6e-3
#E_max = 3e-3     # maximum positron energy to consider, in GeV

# Parameters (NFW density profile)
rho_0 = 0.376   # local DM density [GeV / c^2 cm^{-3}] (Table 3 de Salas+ '19 1906.06133)
a = 11   # scale radius [kpc] (Table 3 de Salas+ '19 1906.06133)

def rho_NFW(r):
    if abs((r/a) * (1 + (r/a))**2) < 1e-9:
        print('r = ', r)
    return rho_0 * ((r/a) * (1 + (r/a))**2)**(-1)

def r(los, b, l):
    if abs(los**2 + r_0**2 - 2*r_0*los*np.cos(b)*np.cos(l)) < 1e-15:
        print('r = ', los**2 + r_0**2 - 2*r_0*los*np.cos(b)*np.cos(l))
        print('los = ', los)
        print('b = ', b)
        print('l = ', l)
    return np.sqrt(los**2 + r_0**2 - 2*r_0*los*np.cos(b)*np.cos(l))

def j_integrand(los, b, l):
    return rho_NFW(r(los, b, l)) * np.cos(b)

def j(delta_b, delta_l):
    b_min, b_max = -delta_b, delta_b
    l_min, l_max = -delta_l, delta_l
    #delta_omega = np.pi * (l_max - l_min) * (np.sin(b_max) - np.sin(b_min))
    delta_omega = np.pi * (l_max - l_min) * (b_max - b_min)
    print(b_max * (180/np.pi))
    print('delta_omega = ', delta_omega)
    #return triple_integral(j_integrand, 0, r_0, b_min, b_max, l_min, l_max, n_steps=n_steps) / (delta_b * delta_l)
    return np.array(tplquad(j_integrand, l_min, l_max, b_min, b_max, 0, 0.99999*r_0)) / delta_omega

# unit conversion factor from units of integral output to those used in 2010.04797 [MeV cm^{-2} sr^{-1}]
unit_conversion_Coogan = 3.0857e24

print("NFW")
print("Delta Omega = 5 deg^2")
delta_omega = 2.39e-2    # 5 square degree observing region around the galactic centre
# ranges of b and l for a 5 deg^2 observing region
delta_b, delta_l = 0.5*np.sqrt(delta_omega/np.pi), 0.5*np.sqrt(delta_omega/np.pi)
#delta_b, delta_l = 0.5*np.sqrt(delta_omega), 0.5*np.sqrt(delta_omega)
print(j(delta_b, delta_l) * unit_conversion_Coogan)


print("|b| < 20 deg, |l| < 60 deg")
# Asymmetric case, including the whole range observed by COMPTEL
delta_b = np.radians(20)
delta_l = np.radians(60)
print(j(delta_b, delta_l) * unit_conversion_Coogan)


# Plot integrand
l_values = np.linspace(-delta_b, delta_b, 100)
los_values = np.linspace(1e-3, r_0, 100)

l_grid, los_grid = np.meshgrid(l_values, los_values)
integrand_grid = j_integrand(los_grid, delta_b/2, l_grid)
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(l_grid, los_grid, integrand_grid)
ax.set_xlabel('$l$')
ax.set_ylabel('Line of sight distance $L$ [kpc]')
ax.set_zlabel('Integrand')

b = 0.1
los = 5
plt.plot(l_values, j_integrand(los, b, l_values))
plt.xlabel('$l$ [radians]')
plt.ylabel('$J_D$')

print("Einasto")
def rho_Einasto(r):
    return rho_0 * np.exp( - (2/alpha) * ( (r/r_s)**alpha - 1))

def j_integrand(los, b, l):
    return rho_Einasto(r(los, b, l))

delta_b, delta_l = np.sqrt(delta_omega/np.pi), np.sqrt(delta_omega/np.pi)
# Parameters (Einasto)
print("Einasto (varying r_s and alpha)")
rho_0 = 0.388   # local DM density [GeV / c^2 cm^{-3}] (maximum value from Table 3 de Salas+ '19 1906.06133)
for r_s in [6.5, 9.2, 14.5]:
    for alpha in [0.09, 0.18, 0.39]:
        print(j(delta_omega, r_s) * unit_conversion_Coogan)








"""Try to reproduce constraints from Auffinger '22 Fig. 2 from COMPTEL"""

def read_col(fname, first_row=0, col=1, convert=int, sep=None, skip_lines=1):
    """Read text files with columns separated by `sep`.

    fname - file name
    col - index of column to read
    convert - function to convert column entry with
    sep - column separator
    If sep is not specified or is None, any
    whitespace string is a separator and empty strings are
    removed from the result.
    """
    
    data = []
    
    with open(fname) as fobj:
        i=0
        for line in fobj:
            i += 1
            #print(line)
            
            if i >= first_row:
                #print(line.split(sep=sep)[col])
                data.append(line.split(sep=sep)[col])
    return data    
     
def read_blackhawk_spectra(fname, col=1):
    """Read spectra files for a particular particle species, calculated
    from BlackHawk.

    fname - file name
    col - index of column to read (i.e. which particle type to use)
        - photons: col = 1 (primary and secondary spectra)
        - 7 for electrons (primary spectra)
    """
    energies_data = read_col(fname, first_row=2, col=0, convert=float)
    spectrum_data = read_col(fname, first_row=2, col=col, convert=float)
    
    energies = []
    for i in range(2, len(energies_data)):
        energies.append(float(energies_data[i]))

    spectrum = []
    for i in range(2, len(spectrum_data)):
        spectrum.append(float(spectrum_data[i]))
        
    return np.array(energies), np.array(spectrum)


# returns a number as a string in standard form
def string_scientific(val):
    exponent = np.floor(np.log10(val))
    coefficient = val / 10**exponent
    return r'${:.0f} \times 10^{:d}$'.format(coefficient, int(exponent))


file_path_extracted = './Extracted_files/A22_COMPTEL/'

def load_data(filename, file_path_extracted = './Extracted_files/'):
    return np.genfromtxt(file_path_extracted+filename, delimiter=',', unpack=True)

lower_bin, upper_bin = load_data('lower_bin.csv', file_path_extracted = './Extracted_files/A22_COMPTEL/'), load_data('upper_bin.csv', file_path_extracted = './Extracted_files/A22_COMPTEL/')
mean_flux = load_data('means.csv', file_path_extracted = './Extracted_files/A22_COMPTEL/')

# energy bins
bins = []
for i in range(len(upper_bin)):
    bins.append(np.array([lower_bin[i], upper_bin[i]]))


cm_to_GeV = 5.06773e13


def J_gal(J, delta_omega):
    return J * r_0 * rho_0 / delta_omega

def calc_f_pbh(m_pbh, J_gal, spectrum, flux_error, flux_mean, E_low_X, E_up_X, CL=0):
    prefactor = 4 * np.pi * m_pbh / J_gal
   
    spectrum_cutoff_1 = spectrum[spectrum > E_low_X]
    spectrum_cutoff = spectrum_cutoff_1[spectrum_cutoff_1 < E_up_X]    
   
    energies_interp = 10**np.linspace(np.log10(E_low_X), np.log10(E_up_X), 1000)
    spectrum_interp = np.interp(energies_interp, spectrum_cutoff)
    
    integral = np.trapz(spectrum_interp, energies_interp)
    
    return cm_to_GeV * prefactor * (E_up_X - E_low_X) * (flux_mean + CL*flux_error) / integral

m_pbh_values = [4e15, 6e15, 8e15, 1e16, 2e16, 4e16, 6e16, 8e16, 1e17]
f_pbh_values = []

J_COMPTEL, delta_omega_COMPTEL = 6.82, np.pi * (60 * 20 * (np.pi/180)**2)
J_INTEGRAL, delta_omega_INTEGRAL = 3.65, np.pi * (30 * 15 * (np.pi/180)**2)

J_gal = J_gal(J_COMPTEL, delta_omega_COMPTEL)

for m_pbh in m_pbh_values:
    
    exponent = np.floor(np.log10(m_pbh))
    coefficient = m_pbh / 10**exponent
        
    file_path_spectra = "../blackhawk_v2.0/results/A22_Fig3_" + "{:.0f}e{:.0f}g/".format(coefficient, exponent)
    # Load photon primary spectrum
    energies, spectrum = read_blackhawk_spectra(file_path_spectra + "instantaneous_primary_spectra.txt", col=1)

    f_pbh = 1.
    
    for i in range(0, 2):
        E_low_X, E_up_X = lower_bin[0][i], upper_bin[0][i]
        E_mean_X = mean_flux[1][i]

        if calc_f_pbh(m_pbh, J_gal, spectrum, flux_error=0, flux_mean=E_mean_X, E_low_X=E_low_X, E_up_X=E_up_X) < f_pbh:
            f_pbh = calc_f_pbh(m_pbh, J_gal, spectrum, flux_error=0, flux_mean=E_mean_X, E_low_X=E_low_X, E_up_X=E_up_X)

    f_pbh_values.append(f_pbh)
    
plt.figure()
plt.plot(m_pbh_values, f_pbh_values, 'x')
plt.xlim(min(m_pbh_values), max(m_pbh_values))
plt.ylim(1e-6, 1)
plt.xlabel('$M_\mathrm{PBH}$ [g]')
plt.ylabel('$f_\mathrm{PBH}')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.tight_layout()




"""
# distance from galactic centre to include positrons from, in kpc
R = 1.5
m_pbh_L19_Iso_1500pc, f_pbh_L19_Iso_1500pc = load_data('Laha19_Iso_1.5kpc.csv')

f_pbh_P_values = []
f_pbh_PS_values = []
m_pbh_values = []

# PBH mass (in grams)
for m_pbh in np.linspace(1, 10, 10) * 10**16:
    
    # PBH mass (in grams)
    m_pbh_values.append(m_pbh) 
    
    exponent = np.floor(np.log10(m_pbh))
    coefficient = m_pbh / 10**exponent

    #file_path_data = "../blackhawk_v2.0/results/Laha16_Fig1_" + "{:.0f}e{:.0f}g/".format(coefficient, exponent)
    #file_path_data = "../BlackHawk_v2.1/results/Laha16_Fig1_" + "{:.0f}e{:.0f}g/".format(coefficient, exponent)
    file_path_data = "../Downloads/version_finale/results/Laha16_Fig1_" + "{:.0f}e{:.0f}g/".format(coefficient, exponent)   # v2.1
    #file_path_data = "../blackhawk_v1.0/results/Laha19_Fig1_" + "{:.0f}e{:.0f}g/".format(coefficient, exponent)
    
    # Load electron primary spectrum
    energies_primary, primary_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_primary_spectra.txt", col=7)
    
    # Load electron secondary spectrum
    energies_secondary, secondary_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_secondary_spectra.txt", col=2)
    
    
    # Cut off primary spectra below 511 keV (already cut-off above 3 MeV)
    primary_spectrum_cutoff = primary_spectrum[energies_primary > E_min]
    energies_primary_cutoff = energies_primary[energies_primary > E_min]

    
    # Cut off secondary spectra below 511 keV, and above 3 MeV
    secondary_spectrum_cutoff_1 = secondary_spectrum[energies_secondary < E_max]
    energies_secondary_cutoff_1 = energies_secondary[energies_secondary < E_max]
    secondary_spectrum_cutoff = secondary_spectrum_cutoff_1[energies_secondary_cutoff_1 > E_min]
    energies_secondary_cutoff = energies_secondary_cutoff_1[energies_secondary_cutoff_1 > E_min]
    

    # Load photon primary spectrum
    photon_energies_primary, photon_primary_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_primary_spectra.txt", col=1)    
    # Load photon secondary spectrum
    photon_energies_secondary, photon_secondary_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_secondary_spectra.txt", col=1)
    
    # Peak photon energy, from Eq. 14 of Swagat's summary of Hawking radiation calculations.
    peak_energy = 2.983e13 / m_pbh
    # Compare analytic result for the peak photon energy to the true maximum value.
    print(peak_energy / photon_energies_primary[np.argmax(photon_primary_spectrum)])
    
    
    plt.figure()
    plt.plot(energies_primary, primary_spectrum, label='Primary')
    plt.plot(energies_secondary, secondary_spectrum, 'x', label='Secondary')
    plt.xlim(E_min, E_max)
    plt.ylim(0, 1.1*max(primary_spectrum))
    plt.xlabel('Energy E [GeV]')
    plt.ylabel('$\mathrm{d}^2 n_e / (\mathrm{d}t\mathrm{d}E)$ [cm$^{-3}$ s$^{-1}$ GeV$^{-1}$]')
    plt.title('$M_\mathrm{PBH}$ = ' + "{:.0f}e{:.0f}".format(coefficient, exponent) + 'g')
    plt.legend()
    plt.tight_layout()
    
    
    
    energies_primary_interp = 10**np.linspace(np.log10(E_min), np.log10(E_max), 100000)
    primary_spectrum_interp = np.interp(energies_primary_interp, energies_primary_cutoff, primary_spectrum_cutoff)
    
    #integral_primary = np.trapz(primary_spectrum_cutoff, energies_primary_cutoff)
    integral_primary = np.trapz(primary_spectrum_interp, energies_primary_interp)
    integral_secondary = np.trapz(secondary_spectrum_cutoff, energies_secondary_cutoff)
    integral = integral_primary + integral_secondary
    
    #integral_primary = np.sum(primary_spectrum_cutoff[:-1] * np.diff(energies_primary_cutoff))
    
    f_pbh_P = cm_to_kpc**3 * g_to_GeV * m_pbh * prefactor / (integral_primary * (R - r_s * np.arctan(R/r_s)))
    f_pbh_PS = cm_to_kpc**3 * g_to_GeV * m_pbh * prefactor / (integral * (R - r_s * np.arctan(R/r_s)))
    f_pbh_P_values.append(f_pbh_P)
    f_pbh_PS_values.append(f_pbh_PS)
   
plt.figure()
plt.plot(m_pbh_L19_Iso_1500pc, f_pbh_L19_Iso_1500pc)
plt.plot(m_pbh_values, f_pbh_P_values, 'x', label='Iso, 1.5 kpc')
#plt.plot(m_pbh_values, f_pbh_PS_values, 'x', label='Primary and secondary spectra')
plt.xlim(1e16, 10**(17.1))
plt.ylim(10**(-3.4), 1)
plt.plot()
plt.xlabel('$M_\mathrm{PBH}$ [g]')
plt.ylabel('$f_\mathrm{PBH}$')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.tight_layout()
"""

