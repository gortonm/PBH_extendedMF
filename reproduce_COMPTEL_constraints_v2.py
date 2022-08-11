#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 11:51:13 2022

@author: ppxmg2
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Script to compare the fluxes used in the calculations of Auffinger '22
# Fig. 3 (2201.01265), to the constraints from Coogan, Morrison & Profumo '21
# (2010.04797)

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


file_path_extracted = './Extracted_files/COMPTEL_Esquare_spectrum/'
def load_data(filename):
    return np.genfromtxt(file_path_extracted+filename, delimiter=',', unpack=True)

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


# Coogan, Morrison & Profumo '21 (2010.04797) cites Essig+ '13 (1309.4091) for
# their constraints on the flux, see Fig. 1 of Essig+ '13
E_Essig13_mean, Esquare_spec_Essig13_mean = load_data('COMPTEL_Essig13_mean.csv')
E_Essig13_1sigma, Esquare_spec_Essig13_1sigma = load_data('COMPTEL_Essig13_upper.csv')

# Intensity (rather than E^2 * intensity)
spec_Essig13_mean = Esquare_spec_Essig13_mean / E_Essig13_mean**2
spec_Essig13_1sigma = Esquare_spec_Essig13_1sigma / E_Essig13_1sigma**2

# Bin widths from Essig '13
E_Essig13_bin_lower, a = load_data('COMPTEL_Essig13_lower_x.csv')
E_Essig13_bin_upper, a = load_data('COMPTEL_Essig13_upper_x.csv')

# Flux constraints from Auffinger '22 Fig. 2
E_Auffinger_mean, spec_Auffinger_mean = load_data('Auffinger_Fig2_COMPTEL_mean.csv')
E_Auffinger_bin_lower, a = load_data('Auffinger_Fig2_COMPTEL_lower_x.csv')
E_Auffinger_bin_upper, a  = load_data('Auffinger_Fig2_COMPTEL_upper_x.csv')

bins_upper_Auffinger = E_Auffinger_bin_upper - E_Auffinger_mean
bins_lower_Auffinger = E_Auffinger_mean - E_Auffinger_bin_lower

# convert energy units from MeV to GeV:
E_Essig13_mean = E_Essig13_mean / 1e3
E_Essig13_1sigma = E_Essig13_1sigma / 1e3
E_Essig13_bin_lower = E_Essig13_bin_lower / 1e3
E_Essig13_bin_upper = E_Essig13_bin_upper / 1e3
spec_Essig13_1sigma = spec_Essig13_1sigma * 1e3
spec_Essig13_mean = spec_Essig13_mean * 1e3
    
E_Auffinger_mean = E_Auffinger_mean / 1e3
E_Auffinger_bin_lower = E_Auffinger_bin_lower / 1e3
E_Auffinger_bin_upper = E_Auffinger_bin_upper / 1e3
spec_Auffinger_mean = spec_Auffinger_mean * 1e3

# calculate errors and bin edges
error_Essig13 = spec_Essig13_1sigma - spec_Essig13_mean
spec_Essig_13_2sigma = spec_Essig13_1sigma + 2*(error_Essig13)
bins_upper_Essig13 = E_Essig13_bin_upper - E_Essig13_mean
bins_lower_Essig13 = E_Essig13_mean - E_Essig13_bin_lower

bins_upper_Auffinger = E_Auffinger_bin_upper - E_Auffinger_mean
bins_lower_Auffinger = E_Auffinger_mean - E_Auffinger_bin_lower

g_to_solar_mass = 1 / 1.989e33     # convert g to solar masses
pc_to_cm = 3.09e18    # convert pc to cm
MeV_to_g = 1.782662e-27    # convert MeV/c^2 to g

# Astrophysical parameters

# Auffinger '22
rho_0_Auffinger = 0.0125     # DM density at the Sun, in solar masses / pc^3
r_s_Auffinger = 17 * 1e3    # scale radius, in pc
r_0_Auffinger = 8.5 * 1e3    # galactocentric distance of Sun, in pc
# CMP '21
rho_0_CMP = 0.00990     # DM density at the Sun, in solar masses / pc^3
r_s_CMP = 11 * 1e3   # scale radius, in pc
r_0_CMP = 8.12 * 1e3    # galactocentric distance of Sun, in pc

J_dimensionless_CMP21 = 6.82   # dimensionless J-factor from Essig '13 Table I

# range of galactic latitude/longitude observed by COMPTEL
b_max_CMP, l_max_CMP = np.radians(20), np.radians(60)
delta_Omega_Auffinger = 4 * l_max_CMP * np.sin(b_max_CMP)


b_max_Auffinger, l_max_Auffinger = np.radians(15), np.radians(30)
delta_Omega_Auffinger = 4 * l_max_CMP * np.sin(b_max_CMP)

# find J-factor, as defined in A22 and CMP21
#J_CMP21 = J_dimensionless_CMP21 * rho_0_CMP * r_0_CMP / delta_Omega_CMP
J_CMP21 = 4.866e25  * (MeV_to_g)    # from Table I of CMP21, converted to units of GeV pc^{-2}
J_A22 = 3.65 * rho_0_Auffinger * r_0_Auffinger / delta_Omega_Auffinger

# Calculate "flux quantities"
f_PBH_A22 = []
f_PBH_CMP21 = []

m_pbh_values = np.array([0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1, 2, 3, 4, 6, 8]) * 10**16
for m_pbh in m_pbh_values:
        
    exponent = np.floor(np.log10(m_pbh))
    coefficient = m_pbh / 10**exponent
    file_path_data = "../blackhawk_v2.0/results/A22_Fig3_" + "{:.0f}e{:.0f}g/".format(coefficient, exponent)
    
    # Load photon spectra from BlackHawk outputs
    energies_primary, primary_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_secondary_spectra.txt", col=1)
    
    # Find flux measured (plus an error, if appropriate), divided by the 
    # integral of the photon spectrum over the energy (energy range given by the
    # bin width), multiplied by the bin width
    
    CMP_flux_quantity = []
    Auffinger_flux_quantity = []
    
    CMP21 = True
    A22 = False
    # COMPTEL data used in Coogan, Morrison & Profumo (2021) (2010.04797)
    for i in range(0, 9):
        E_min = E_Essig13_bin_lower[i]    # convert from MeV to GeV
        E_max = E_Essig13_bin_upper[i]    # convert from MeV to GeV

        # Load photon primary spectrum
        energies_primary_interp = 10**np.linspace(np.log10(E_min), np.log10(E_max), 100000)
        primary_spectrum_interp = np.interp(energies_primary_interp, energies_primary, primary_spectrum)
        integral_primary = np.trapz(primary_spectrum_interp, energies_primary_interp)
        
        CMP_flux_quantity.append(spec_Essig_13_2sigma[i] * (E_max - E_min) / integral_primary)
        
    # COMPTEL data used in Auffinger (2022) (2201.01265)
    A22 = True
    for i in range(0, 3):
        
        E_min = E_Auffinger_bin_lower[i]    # convert from MeV to GeV
        E_max = E_Auffinger_bin_upper[i]    # convert from MeV to GeV       
        
        # Load photon primary spectrum
        energies_primary_interp = 10**np.linspace(np.log10(E_min), np.log10(E_max), 100000)
        primary_spectrum_interp = np.interp(energies_primary_interp, energies_primary, primary_spectrum)
        integral_primary = np.trapz(primary_spectrum_interp, energies_primary_interp)
        
        Auffinger_flux_quantity.append(spec_Auffinger_mean[i] * (E_max - E_min) / integral_primary)
    
    f_PBH_A22.append(4 * np.pi * m_pbh * min(Auffinger_flux_quantity) * (pc_to_cm)**2 * (g_to_solar_mass)  / J_A22)
    f_PBH_CMP21.append(4 * np.pi * m_pbh * min(CMP_flux_quantity) / J_CMP21 )

    print('M_{PBH} [g] : ' + ' {0:1.0e}'.format(m_pbh))
    print('Bin with minimum f_{PBH, i} : ', np.argmin(Auffinger_flux_quantity))

# Load result extracted from Fig. 3 of CMP '21
file_path_extracted = './Extracted_files/'
m_pbh_CMP21_extracted, f_PBH_CMP21_extracted = load_data("CMP21_Fig3.csv")
m_pbh_A22_extracted, f_PBH_A22_extracted = load_data("A22_Fig3.csv")

plt.figure(figsize=(7,7))
plt.plot(m_pbh_CMP21_extracted, f_PBH_CMP21_extracted, label="CMP '21 (Extracted)")
plt.plot(m_pbh_A22_extracted, f_PBH_A22_extracted, label="Auffinger '22 (Extracted)")
plt.plot(m_pbh_values[m_pbh_values > 2e15], np.array(f_PBH_CMP21)[m_pbh_values > 2e15], 'x', label="CMP '21 (Reproduced)")
plt.plot(m_pbh_values, f_PBH_A22, 'x', label="Auffinger '22 (Reproduced)")

plt.xlabel('$M_\mathrm{PBH}$ [g]')
plt.ylabel('$f_\mathrm{PBH}$')
plt.tight_layout()
plt.legend(fontsize='small')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e14, 1e18)
plt.ylim(1e-10, 1)


#%% Attempt to calculate J-factor for CMP '21
from scipy.integrate import tplquad

MeV_to_Modot = 8.96260432e-61

def rho_NFW(r):
    if CMP21:
        rho_0 = rho_0_CMP
        r_s = r_s_CMP
    elif A22:
        rho_0 = rho_0_Auffinger
        r_s = r_s_Auffinger
        
    #print(r_s)
    return rho_0 * ((r/r_s) * (1 + (r/r_s))**2)**(-1)

def r(los, b, l):
    if CMP21:
        r_0 = r_0_CMP
    elif A22:
        r_0 = r_0_Auffinger
    return np.sqrt(los**2 + r_0**2 - 2*r_0*los*np.cos(b)*np.cos(l))

def j_integrand(los, b, l):
    return rho_NFW(r(los, b, l)) * np.cos(b)
    #return rho_NFW(r(los, b, l))

def j(b_max, l_max):
    delta_omega = 4 * l_max * np.sin(b_max)  # correct (Table I Cirelli+ '2011) 
    print(b_max * (180/np.pi))
    print('delta_omega = ', delta_omega)
    if CMP21:
        r_0 = r_0_CMP
    elif A22:
        r_0 = r_0_Auffinger
    return 4 * np.array(tplquad(j_integrand, 0, l_max, 0, b_max, 0, 0.99999*r_0_CMP)) * (1/MeV_to_Modot) * (1/pc_to_cm)**2 / delta_omega
    
r_0_Essig13 = 8.5 * 1e3
rho_0_Essig13 = 0.3 * 0.026331617
def j_dimensionless(b_max, l_max):
    delta_omega = 4 * l_max * np.sin(b_max)  # correct (Table I Cirelli+ '2011)   
    return 4 * np.array(tplquad(j_integrand, 0, l_max, 0, b_max, 0, 0.99999*r_0_Essig13)) / (rho_0_Essig13 * r_0_Essig13 * delta_omega)



def r2(los, theta):
    return np.sqrt(los**2 + r_0_CMP**2 - 2*r_0_CMP*los*np.cos(theta))


from reproduce_extended_MF import double_integral, triple_integral
from scipy.integrate import dblquad

def j_psi(psi, n_steps=10000):
    los_values = np.linspace(0, r_0_CMP, n_steps)
    return np.trapz(rho_NFW(r2(los_values, psi)), los_values)

def j_theta(theta_max):
    delta_omega = 2 * np.pi * (1-np.cos(theta_max))
    print(delta_omega)
    theta_values = np.linspace(1e-5, theta_max, 10000)
    r_0_CMP21 = True
    integrand_values = [np.sin(theta) * j_psi(theta) for theta in theta_values]
    return (2 * np.pi / (delta_omega)) * np.trapz(integrand_values, theta_values) * (1/MeV_to_Modot) * (1/pc_to_cm)**2
"""
CMP21 = True
A22 = False    
print(j_dimensionless(b_max_CMP, l_max_CMP) / 3.65)

CMP21 = False
A22 = True
print(j_dimensionless(b_max_Auffinger, l_max_Auffinger) / 6.82)

CMP21 = True
A22 = False
print(j(b_max_CMP, l_max_CMP) / 4.866e25)
"""
#print(j(np.radians(5), np.radians(5)))
#print(j(np.radians(5), np.radians(30)))

CMP21 = True
print(j_theta(np.radians(5)) / 1.597e26)
    
"""
def arcsech(x):
    return np.log( (1 + np.sqrt(1-x**2))/ x)

def X(s):
    if 0 <= s <= 1:
        return arcsech(s) / np.sqrt(1 - s**2)
    else: 
        return arcsech(s) / np.sqrt(s**2 - `)
    
def f_j_factor(d):
    return (4 * np.pi * rho_0_CMP * r_s_CMP**3 / d**2) * (np.log(d*theta/(2*r_S_CMP)) + X(d*theta / r_s)) - d

r_0_Essig13 = 8.5 * 1e3


print(j_dimensionless(b_max_CMP, l_max_CMP))

print(j(np.radians(2.5), np.radians(2.5)))
print(j(b_max_CMP, l_max_CMP))



R = np.sqrt(b**2 + l**2)
theta = np.tan(b/l)
"""