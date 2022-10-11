#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 13:54:33 2022

@author: ppxmg2
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from reproduce_COMPTEL_constraints_v2 import load_data, read_blackhawk_spectra
from J_factor_A22 import j_avg

# Script to understand the meaning of the 'refined flux' values used in Isatis

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


file_path_extracted = './Extracted_files/'

#%%

# Unit conversions
g_to_solar_mass = 1 / 1.989e33    # g to solar masses
pc_to_cm = 3.0857e18    # conversion factor from pc to cm

r_s = 17 * 1000 * pc_to_cm    # scale radius, in cm
r_odot = 8.5 * 1000 * pc_to_cm   # galactocentric solar radius, in cm

rho_0 = 8.5e-25	# characteristic halo density in g/cm^3

b_max_Auffinger, l_max_Auffinger = np.radians(15), np.radians(30)

def find_r(los, b, l):
    return np.sqrt(r_odot**2 + los**2 - 2*r_odot*los*np.cos(b)*np.cos(l))


def rho_NFW(los, b, l):
    r = find_r(los, b, l)
    return rho_0 * (r_s / r) * (1 + (r/r_s))**(-2)


def refined_energies(energies, n_refined):
    E_min = min(energies)
    E_max = max(energies)
    step = (np.log10(10*E_max) - np.log10(E_min/10)) / (n_refined - 1)
    
    ener_refined = []
    for i in range(0, n_refined):
        ener_refined.append( 10**(np.log10(E_min/10) + i*step) )
    return ener_refined

def refined_flux(flux, ener_spec, n_refined):
    ener_refined = refined_energies(energies, n_refined)
    
    nb_spec = len(ener_spec)
    
    flux_refined = []
    
    c = 0
    for i in range(n_refined):
        while c < nb_spec and ener_spec[c] < ener_refined[i]:
            c += 1
        if c > 0 and c < nb_spec and flux[c-1] !=0:
            y = np.log10(flux[c-1]) + ((np.log10(flux[c]) - np.log10(flux[c-1])) / (np.log10(ener_spec[c]) - np.log10(ener_spec[c-1]))) * (np.log10(ener_refined[i]) - np.log10(ener_spec[c-1]))
            flux_refined.append(10**y)
        else: 
            flux_refined.append(0)
            
    return flux_refined

def J_D(l_min, l_max, b_min, b_max):
    nb_angles = 100
    nb_radii = 100
    r_max = 200 * 1000 * pc_to_cm  # maximum radius of Galactic halo in cm
    
    b, l = [], []
    for i in range(0, nb_angles):
        l.append(l_min + i*(l_max - l_min)/(nb_angles - 1))
        b.append(b_min + i*(b_max - b_min)/(nb_angles - 1))
        
    result = 0
    for i in range(0, nb_angles-1): # integral over l
        for j in range(0, nb_angles-1): # integral over b
            s_max = r_odot * np.cos(l[i]) * np.cos(b[j]) + np.sqrt(r_max**2 - r_odot**2 * (1-(np.cos(l[i])*np.cos(b[j]))**2))
            s = []
            for k in range(0, nb_radii):
                s.append(0. + k * (s_max - 0.) / (nb_radii - 1))
                
            
            for k in range(0, nb_radii-1): # integral over s(r(l, b))
                metric = abs(np.cos(b[i])) * (l[i+1] - l[i]) * (b[j+1] - b[j]) * (s[k+1] - s[k])
                result += metric * rho_NFW(s[k], b[j], l[i])
                
    Delta = 0
    for i in range(0, nb_angles-1):
        for j in range(0, nb_angles-1):
            Delta += abs(np.cos(b[i])) * (l[i+1] - l[i]) * (b[j+1] - b[j])
    return result / Delta


def J_D_v2(l_min, l_max, b_min, b_max):
    nb_angles = 100
    nb_radii = 100
    r_max = 200 * 1000 * pc_to_cm  # maximum radius of Galactic halo in cm
    
    b, l = [], []
    for i in range(0, nb_angles):
        l.append(l_min + i*(l_max - l_min)/(nb_angles - 1))
        b.append(b_min + i*(b_max - b_min)/(nb_angles - 1))
        
    result = 0
    for i in range(0, nb_angles-1): # integral over l
        for j in range(0, nb_angles-1): # integral over b
            s_max = r_odot * np.cos(l[i]) * np.cos(b[j]) + np.sqrt(r_max**2 - r_odot**2 * (1-(np.cos(l[i])*np.cos(b[j]))**2))
            s = []
            for k in range(0, nb_radii):
                s.append(0. + k * (s_max - 0.) / (nb_radii - 1))
                
            
            for k in range(0, nb_radii-1): # integral over s(r(l, b))
                metric = abs(np.cos(b[j])) * (l[i+1] - l[i]) * (b[j+1] - b[j]) * (s[k+1] - s[k])
                result += metric * rho_NFW(s[k], b[j], l[i])
                
    Delta = 0
    for i in range(0, nb_angles-1):
        for j in range(0, nb_angles-1):
            Delta += abs(np.cos(b[j])) * (l[i+1] - l[i]) * (b[j+1] - b[j])
    return result / Delta



def J_D_v2a(l_min, l_max, b_min, b_max):
    nb_angles = 1000
    nb_radii = 1000
    r_max = 200 * 1000 * pc_to_cm  # maximum radius of Galactic halo in cm
    
    b, l = [], []
    for i in range(0, nb_angles):
        l.append(l_min + i*(l_max - l_min)/(nb_angles - 1))
        b.append(b_min + i*(b_max - b_min)/(nb_angles - 1))
        
    result = 0
    for i in range(0, nb_angles-1): # integral over l
        for j in range(0, nb_angles-1): # integral over b
            s_max = r_odot * np.cos(l[i]) * np.cos(b[j]) + np.sqrt(r_max**2 - r_odot**2 * (1-(np.cos(l[i])*np.cos(b[j]))**2))
            s = []
            for k in range(0, nb_radii):
                s.append(0. + k * (s_max - 0.) / (nb_radii - 1))
                
            
            for k in range(0, nb_radii-1): # integral over s(r(l, b))
                metric = abs(np.cos(b[j])) * (l[i+1] - l[i]) * (b[j+1] - b[j]) * (s[k+1] - s[k])
                result += metric * rho_NFW(s[k], b[j], l[i])
                
    Delta = 0
    for i in range(0, nb_angles-1):
        for j in range(0, nb_angles-1):
            Delta += abs(np.cos(b[j])) * (l[i+1] - l[i]) * (b[j+1] - b[j])
    return result / Delta




def J_D_v2b(l_min, l_max, b_min, b_max):
    
    r_s = 17 * 1000    # scale radius, in pc
    r_odot = 8.5 * 1000   # galactocentric solar radius, in pc
    rho_0 = 0.0125 * g_to_solar_mass**(-1) * pc_to_cm**(-3)  # normaliseation for NFW profile, in g cm^{-3}
    r_max = 200 * 1000    # maximum halo radius, in pc

    nb_angles = 100
    nb_radii = 100
    r_max = 200 * 1000 * pc_to_cm  # maximum radius of Galactic halo in cm
    
    b, l = [], []
    for i in range(0, nb_angles):
        l.append(l_min + i*(l_max - l_min)/(nb_angles - 1))
        b.append(b_min + i*(b_max - b_min)/(nb_angles - 1))
        
    result = 0
    for i in range(0, nb_angles-1): # integral over l
        for j in range(0, nb_angles-1): # integral over b
            s_max = r_odot * np.cos(l[i]) * np.cos(b[j]) + np.sqrt(r_max**2 - r_odot**2 * (1-(np.cos(l[i])*np.cos(b[j]))**2))
            s = []
            for k in range(0, nb_radii):
                s.append(0. + k * (s_max - 0.) / (nb_radii - 1))
                
            
            for k in range(0, nb_radii-1): # integral over s(r(l, b))
                metric = abs(np.cos(b[j])) * (l[i+1] - l[i]) * (b[j+1] - b[j]) * (s[k+1] - s[k])
                result += metric * rho_NFW(s[k], b[j], l[i])
                
    Delta = 0
    for i in range(0, nb_angles-1):
        for j in range(0, nb_angles-1):
            Delta += abs(np.cos(b[j])) * (l[i+1] - l[i]) * (b[j+1] - b[j])
    return result / Delta


#%%

def galactic(spectrum):
    n_spec = len(spectrum)
    
    # Calculate J-factor
    b_max_Auffinger, l_max_Auffinger = np.radians(15), np.radians(30)
    j_factor = J_D(-l_max_Auffinger, l_max_Auffinger, -b_max_Auffinger, b_max_Auffinger)
    print('J [g cm^{-2} sr^{-1} = ', j_factor)
    
    galactic = []
    for i in range(n_spec):
        val = j_factor * spectrum[i] / (4*np.pi*m_pbh)
        galactic.append(val)
        
    return np.array(galactic)

f_PBH_isatis = []
#m_pbh_values = np.array([0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.5, 2, 3, 4, 6, 8]) * 10**16
m_pbh_values = 10**np.linspace(14, 18, 25)

# COMPTEL data
flux_minus = np.array([5.40770e-01, 7.80073e-02, 7.83239e-03])
flux_plus = np.array([5.21580e-01, 7.35839e-02, 7.78441e-03])
flux = np.array([2.38601e+00, 3.44189e-01, 3.40997e-02])

energies_minus = np.array([7.21020e-04, 2.50612e-03, 7.20580e-03])
energies_plus = np.array([1.23204e-03, 4.47746e-03, 1.26645e-02])
energies = np.array([1.73836e-03, 5.51171e-03, 1.73730e-02])

# Number of interpolation points
n_refined = 500


for i, m_pbh in enumerate(m_pbh_values):
    # Load photon spectra from BlackHawk outputs
    exponent = np.floor(np.log10(m_pbh))
    coefficient = m_pbh / 10**exponent
    
    #file_path_data = "../blackhawk_v2.0/results/A22_Fig3_" + "{:.1f}e{:.0f}g/".format(coefficient, exponent)
    file_path_data = "./../Downloads/version_finale/results/runs_COMPTEL_0510_{:.0f}/".format(i+1)

    print("{:.1f}e{:.0f}g/".format(coefficient, exponent))
    
    ener_spec, spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_secondary_spectra.txt", col=1)
        
    flux_galactic = galactic(spectrum)
    ener_refined = refined_energies(energies, n_refined)
    flux_refined = refined_flux(flux_galactic, ener_spec, n_refined)
    
    def binned_flux(galactic_refined, ener_refined, ener_COMPTEL, ener_COMPTEL_minus, ener_COMPTEL_plus):
        flux_binned = []
        nb_refined = len(galactic_refined)
        nb_COMPTEL = len(ener_COMPTEL)
        
        for i in range(nb_COMPTEL):
            val_binned = 0
            c = 0
            while c < nb_refined and ener_refined[c] < ener_COMPTEL[i] - ener_COMPTEL_minus[i]:
                c += 1
            if c > 0 and c+1 < nb_refined:
                while c < nb_refined and ener_refined[c] < ener_COMPTEL[i] + ener_COMPTEL_plus[i]:
                    val_binned += (ener_refined[c+1] - ener_refined[c]) * (galactic_refined[c+1] + galactic_refined[c]) / 2
                    c += 1
            #print(c)
            #print(ener_refined[c])
            #print(ener_COMPTEL[i] + ener_COMPTEL_plus[i])
            flux_binned.append(val_binned)
        #print(val_binned)
        return np.array(flux_binned)
    
    # Print calculated and mean flux:
    bin_of_interest = 3
    print('Binned flux (bin {:.0f}) = {:.6e}'.format(bin_of_interest, (g_to_solar_mass * (pc_to_cm)**(2))**(-1) * binned_flux(flux_refined, ener_refined, energies, energies_minus, energies_plus)[bin_of_interest-1]))
    print('Measured flux (bin {:.0f}) = {:.6e}'.format(bin_of_interest, (flux * (energies_plus + energies_minus))[bin_of_interest-1]))
    
    # Calculate constraint on f_PBH
    f_PBH = min(flux * (energies_plus + energies_minus) / binned_flux(flux_refined, ener_refined, energies, energies_minus, energies_plus))
    f_PBH_isatis.append(f_PBH)

# Load result extracted from Fig. 3 of Auffinger '22
file_path_extracted = './Extracted_files/'
m_pbh_A22_extracted, f_PBH_A22_extracted = load_data("A22_Fig3.csv")

# Load result from Isatis

Isatis_path = './../Downloads/version_finale/scripts/Isatis/'
results_name = "test_COMPTEL" 

constraints_file = np.genfromtxt("%sresults_photons_%s.txt"%(Isatis_path,results_name),dtype = "str")
constraints_names_bis = constraints_file[0,1:]
constraints = np.zeros([len(constraints_file)-1,len(constraints_file[0])-1])
for i in range(len(constraints)):
    for j in range(len(constraints[0])):
        constraints[i,j] = float(constraints_file[i+1,j+1])

constraint_COMPTEL = []
for i in range(len(constraints)):
    constraint_COMPTEL.append(constraints[i][1])

 
results_name_mod = "test_COMPTEL_modified" 

constraints_file = np.genfromtxt("%sresults_photons_%s.txt"%(Isatis_path,results_name_mod),dtype = "str")
constraints_names_bis = constraints_file[0,1:]
constraints = np.zeros([len(constraints_file)-1,len(constraints_file[0])-1])
for i in range(len(constraints)):
    for j in range(len(constraints[0])):
        constraints[i,j] = float(constraints_file[i+1,j+1])

constraint_COMPTEL_modified = []
for i in range(len(constraints)):
    constraint_COMPTEL_modified.append(constraints[i][1])

    
Mmin = 1e14
Mmax = 1e18
masses_Isatis = np.logspace(np.log10(Mmin),np.log10(Mmax),len(constraint_COMPTEL))

plt.plot(masses_Isatis, constraint_COMPTEL, 'x', label="Isatis (unmodified)")
plt.plot(masses_Isatis, constraint_COMPTEL_modified, 'x', label="Isatis (modified loop condition)", markersize='9', color='tab:red', alpha=0.7)
plt.plot(m_pbh_values, f_PBH_isatis, 'x', label="Reproduction (BlackHawk spectra)", color='g')

plt.xlabel('$M_\mathrm{PBH}$ [g]')
plt.ylabel('$f_\mathrm{PBH}$')
plt.tight_layout()
plt.legend(fontsize='small')
plt.xscale('log')
plt.yscale('log')
#plt.title('Excluding highest-energy bin')
plt.xlim(1e14, 1e18)
plt.ylim(1e-10, 1)
plt.tight_layout()




plt.figure(figsize=(8,8))
plt.plot(m_pbh_A22_extracted, f_PBH_A22_extracted, label="Extracted from Auffinger '22 ")
plt.plot(masses_Isatis, constraint_COMPTEL, 'x', label="Isatis (unmodified)")
plt.plot(masses_Isatis, constraint_COMPTEL_modified, 'x', label="Isatis (modified loop condition)", markersize='9', color='tab:red', alpha=0.7)
plt.plot(m_pbh_values, f_PBH_isatis, 'x', label="Reproduction (BlackHawk spectra)", color='g')

plt.xlabel('$M_\mathrm{PBH}$ [g]')
plt.ylabel('$f_\mathrm{PBH}$')
plt.tight_layout()
plt.legend(fontsize='small')
plt.xscale('log')
plt.yscale('log')
#plt.title('Excluding highest-energy bin')
plt.xlim(1e14, 1e18)
plt.ylim(1e-10, 1)
plt.tight_layout()


# Plot only unmodified Isatis result and my reproduction (for email to Jeremy Auffinger)
plt.figure(figsize=(7,7))
plt.plot(masses_Isatis, constraint_COMPTEL, 'x', label="Isatis")
plt.plot(m_pbh_values, np.array(f_PBH_isatis), 'x', label="Reproduction")

plt.xlabel('$M_\mathrm{PBH}$ [g]')
plt.ylabel('$f_\mathrm{PBH}$')
plt.tight_layout()
plt.legend(fontsize='small')
plt.xscale('log')
plt.yscale('log')
#plt.title('Excluding highest-energy bin')
plt.xlim(1e14, 1e18)
plt.ylim(1e-10, 1)
plt.tight_layout()


#%%
plt.figure()
plt.plot(ener_spec, flux_galactic, 'x', label = 'Original')
plt.plot(ener_refined, flux_refined, 'x', label='Refined')
plt.title('$M_\mathrm{PBH}$ = ' + "{:.1f}e{:.0f}".format(coefficient, exponent) + 'g')
plt.xlabel('Energy [GeV]')
plt.ylabel('${\\rm d}\Phi/{\\rm d}E\,\, ({\\rm GeV^{-1}} \cdot {\\rm s}^{-1}\cdot{\\rm cm}^{-2} \cdot {\\rm sr}^{-1})$')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.tight_layout()
