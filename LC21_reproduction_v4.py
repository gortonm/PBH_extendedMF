#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:33:43 2022
@author: Matthew Gorton, contributions from Liza Sazonova
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from reproduce_COMPTEL_constraints_v2 import read_blackhawk_spectra, load_data
from tqdm import tqdm
from scipy.integrate import cumulative_trapezoid

# Specify the plot style
mpl.rcParams.update({'font.size': 16,'font.family':'serif'})
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
# colorblind friendly style
plt.style.use('tableau-colorblind10')

# Express all quantities in [g, cm, microgauss, s]

# physical constants
k_B = 8.617333262e-8  # Boltzmann constant, in keV / K
c = 2.99792458e10  # speed of light, in cm / s

# unit conversions
keV_to_K = 1.160451812e7
kpc_to_cm = 3.0857e21
Mpc_to_cm = 3.0857e24
erg_to_GeV = 624.15064799632
solMass_to_g = 1.989e33

# electron/positron mass, in GeV / c^2
m_e = 5.11e-4 / c ** 2

A262 = False
NGC5044 = False
A85 = True
A2199 = False

#%%
if A262:
    # quantities from Table I of Lee & Chan (2021) for A262
    T_c_keV = 1
    T_c_K = T_c_keV * keV_to_K
    rho_s = 14.1e14 * solMass_to_g / (Mpc_to_cm) ** 3
    r_s = 172 * kpc_to_cm
    R = 2 * kpc_to_cm
    z = 0.0161
    beta = 0.433
    r_c = 30 * kpc_to_cm
    n_0 = 0.94e-2
    B_0 = 2.9
    L_0 = 5.6e38 * erg_to_GeV
    extension = "A262"
    
elif A2199:
    # quantities from Table I of Lee & Chan (2021) for A2199
    T_c_keV = 2
    T_c_K = T_c_keV * keV_to_K
    rho_s = 9.56 * 1e14 * solMass_to_g / (Mpc_to_cm) ** 3
    r_s = 334 * kpc_to_cm
    R = 3 * kpc_to_cm
    z = 0.0302
    beta = 0.665
    r_c = 102 * kpc_to_cm
    n_0 = 0.97 * 1e-2
    B_0 = 4.9
    L_0 = 2.3e39 * erg_to_GeV 
    extension = "A2199"
    
elif A85:
    # quantities from Table I of Lee & Chan (2021) for A85
    T_c_keV = 3
    T_c_K = T_c_keV * keV_to_K
    rho_s = 8.34 * 1e14 * solMass_to_g / (Mpc_to_cm) ** 3
    r_s = 444 * kpc_to_cm
    R = 3 * kpc_to_cm
    z = 0.0556
    beta = 0.532
    r_c = 60 * kpc_to_cm
    n_0 = 3.00 * 1e-2
    B_0 = 11.6
    L_0 = 2.7e40 * erg_to_GeV 
    extension = "A85"

elif NGC5044:
    # quantities from Table I of Lee & Chan (2021) for NGC5044
    T_c_keV = 0.8
    T_c_K = T_c_keV * keV_to_K
    rho_s = 14.7e14 * solMass_to_g / (Mpc_to_cm) ** 3
    r_s = 127 * kpc_to_cm
    R = 1 * kpc_to_cm
    z = 0.0090
    beta = 0.524
    r_c = 8 * kpc_to_cm
    n_0 = 4.02e-2
    B_0 = 5.0
    L_0 = 6.3e38 * erg_to_GeV
    extension = "NGC5044"
    

n_steps = 1000

# energy range to integrate over (in GeV)
E_min = m_e * c ** 2
E_max = 5

# radius range (in cm)
r_min = 1e-3
r_values = 10 ** np.linspace(np.log10(r_min), np.log10(R), n_steps)

epsilon = 0.5  # choose 0.5 to maximise magnetic field

const_B = False
numbered_mass_range = True
interp_test = False


# number density, in cm^{-3}
def n(r):
    return n_0 * (1 + r ** 2 / r_c ** 2) ** (-3 * beta / 2)


# magnetic field, in microgauss
def B(r):
    if const_B:
        return B_0
    else:
        return 11 * epsilon ** (-0.5) * np.sqrt(n(r) / 0.1) * (T_c_keV / 2) ** (3 / 4)


# Lorentz factor
def gamma(E):
    return E / E_min


def b_Coul(E, r):
    return 1e-16 * (6.13 * n(r) * (1 + (1 / 75) * np.log(gamma(E) / n(r))))


def b_T(E, r):
    b_IC = 1e-16 * (0.25 * E ** 2 * (1+z)**4)
    b_syn = 1e-16 * (0.0254 * E ** 2 * B(r) ** 2)
    b_brem = 1e-16 * (1.51 * n(r) * (np.log(gamma(E) / n(r)) + 0.36))
    return b_IC + b_syn + b_brem + b_Coul(E, r)


def density_sq(r):
    # write r^2 * DM density to avoid divergence at r=0
    return rho_s * r_s * r / (1 + (r/r_s))**2

def L(m_pbh, r_values, ep_spec, ep_energies):
    
    r_integral = []
    
    for r in r_values:
        outer_E_integrand = []
        
        for E in E_values:
            inner_E_integral = np.trapz(ep_spec[ep_energies>E], ep_energies[ep_energies>E])
            outer_E_integrand.append((b_Coul(E, r) / b_T(E, r)) * inner_E_integral)
            
        outer_E_integral = np.trapz(outer_E_integrand, E_values)
        r_integral.append(density_sq(r) * outer_E_integral)
    
    integral = 8*np.pi*np.trapz(r_integral, r_values) / m_pbh
    return integral
            
def L_optimised(m_pbh, r_values, ep_spec, ep_energies):
    
    # Create meshgrid of energy and radius values, and compute the inner
    # integral (over E prime) at each point
    # method modified from one written by Liza Sazonova
    
    r_grid, E_grid = np.meshgrid(r_values, ep_energies)

    # cumulatively integrate over E prime to calculate the integral of the
    # spectrum in the energy range (E_min, E)
    spectral_int = cumulative_trapezoid(ep_spec, ep_energies, initial=0)
    # subtract from the final value to obtain the integral in the range
    # (E, E_max), rather than (E_min, E)
    spectral_int = spectral_int[-1] - spectral_int

    r_terms = density_sq(r_grid)   # terms depending on r
    b_terms = b_Coul(E_grid, r_grid)/b_T(E_grid, r_grid)
    
    integrand = r_terms*b_terms*spectral_int[:,np.newaxis]
    
    temp = np.trapz(integrand, x=ep_energies, axis=0)
    res  = np.trapz(temp, x=r_values)
    
    return res*np.pi*8/m_pbh

Lambda_0 = 1.4e-27

def L_0():
    #r_values = 10**np.linspace(np.log10(r_min), np.log10(R), 100000)
    r_values = np.linspace(r_min, 2*R, 1000)    
    integrand_values = [n(r)**2 * r**2 for r in r_values] 
    return 4 * np.pi * Lambda_0 * np.sqrt(T_c_K) * np.trapz(integrand_values, r_values)


numbered_mass_range = True
upper_mass_range = False


if numbered_mass_range == True:
    m_pbh_values = 10 ** np.linspace(np.log10(5e14), 17, 25)
    #m_pbh_values = 10 ** np.linspace(np.log10(5e14), 19, 20)
    file_path_data_base = "../Downloads/version_finale/results/"
   
elif upper_mass_range or interp_test:
    file_path_data_base = "../Downloads/blackhawk_v1.1/results"
    
    if upper_mass_range:
        m_pbh_values = 10 ** np.linspace(16, 17, 20)


#%% methods to calculate the primary PBH positron/electron spectrum, using the
# approximate expressions given in Eq. (3) of Lee & Chan (2021)

#c = 3.00e5     # speed of light, in km/s
h_bar = 6.582e-16    # reduced Planck constrant, in eV s
G = 4.30e-3 * (1e5)**2    # Newton's gravitational constant, in pc (cm/s)^2 / solar masses

Msun_to_g = 1.989e33    # convert from 1 solar mass to g
pc_to_km = 3.08567758e18    # convert from 1 pc to cm

greybody_common_factor = (G**2 / (h_bar**3 * c**6 * (1e-9)**3 )) * (pc_to_km / Msun_to_g)**2
matching_factor = 23

# electron/positron greybody factor, using the approximate values given in Eq. (3) of Lee & Chan (2021)
def greybody(E, m_pbh):
    x = (1e9 * G * E * m_pbh / (h_bar * c**3)) * pc_to_km / Msun_to_g
    if x < 1:
        return 16 * greybody_common_factor * (E * m_pbh)**2
    elif x >= 1:
        #print('x >= 1')
        return 27 * greybody_common_factor * (E * m_pbh)**2
    else:
        print('Error: invalid input for PBH mass or electron/positron energy.')
        
# Hawking temperature of a black hole, in GeV, from Eq. (2) of Lee & Chan (2021)
def T(m_pbh):
    return 1.06 * (1e13 / m_pbh)

g_anti = 1    # antiparticle multiplicity
g_helicity = 1     # positron/electron spin multiplicity

# primary electron+positron spectrum
def dNdEdt(E, m_pbh):
    return g_anti * g_helicity * greybody(E, m_pbh) / (2 * np.pi * (np.exp(E / T(m_pbh)) + 1))


# compare greybody factors to those from MacGibbon & Webber (1990)
# Note that the greybody factors from MacGibbon & Webber (1990) are dimensionless, 
# and differ by a factor hbar
print(greybody(E=1e-2, m_pbh=1e13) * (1e-9 * h_bar))
print(greybody(E=10, m_pbh=1e13) * (1e-9 * h_bar))


E_values = 10**np.linspace(np.log10(E_min), np.log10(E_max), 1000)

# Load BlackHawk primary spectrum

m_pbh_values = [1e15, 1e16, 1e17]
x_max = [5, 3e-2, 4e-3]

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
#fig_ratio, ax_ratio = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
ratio_lines = []

for i, m_pbh in enumerate(m_pbh_values):
    ax = axes[i]
    
    exponent = np.floor(np.log10(m_pbh))
    coefficient = m_pbh / 10**exponent
    file_path_data = "../blackhawk_v2.0/results/A22_Fig3_" + "{:.1f}e{:.0f}g/".format(coefficient, exponent)
    
    # Load electron primary spectrum
    energies_primary, primary_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_primary_spectra.txt", col=7)
    
    # Plot dNdEdt as a check
    spectrum = np.array([dNdEdt(E, m_pbh) for E in E_values])
    
    ax.plot(E_values, spectrum, label='Approximate')
    #plt.plot(E_values, matching_factor*spectrum, label='Approximate ' + r'($\times {:.2f}$)'.format(matching_factor))
    ax.plot(energies_primary, 0.5*np.array(primary_spectrum), label='BlackHawk')
    ax.set_xlabel('$E$ [GeV]')
    #ax.set_ylabel('$\mathrm{d}^2 N_{e^\pm} / (\mathrm{d}t~\mathrm{d}E_{e^\pm})$ [s$^{-1}$ GeV$^{-1}$]')
    ax.set_ylim(1e18, 1e22)
    ax.set_xlim(3e-4, x_max[i])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('$M_\mathrm{PBH}$' + '= {:.0e} g'.format(m_pbh))
    
    if i == 0:
        ax.set_ylabel('$\mathrm{d}^2 N_{e^\pm} / (\mathrm{d}t~\mathrm{d}E_{e^\pm})$ [s$^{-1}$ GeV$^{-1}$]')
        
    if i == 2:
        ax.legend(fontsize='small')
        
    ratio = np.array([dNdEdt(E, m_pbh) for E in energies_primary]) / primary_spectrum
    energies_primary = energies_primary[np.isfinite(ratio)]
    ratio = ratio[np.isfinite(ratio)]
    
    #ax_ratio.plot(energies_primary, ratio, label='$M_\mathrm{PBH}$' + '= {:.0e} g'.format(m_pbh))

# add an invisible axis, for the axis labels that apply for the whole figure
#fig.add_subplot(111, frameon=False)
#plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
fig.tight_layout()
"""
ax_ratio.set_xlim(3e-4, 0.1)
#fig_ratio.yscale('log')
ax_ratio.set_xscale('log')
ax_ratio.set_ylabel('Ratio of positron spectra \n [Approximate / BlackHawk]')
ax_ratio.set_xlabel('$E~\mathrm{[GeV]}$')
legend = ['$M_\mathrm{PBH}$' + '= {:.0e} g'.format(m_pbh) for m_pbh in m_pbh_values]
ax_ratio.legend(legend)
fig_ratio.tight_layout()
"""

#%% Calculated using the approximate expressions for the greybody factor

include_secondary = False
secondary_ratio = True

def main(include_secondary=False):
    f_pbh_values = []
    for i, m_pbh in tqdm(enumerate(m_pbh_values), total=len(m_pbh_values)):
        
        ep_energies = 10**np.linspace(np.log10(E_min), np.log10(E_max), n_steps)
        ep_spec_prim = np.array([dNdEdt(E, m_pbh) for E in ep_energies])
        
        if include_secondary:
            file_path_data = file_path_data_base + "LC21_{:.0f}/".format(i + 1)
          
            
            # Load electron primary and secondary spectrum
            energies_primary, primary_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_primary_spectra.txt", col=7)
            energies_secondary, secondary_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_secondary_spectra.txt", col=2)
            
            # interpolate primary spectrum to evaluate at energies which the secondary spectrum is evaluated at
            primary_spectrum_interp = 10**np.interp(np.log10(energies_secondary), np.log10(energies_primary), np.log10(primary_spectrum))
            
            if secondary_ratio:
                ratio_secondary_primary = np.array(secondary_spectrum) / primary_spectrum_interp
                
                # evaluate secondary spectrum at energies used in the direct calculation
                ratio_secondary_primary_interp = 10**(np.interp(np.log10(ep_energies), np.log10(energies_secondary), np.log10(ratio_secondary_primary)))
                secondary_spectrum_interp = ((ratio_secondary_primary_interp-1) * ep_spec_prim)
                
                ep_spec = ep_spec_prim + secondary_spectrum_interp
                
            # calculate secondary spectrum contribution from the *difference* to the primary spectrum from BlackHawk
            else:
                ep_spec_sec = np.array(secondary_spectrum) - primary_spectrum_interp
                ep_spec_sec_interp = 10**np.interp(np.log10(ep_energies), np.log10(energies_secondary), np.log10(ep_spec_sec))
                ep_spec = ep_spec_prim + ep_spec_sec_interp
                
            ep_energies = ep_energies[ep_spec > 0]
            ep_spec = ep_spec[ep_spec > 0]
        
        else:
            ep_spec = ep_spec_prim
                
        luminosity_predicted = L_optimised(m_pbh, r_values, ep_spec, ep_energies)
        f_pbh_values.append(L_0 / luminosity_predicted)
    return f_pbh_values

if __name__ == "__main__":

    file_path_extracted = "./Extracted_files/"
    m_pbh_LC21_extracted, f_PBH_LC21_extracted = load_data(
        "LC21_" + extension + "_NFW.csv"
    )
    
    f_pbh_values = main(include_secondary)
            
    plt.figure(figsize=(6, 4))
    plt.plot(m_pbh_values, np.array(f_pbh_values), label='Reproduction')
    plt.plot(m_pbh_LC21_extracted, f_PBH_LC21_extracted, label='Fig. 1 (Lee \& Chan (2021))', color='tab:orange')
    plt.plot(m_pbh_values, np.array(f_pbh_values)/matching_factor, label=r'Reproduction / {:.1f}'.format(matching_factor), linestyle='dashed', color='tab:green')
    plt.xlabel('$M_\mathrm{PBH}$ [g]')
    plt.ylabel('$f_\mathrm{PBH}$')
    plt.title(extension)
    plt.tight_layout()
    plt.legend(fontsize='small')
    plt.ylim(1e-8, 1)
    if upper_mass_range:
        plt.xlim(1e16, 1e17)
        plt.ylim(1e-2, 1)
    else:
        plt.xlim(4e14, 1e17)
    plt.yscale('log')
    plt.xscale('log')
    plt.tight_layout()
    
    
    extracted_interpolated = 10 ** np.interp(
        np.log10(m_pbh_values),
        np.log10(m_pbh_LC21_extracted),
        np.log10(f_PBH_LC21_extracted),
    )
            
    ratio = extracted_interpolated / np.array(f_pbh_values)
    frac_diff = ratio - 1
    

    plt.figure()
    plt.plot(m_pbh_values, ratio, "x")
    plt.xlabel("$M_\mathrm{PBH}$ [g]")
    plt.ylabel("$f_\mathrm{PBH, extracted} / (23 \times f_\mathrm{PBH, calculated})$")
    #plt.xscale("log")
    #plt.yscale('log')
    plt.xlim(min(m_pbh_values), max(m_pbh_LC21_extracted))  # upper limit is where f_PBH = 1 in Fig. 1 of Lee & Chan (2021)
    plt.title(extension)
    plt.tight_layout()

    plt.figure()
    plt.plot(m_pbh_values, frac_diff, "x")
    plt.xlabel("$M_\mathrm{PBH}$ [g]")
    plt.ylabel("$(f_\mathrm{PBH, extracted} / f_\mathrm{PBH, calculated}) - 1$")
    #plt.xscale("log")
    #plt.yscale('log')
    plt.xlim(min(m_pbh_values), max(m_pbh_LC21_extracted))  # upper limit is where f_PBH = 1 in Fig. 1 of Lee & Chan (2021)
    plt.title(extension)
    plt.tight_layout()
    
    print("M_PBH =", m_pbh_values)
    print("f_PBH =", f_pbh_values)
    print("ratio =", ratio)
    print("1 / ratio =", 1/ratio)
    print("fractional difference =", frac_diff)
    
    if include_secondary:
        ps = '_tot'
    else:
        ps = '_prim'
    np.savetxt('./Extracted_files/fPBH_LC21_' + extension + ps + '.txt', np.transpose([m_pbh_values, f_pbh_values, 1/ratio]), delimiter="\t", fmt="%0.8e")    

    
#%%
matching_factor = 4.5

if include_secondary:
    ps = '_tot'
    title = 'Total emission'
else:
    ps = '_prim'
    title = 'Primary emission only'


file_path_extracted = "./Extracted_files/"

m_pbh_values, f_pbh_A262, ratio_A262 = np.loadtxt("./Extracted_files/fPBH_LC21_A262" + ps + ".txt", delimiter='\t', unpack=True)
m_pbh_values, f_pbh_A2199, ratio_A2199 = np.loadtxt("./Extracted_files/fPBH_LC21_A2199" + ps + ".txt", delimiter='\t', unpack=True)
m_pbh_values, f_pbh_A262, ratio_A85 = np.loadtxt("./Extracted_files/fPBH_LC21_A85" + ps + ".txt", delimiter='\t', unpack=True)
m_pbh_values, f_pbh_NGC5044, ratio_NGC5044 = np.loadtxt("./Extracted_files/fPBH_LC21_NGC5044" + ps + ".txt", delimiter='\t', unpack=True)

"""
plt.figure(figsize=(7.5, 6))
plt.plot(m_pbh_values, ratio_A262-matching_factor, 'x', label="A262")
plt.plot(m_pbh_values, ratio_NGC5044-matching_factor, 'x', label="NGC5044")
plt.xlabel("$M_\mathrm{PBH}$ [g]")
plt.ylabel("$f_\mathrm{PBH, calculated} / f_\mathrm{PBH, extracted}$ " + " - {:.0f}".format(matching_factor))
plt.xscale('log')
plt.ylim(-2, 2)
plt.legend()
plt.tight_layout()
"""

plt.figure(figsize=(6, 6))
plt.plot(m_pbh_values[m_pbh_values < 5e16], matching_factor / (ratio_A262[m_pbh_values < 5e16]), 'x', label="A262")
plt.plot(m_pbh_values[m_pbh_values < 5e16], matching_factor / (ratio_A2199[m_pbh_values < 5e16]), 'x', label="A2199")
plt.plot(m_pbh_values[m_pbh_values < 5e16], matching_factor / (ratio_A85[m_pbh_values < 5e16]), 'x', label="A85")
plt.plot(m_pbh_values[m_pbh_values < 3e16], matching_factor / (ratio_NGC5044[m_pbh_values < 3e16]), 'x', label="NGC5044")
plt.xlabel("$M_\mathrm{PBH}$ [g]")
plt.ylabel(r"4.5 " + " $f_\mathrm{PBH,extracted} / f_\mathrm{PBH,calculated}$")
plt.xscale('log')
plt.xlim(5e14, 5e16)
plt.ylim(0.5, 1.5)
plt.title(title)
plt.legend()
plt.tight_layout()

