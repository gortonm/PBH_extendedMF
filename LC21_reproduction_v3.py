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


# Express all quantities in [g, cm, microgauss, s]

# physical constants
k_B = 8.617333262e-8  # Boltzmann constant, in keV / K
c = 2.99792458e10  # speed of light, in cm / s

# unit conversions
keV_to_K = 1.160451812e7 * k_B
kpc_to_cm = 3.0857e21
Mpc_to_cm = 3.0857e24
erg_to_GeV = 624.15064799632
solMass_to_g = 1.989e33

# electron/positron mass, in GeV / c^2
m_e = 5.11e-4 / c ** 2

A262 = True
NGC5044 = False

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
numbered_mass_range = False
upper_mass_range = False
interp_test = True


#%%
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
    return E / (m_e * c ** 2)


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


def b_term(E, r):
    return b_Coul(E, r)/b_T(E, r)


def L(m_pbh, r_values, ep_spec, ep_energies):
    
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
    
    return res*np.pi*4/m_pbh


if numbered_mass_range == True:
    m_pbh_values = 10 ** np.linspace(np.log10(5e14), 17, 25)
    #m_pbh_values = 10 ** np.linspace(np.log10(5e14), 19, 20)
    file_path_data_base = "../Downloads/version_finale/results/"
   
    
elif upper_mass_range or interp_test:
    file_path_data_base = "../Downloads/blackhawk_v1.1/results"
    
    if upper_mass_range:
        m_pbh_values = 10 ** np.linspace(16, 17, 20)[0:15]

def main():

    for i, m_pbh in tqdm(enumerate(m_pbh_values), total=len(m_pbh_values)):
        
        if upper_mass_range:
            file_path_data = file_path_data_base + "/LC21_upper_range_{:.0f}/".format(i + 1)
            
        elif interp_test:
            file_path_data = file_path_data_base + "/LC21_interp_test_" + extension + "_{:.0f}/".format(len(m_pbh_values)-i)
            print(file_path_data)
        else:
            file_path_data = file_path_data_base + "LC21_{:.0f}/".format(i + 1)
            #file_path_data = file_path_data_base + "LC21_higherM_{:.0f}/".format(i + 1)
            #file_path_data = file_path_data_base + "/100000_steps/LC21_upper_range_{:.0f}/".format(i + 1)

        
        #if upper_mass_range or interp_test:
        if upper_mass_range:
            ep_energies_load, ep_spec_load = read_blackhawk_spectra(file_path_data + "instantaneous_primary_spectra.txt", col=7)
        else:
            ep_energies_load, ep_spec_load = read_blackhawk_spectra(file_path_data + "instantaneous_secondary_spectra.txt", col=2)
        
        ep_energies = ep_energies_load[ep_spec_load > 0]
        ep_spec = ep_spec_load[ep_spec_load > 0]
        
        print("\n Number of non-zero values: ", len(ep_energies))
                    
        print("E_min = {:.2e} GeV".format(min(ep_energies)))
        print("E_max = {:.2e} GeV".format(max(ep_energies)))
        print("M_PBH = {:.2e} g".format(m_pbh))

        # Evaluate photon spectrum at a set of pre-defined energies
        luminosity_predicted = L(m_pbh, r_values, ep_spec, ep_energies)
        print(luminosity_predicted)
        f_pbh_values.append(L_0 / luminosity_predicted)

#%%
if __name__ == "__main__":

    file_path_extracted = "./Extracted_files/"
    m_pbh_LC21_extracted, f_PBH_LC21_extracted = load_data(
        "LC21_" + extension + "_NFW.csv"
    )
    
    f_pbh_values = []
    
    if interp_test:
        m_pbh_values = m_pbh_LC21_extracted[-5:]
        print(m_pbh_values)
    
    main()
    
    index = 3
    f_pbh_PL = f_PBH_LC21_extracted[0] * (m_pbh_LC21_extracted / m_pbh_LC21_extracted[0])**index
    
    plt.figure(figsize=(7, 6))
    plt.plot(m_pbh_values, np.array(f_pbh_values), label='Reproduction')
    plt.plot(m_pbh_LC21_extracted, f_PBH_LC21_extracted, label='Fig. 1 (Lee \& Chan (2021))', color='tab:orange')
    #plt.plot(m_pbh_LC21_extracted, np.array(f_pbh_PL), label='Power-law $(n={:.0f})$'.format(index), color='tab:green')
    plt.plot(m_pbh_values, 0.5*np.array(f_pbh_values), label=r'0.5 $\times$ Reproduction', color='tab:green')

    plt.plot()
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
    
    if interp_test:
        ratio = np.array(f_PBH_LC21_extracted[-5:]) / np.array(f_pbh_values)

    else:
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
    plt.ylabel("$f_\mathrm{PBH, extracted} / f_\mathrm{PBH, calculated}$")
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

    print("f_PBH =", f_pbh_values)
    print("ratio =", ratio)
    print("2 * ratio =", 2*ratio)
    print("fractional difference =", frac_diff)


#%% Look at behaviour of energy loss terms, varying E

r = 0
E_values = 10**np.linspace(np.log10(E_min), np.log10(E_max), n_steps)


b_IC = 1e-16 * (0.25 * E_values ** 2 * (1+z)**4)
b_syn = 1e-16 * (0.0254 * E_values ** 2 * B(r) ** 2)
b_brem = 1e-16 * (1.51 * n(r) * (np.log(gamma(E_values) / n(r)) + 0.36))


plt.figure(figsize=(7, 6))
plt.plot(E_values, b_Coul(E_values, r), label='$b_\mathrm{Coul}$', linestyle='dotted')
plt.plot(E_values, b_IC, label='$b_\mathrm{IC}$', linestyle='dotted')
plt.plot(E_values, b_syn, label='$b_\mathrm{syn}$', linestyle='dotted')
plt.plot(E_values, b_brem, label='$b_\mathrm{brem}$', linestyle='dotted')

plt.plot(E_values, b_T(E_values, r), label='Total', linewidth=2)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('$E$ [GeV]')
plt.ylabel('Energy loss term $[\mathrm{GeV}~\mathrm{s}^{-1}]$')
plt.legend()
plt.tight_layout()
plt.xlim(E_min, E_max)
plt.ylim(1e-18, 3e-15)



#%% Look at behaviour of b_C / b_T, varying E

r = 0
E_values = 10**np.linspace(np.log10(E_min), np.log10(E_max), n_steps)

plot_test = True
label1 = ''

if plot_test:
    label1 = '$b_\mathrm{Coul}/b_\mathrm{T}$'

plt.figure(figsize=(7, 4))
plt.plot(E_values, b_Coul(E_values, r)/b_T(E_values, r), linewidth=2, label=label1)
if plot_test:
    plt.plot(E_values, b_Coul(E_min, r)/b_T(E_values, r), linewidth=2, label='$b_\mathrm{Coul}(E_\mathrm{max})/b_\mathrm{T}$')
    plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$E$ [GeV]')
plt.ylabel('$b_\mathrm{Coul}/b_\mathrm{T}$')
plt.tight_layout()
plt.xlim(E_min, E_max)