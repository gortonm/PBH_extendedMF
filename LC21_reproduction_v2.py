#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:33:43 2022

@author: ppxmg2
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, dblquad

from reproduce_COMPTEL_constraints_v2 import read_blackhawk_spectra, load_data
from tqdm import tqdm

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

# energy range to integrate over (in GeV)
E_min = m_e * c ** 2
E_max = 5

r_min = 1
n_steps = 1000

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

epsilon = 0.5  # choose 0.5 to maximise magnetic field

const_B = True
scipy = False
trapz = True
numbered_mass_range = True

r_values = 10 ** np.linspace(np.log10(r_min), np.log10(R), n_steps)
E_values = 10 ** np.linspace(np.log10(E_min), np.log10(E_max), n_steps)

# DM density for a NFW profile, in g cm^{-3}
def rho_NFW(r):
    return rho_s * (r_s / r) / (1 + (r / r_s)) ** 2


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


def spec(E, ep_energies, ep_spec):
    return np.interp(E, ep_energies, ep_spec)


def Q(E, r, m_pbh, ep_energies, ep_spec):
    return spec(E, ep_energies, ep_spec) * rho_NFW(r) / m_pbh


def dndE(E, r, m_pbh, ep_energies, ep_spec):
    E_prime_values = 10 ** np.linspace(np.log10(E), np.log10(E_max), n_steps)
    Q_values = [Q(E_prime, r, m_pbh, ep_energies, ep_spec) for E_prime in E_prime_values]
    
    if trapz:
        return np.trapz(Q_values, E_prime_values) / b_T(E, r)

    if scipy:
        return quad(Q, E, E_max, args=(r, m_pbh, ep_energies, ep_spec))[0] / b_T(E, r)

def L_integrand(E, r, m_pbh, ep_energies, ep_spec):
    return dndE(E, r, m_pbh, ep_energies, ep_spec) * r ** 2 * b_Coul(E, r)


def L(m_pbh, ep_energies, ep_spec):

    if trapz:
        integrand = [np.trapz(L_integrand(E_values, r, m_pbh, ep_energies, ep_spec), E_values) for r in r_values]
        return 4 * np.pi * np.trapz(integrand, r_values)

    if scipy:
        return 4 * np.pi * np.array(dblquad(L_integrand, r_min, R, E_min, E_max, args=(m_pbh, ep_energies, ep_spec) )[0])


if numbered_mass_range == True:
    #m_pbh_values = 10 ** np.linspace(np.log10(5e14), 17, 25)
    m_pbh_values = 10 ** np.linspace(np.log10(5e14), 19, 20)
    file_path_data_base = "../Downloads/version_finale/results/"


f_pbh_values = []

energies_ref = 10 ** np.linspace(np.log10(E_min), np.log10(E_max), n_steps)


def main():
    for i, m_pbh in tqdm(enumerate(m_pbh_values), total=len(m_pbh_values)):

        if i % 1 == 0:

            #file_path_data = file_path_data_base + "LC21_{:.0f}/".format(i + 1)
            file_path_data = file_path_data_base + "LC21_higherM_{:.0f}/".format(i + 1)

            ep_energies_load, ep_spec_load = read_blackhawk_spectra(
                file_path_data + "instantaneous_secondary_spectra.txt", col=2
            )
            
            ep_energies = ep_energies_load[ep_spec_load > 0]
            ep_spec = ep_spec_load[ep_spec_load > 0]

            print("M_PBH = {:.2e} g".format(m_pbh))

            # Evaluate photon spectrum at a set of pre-defined energies
            luminosity_predicted = L(m_pbh, ep_energies, ep_spec)
            f_pbh_values.append(L_0 / luminosity_predicted)


if __name__ == "__main__":

    file_path_extracted = "./Extracted_files/"
    m_pbh_LC21_extracted, f_PBH_LC21_extracted = load_data(
        "LC21_" + extension + "_NFW.csv"
    )

    f_pbh_values = []
    main()
    
    

    index = 3
    f_pbh_PL = f_PBH_LC21_extracted[0] * (m_pbh_LC21_extracted / m_pbh_LC21_extracted[0])**index
    
    plt.figure(figsize=(7, 6))
    plt.plot(m_pbh_values, 0.5*np.array(f_pbh_values), label='Reproduction')
    plt.plot(m_pbh_LC21_extracted, f_PBH_LC21_extracted, label='Extracted', color='tab:orange')
    plt.plot(m_pbh_LC21_extracted, np.array(f_pbh_PL), label='Power-law $(n={:.0f})$'.format(index), color='tab:green')

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


    extracted_interpolated = 10 ** np.interp(
        np.log10(m_pbh_values),
        np.log10(m_pbh_LC21_extracted),
        np.log10(f_PBH_LC21_extracted),
    )
    extracted_interpolated_fewer = []
    m_pbh_fewer = []
    for i in range(0, len(extracted_interpolated)):
        if i % 1 == 0:
            extracted_interpolated_fewer.append(extracted_interpolated[i])
            m_pbh_fewer.append(m_pbh_values[i])
    ratio = extracted_interpolated_fewer / np.array(f_pbh_values)
    frac_diff = ratio - 1

    plt.figure()
    plt.plot(m_pbh_fewer, f_pbh_values, label='Reproduction')
    plt.plot(m_pbh_LC21_extracted, f_PBH_LC21_extracted, label="Extracted (Fig. 1)")
    plt.xlabel("$M_\mathrm{PBH}$ [g]")
    plt.ylabel("$f_\mathrm{PBH}$")
    plt.title(extension)
    plt.tight_layout()
    plt.legend()
    #plt.ylim(1e-8, 1)
    #plt.xlim(4e14, 1e17)
    plt.yscale("log")
    plt.xscale("log")

    plt.figure()
    plt.plot(m_pbh_fewer, ratio, "x")
    plt.xlabel("$M_\mathrm{PBH}$ [g]")
    plt.ylabel("$f_\mathrm{PBH, extracted} / f_\mathrm{PBH, calculated}$")
    plt.xscale("log")
    # plt.yscale('log')
    #plt.xlim(4e14, 6e16)  # upper limit is where f_PBH = 1 in Fig. 1 of Lee & Chan (2021)
    plt.title(extension)
    plt.tight_layout()

    plt.figure()
    plt.plot(m_pbh_fewer, frac_diff, "x")
    plt.xlabel("$M_\mathrm{PBH}$ [g]")
    plt.ylabel("$(f_\mathrm{PBH, extracted} / f_\mathrm{PBH, calculated}) - 1$")
    plt.xscale("log")
    # plt.yscale('log')
    plt.xlim(4e14, 6e16)  # upper limit is where f_PBH = 1 in Fig. 1 of Lee & Chan (2021)
    plt.title(extension)
    plt.tight_layout()

    print("f_PBH =", f_pbh_values)
    print("ratio =", ratio)
    print("fractional difference =", frac_diff)


#%% Test case for small r_c, r_s
R = min((r_c/100, r_s/100))

def b_Coul_approx(E):
    return b_Coul(E, r=0)

def b_T_approx(E):
    return b_T(E, r=0)


def b_Coul_approx(E):
    return 1e-16 * (6.13 * n_0 * (1 + (1 / 75) * np.log(gamma(E) / n_0)))


def b_T_approx(E):
    b_IC = 1e-16 * (0.25 * E ** 2 * (1+z)**4)
    b_syn = 1e-16 * (0.0254 * E ** 2 * B_0 ** 2)
    b_brem = 1e-16 * (1.51 * n_0 * ( np.log(gamma(E) / n_0) + 0.36))
    
    
    # print('b_IC =', b_IC)
    # print('b_syn =', b_syn)
    # print('b_brem =', b_brem)
    # print('b_Coul =', b_Coul_approx(E))

    
    return b_IC + b_syn + b_brem + b_Coul_approx(E)


def approx_L(m_pbh, ep_energies, ep_spec):
    
    integrand_values = []
    for E in E_values:
    
        E_prime_values = 10 ** np.linspace(np.log10(E), np.log10(E_max), n_steps)
        spec_values = [spec(E_prime, ep_energies, ep_spec) for E_prime in E_prime_values]
        integrand_values.append(np.trapz(spec_values, E_prime_values) * b_Coul_approx(E) / b_T_approx(E))
        
    integral = np.trapz(integrand_values, E_values)
    
    return 2 * np.pi * rho_s * r_s * (R**2 - r_min**2) * integral / m_pbh

m_pbh, i = 5e14, 0

file_path_data = file_path_data_base + "LC21_{:.0f}/".format(i + 1)

ep_energies, ep_spec = read_blackhawk_spectra(
    file_path_data + "instantaneous_secondary_spectra.txt", col=2
)


print('Approximate L (r << r_s, r_c) [GeV/s] =', approx_L(m_pbh, ep_energies, ep_spec))
print('Full L [GeV/s] =', L(m_pbh, ep_energies, ep_spec))



m_pbh, i = 1e17, 24

file_path_data = file_path_data_base + "LC21_{:.0f}/".format(i + 1)

ep_energies, ep_spec = read_blackhawk_spectra(
    file_path_data + "instantaneous_secondary_spectra.txt", col=2
)


print('Approximate L (r << r_s, r_c) [GeV/s] =', approx_L(m_pbh, ep_energies, ep_spec))
print('Full L [GeV/s] =', L(m_pbh, ep_energies, ep_spec))


#%% Limit where spectrum is a delta function around the max energy

def heaviside(x):
    result = []
    for i in range(len(x)):
        if x[i] > 0:
            result.append(1)
        else:
            result.append(0)
    return np.array(result)

def L_integrand(E, r, E_peak, normalisation):
    return r**2 * heaviside(E_peak-E) * b_Coul(E, r) / b_T(E, r)

def L(m_pbh, E_peak, normalisation):

    if trapz:
        integrand = [np.trapz(L_integrand(E_values, r, E_peak, normalisation), E_values) for r in r_values]
        return 4 * np.pi * rho_s * r_s * normalisation * np.trapz(integrand, r_values) / m_pbh

    if scipy:
        return 4 * np.pi * rho_s * r_s * normalisation * np.array(dblquad(L_integrand, r_min, R, E_min, E_max, args=(m_pbh, ep_energies, ep_spec) )[0]) / m_pbh
    

for i, m_pbh in enumerate(m_pbh_values):

    file_path_data = file_path_data_base + "LC21_{:.0f}/".format(i + 1)

    ep_energies_load, ep_spec_load = read_blackhawk_spectra(
        file_path_data + "instantaneous_secondary_spectra.txt", col=2
    )
    
    ep_energies = ep_energies_load[ep_spec_load > 0]
    ep_spec = ep_spec_load[ep_spec_load > 0]
    
    print(max(ep_spec))
    print(ep_spec[np.argmax(ep_spec)])
    E_peak = ep_energies[np.argmax(ep_spec)]
    normalisation = np.trapz(ep_spec, ep_energies)

    print("M_PBH = {:.2e} g".format(m_pbh))

    # Evaluate photon spectrum at a set of pre-defined energies
    luminosity_predicted = L(m_pbh, E_peak, normalisation)
    f_pbh_values.append(L_0 / luminosity_predicted)


#%% Look at behaviour of b_C / b_T at large E

r_values = np.array([0.01, 0.1, 1]) * kpc_to_cm
E_values = 10**np.linspace(np.log10(E_min), np.log10(E_max), n_steps)
for r in r_values:
    plt.plot(E_values, b_Coul(E_values, r)/b_T(E_values, r), label=r / kpc_to_cm)

square_fit_pivot = b_Coul(E_max, r_values[-1])/b_T(E_max, r_values[-1])
square_fit = (E_values/E_values[-1])**(-2) * square_fit_pivot

PL_pivot = b_Coul(E_values[300], r_values[-1])/b_T(E_values[300], r_values[-1])
PL_fit = (E_values/E_values[300])**(-0.08) * PL_pivot


plt.plot(E_values, square_fit, linestyle='dashed', color='k')
plt.plot(E_values, PL_fit, linestyle='dotted', color='k')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('$E$ [GeV]')
plt.ylabel('$b_C / b_T$')
plt.legend(title='$r$ [kpc]')
plt.tight_layout()
plt.ylim(0.005, 1)
plt.xlim(E_min, E_max)
#%%
r_values = 10**np.linspace(np.log10(1e-10 * kpc_to_cm), np.log10(R), n_steps)
E_values = 10**np.linspace(np.log10(E_min), np.log10(E_max), n_steps)
[energies_mg, radii_mg] = np.meshgrid(E_values, r_values)

luminosity_grid = np.zeros(shape=(n_steps, n_steps))
for i in range(len(E_values)):
    for j in range(len(r_values)):
        luminosity_grid[i][j] = b_Coul(E_values[i], r_values[j]) / b_T(E_values[i], r_values[j])

fig = plt.figure()
ax = fig.gca(projection='3d')

# make 3D plot of integrand
surf = ax.plot_surface(energies_mg, radii_mg, 4*np.pi*luminosity_grid)
ax.set_xlabel('$E$ [GeV]', fontsize=14)
ax.set_ylabel('$r$ [kpc]', fontsize=14)
ax.set_zlabel('Luminosity integrand [$\mathrm{kpc}^{-1} \cdot \mathrm{s}^{-1}$]', fontsize=14)
ax.set_xscale('log')
ax.set_yscale('log')
plt.title('$b_C / b_T$', fontsize=14)

# make heat map
heatmap = plt.figure()
ax1 = heatmap.gca()
plt.pcolormesh(energies_mg, radii_mg, np.log10(1+4*np.pi*(luminosity_grid)), cmap='jet')
plt.xlabel('$E$ [GeV]')
plt.ylabel('$r$ [kpc]')
plt.xscale('log')
plt.yscale('log')
plt.title('$b_C / b_T$', fontsize=16)
plt.colorbar()
plt.tight_layout()


#%% Plot spectra
m_pbh_values = np.array([0.1, 1.0, 3, 10, 15]) * 10**16

plt.figure(figsize=(11, 8))

colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:purple', 'k']
max_y = 0

for i, m_pbh in enumerate(m_pbh_values):
    
    exponent = np.floor(np.log10(m_pbh))
    coefficient = m_pbh / 10**exponent
    file_path_data = "../blackhawk_v2.0/results/A22_Fig3_" + "{:.1f}e{:.0f}g/".format(coefficient, exponent)
    
    # Compute Hawking temperature of the BH (from Lee & Chan 2021 Eq. 2)
    T_BH = 1.06 * (1e13 / m_pbh)

    # Load electron secondary spectrum
    energies_secondary, secondary_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_secondary_spectra.txt", col=2)
    energies_primary, primary_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_primary_spectra.txt", col=7)
    
    plt.plot(energies_secondary, secondary_spectrum, label='{:.1e}'.format(m_pbh), color=colors[i])
    plt.plot(energies_primary, primary_spectrum, linestyle='dotted', color=colors[i])
    #plt.vlines(x=T_BH, ymin=min(secondary_spectrum), ymax=100*max(secondary_spectrum), color=colors[i], linestyle='dashed')
    max_y = max(max_y, max(secondary_spectrum))
    
plt.legend(title='$M_\mathrm{PBH}$ [g]')
plt.xlabel('$E$ [GeV]')
plt.ylabel('$\mathrm{d}^2 N_{e^\pm} / (\mathrm{d}t~\mathrm{d}E_{e^\pm})$ [s$^{-1}$ GeV$^{-1}$]')
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e18, 2*max_y)
plt.xlim(E_min, 5)
plt.tight_layout()
