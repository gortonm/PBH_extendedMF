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
    b_brem = 1e-16 * (1.51 * n(r) * (np.log(gamma(E) / n(r))) + 0.36)
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
    m_pbh_values = 10 ** np.linspace(np.log10(5e14), 17, 25)
    file_path_data_base = "../Downloads/version_finale/results/"


f_pbh_values = []

energies_ref = 10 ** np.linspace(np.log10(E_min), np.log10(E_max), n_steps)


def main():
    for i, m_pbh in enumerate(m_pbh_values):

        if i % 5 == 0:

            file_path_data = file_path_data_base + "LC21_{:.0f}/".format(i + 1)

            ep_energies, ep_spec = read_blackhawk_spectra(
                file_path_data + "instantaneous_secondary_spectra.txt", col=2
            )

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

    extracted_interpolated = 10 ** np.interp(
        np.log10(m_pbh_values),
        np.log10(m_pbh_LC21_extracted),
        np.log10(f_PBH_LC21_extracted),
    )
    extracted_interpolated_fewer = []
    m_pbh_fewer = []
    for i in range(0, len(extracted_interpolated)):
        if i % 5 == 0:
            extracted_interpolated_fewer.append(extracted_interpolated[i])
            m_pbh_fewer.append(m_pbh_values[i])
    ratio = extracted_interpolated_fewer / np.array(f_pbh_values)
    frac_diff = ratio - 1

    plt.figure()
    plt.plot(m_pbh_fewer, f_pbh_values)
    plt.plot(m_pbh_LC21_extracted, f_PBH_LC21_extracted, label="Extracted (Fig. 1)")
    plt.xlabel("$M_\mathrm{PBH}$ [g]")
    plt.ylabel("$f_\mathrm{PBH}$")
    plt.title("extension")
    plt.tight_layout()
    plt.legend()
    plt.ylim(1e-8, 1)
    plt.xlim(4e14, 1e17)
    plt.yscale("log")
    plt.xscale("log")

    plt.figure()
    plt.plot(m_pbh_fewer, ratio, "x")
    plt.xlabel("$M_\mathrm{PBH}$ [g]")
    plt.ylabel("$f_\mathrm{PBH, extracted} / f_\mathrm{PBH, calculated}$")
    plt.xscale("log")
    # plt.yscale('log')
    plt.xlim(
        4e14, 6e16
    )  # upper limit is where f_PBH = 1 in Fig. 1 of Lee & Chan (2021)
    plt.title(extension)
    plt.tight_layout()

    plt.figure()
    plt.plot(m_pbh_fewer, frac_diff, "x")
    plt.xlabel("$M_\mathrm{PBH}$ [g]")
    plt.ylabel("$(f_\mathrm{PBH, extracted} / f_\mathrm{PBH, calculated}) - 1$")
    plt.xscale("log")
    # plt.yscale('log')
    plt.xlim(
        4e14, 6e16
    )  # upper limit is where f_PBH = 1 in Fig. 1 of Lee & Chan (2021)
    plt.title(extension)
    plt.tight_layout()

    print("f_PBH =", f_pbh_values)
    print("ratio =", ratio)
    print("fractional difference =", frac_diff)