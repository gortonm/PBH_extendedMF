#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 13:54:33 2022

@author: ppxmg2
"""

import numpy as np
import matplotlib as mpl

# Calculate constraints from Galactic Centre gamma rays, following the method
# used in the Isatis code by Jérémy Auffinger.

# Specify the plot style
mpl.rcParams.update({'font.size': 24,'font.family': 'serif'})
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

# Unit conversions
g_to_solar_mass = 1 / 1.989e33    # g to solar masses
pc_to_cm = 3.0857e18    # conversion factor from pc to cm

r_s = 17 * 1000 * pc_to_cm    # scale radius, in cm
r_odot = 8.5 * 1000 * pc_to_cm   # galactocentric solar radius, in cm

rho_0 = 8.5e-25	    # characteristic halo density in g/cm^3


def read_col(fname, first_row=0, col=1, convert=int, sep=None):
    """
    Read text files (e.g. from Blackhawk spectrum).

    Parameters
    ----------
    fname : String
        File name.
    first_row : Integer, optional
        Controls which row(s) to skip. The default is 0 (skips first row).
    col : Integer, optional
        Index of column to read.
    convert : Function, optional
        Function to convert column entry with. The default is int.
    sep : String, optional
        Column separator. The default is None.

    Returns
    -------
    data : Array-like
        Data from a column of a text file (e.g. BlackHawk spectrum).

    """    
    data = []
    
    with open(fname) as fobj:
        i=0
        for line in fobj:
            i += 1
            
            if i >= first_row:
                data.append(line.split(sep=sep)[col])
    return data    
     

def read_blackhawk_spectra(fname, col=1):
    """
    

    Parameters
    ----------
    fname : String
        File name.
    col : Integer, optional
        Index of column to read (i.e. which particle type to use), e.g.
        1 for photons, primary and secondary spectra.
        7 for electrons (primary spectra)
        The default is 1.

    Returns
    -------
    Array-like
        Energies and spectrum values calculate using BlackHawk.

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


def find_r(los, b, l):
    """
    Convert line of sight distance to galactocentric distance.

    Parameters
    ----------
    los : Float
        Line of sight distance, in cm.
    b : Float
        Galactic latitude, in radians.
    l : Float
        Galactic longitude, in radians.

    Returns
    -------
    Float
        Galactocentric distance, in cm.

    """
    return np.sqrt(r_odot**2 + los**2 - 2*r_odot*los*np.cos(b)*np.cos(l))


def rho_NFW(los, b, l):
    """
    NFW density profile.

    Parameters
    ----------
    los : Float
        Line of sight distance, in cm.
    b : Float
        Galactic latitude, in radians.
    l : Float
        Galactic longitude, in radians.

    Returns
    -------
    Array-like
        Dark matter density, in g / cm^3.

    """
    r = find_r(los, b, l)
    return rho_0 * (r_s / r) * (1 + (r/r_s))**(-2)


def refined_energies(energies, n_refined):
    """
    Find energies evenly-spaced in log space in the range of the instrument.

    Parameters
    ----------
    energies : Array-like
        Energies at which measured flux is calculated, in GeV.
    n_refined : Integer
        Number of (evenly-spaced) energies to compute.

    Returns
    -------
    ener_refined : Array-like
        Energies evenly spaced in log-space, in GeV.

    """
    E_min = min(energies)
    E_max = max(energies)
    step = (np.log10(10*E_max) - np.log10(E_min/10)) / (n_refined - 1)

    ener_refined = []
    for i in range(0, n_refined):
        ener_refined.append(10**(np.log10(E_min/10) + i*step))
    return ener_refined


def refined_flux(flux, ener_spec, n_refined):
    """
    Calculate flux at energies evenly-spaced in log-space.

    Parameters
    ----------
    flux : Array-like
        Flux measured by instrument.
    ener_spec : Array-like
        Energies at which flux is measured by instrument.
    n_refined : Integer
        Number of flux values to compute.

    Returns
    -------
    flux_refined : Array-like
        Flux, evaluated at energies evenly-spaced in log-space.

    """
    ener_refined = refined_energies(energies, n_refined)

    nb_spec = len(ener_spec)

    flux_refined = []

    c = 0
    for i in range(n_refined):
        while c < nb_spec and ener_spec[c] < ener_refined[i]:
            c += 1
        if c > 0 and c < nb_spec and flux[c-1] != 0:
            y = np.log10(flux[c-1]) + ((np.log10(flux[c]) - np.log10(flux[c-1])) / (np.log10(ener_spec[c]) - np.log10(ener_spec[c-1]))) * (np.log10(ener_refined[i]) - np.log10(ener_spec[c-1]))
            flux_refined.append(10**y)
        else:
            flux_refined.append(0)

    return flux_refined


def J_D(l_min, l_max, b_min, b_max):
    """
    Calculate J-factor.

    Parameters
    ----------
    l_min : Float
        Minimum galactic longitude, in radians.
    l_max : Float
        Maximum galactic longitude, in radians.
    b_min : Float
        Minimum galactic latitude, in radians.
    b_max : Float
        Maximum galactic latitude, in radians.

    Returns
    -------
    Float
        J-factor.

    """
    nb_angles = 100
    nb_radii = 100
    r_max = 200 * 1000 * pc_to_cm  # maximum radius of Galactic halo in cm

    b, l = [], []
    for i in range(0, nb_angles):
        l.append(l_min + i*(l_max - l_min)/(nb_angles - 1))
        b.append(b_min + i*(b_max - b_min)/(nb_angles - 1))

    result = 0
    for i in range(0, nb_angles-1):  # integral over l
        for j in range(0, nb_angles-1):  # integral over b
            s_max = r_odot * np.cos(l[i]) * np.cos(b[j]) + np.sqrt(r_max**2 - r_odot**2 * (1-(np.cos(l[i])*np.cos(b[j]))**2))
            s = []
            for k in range(0, nb_radii):
                s.append(0. + k * (s_max - 0.) / (nb_radii - 1))

            for k in range(0, nb_radii-1):  # integral over s(r(l, b))
                metric = abs(np.cos(b[i])) * (l[i+1] - l[i]) * (b[j+1] - b[j]) * (s[k+1] - s[k])
                result += metric * rho_NFW(s[k], b[j], l[i])

    Delta = 0
    for i in range(0, nb_angles-1):
        for j in range(0, nb_angles-1):
            Delta += abs(np.cos(b[i])) * (l[i+1] - l[i]) * (b[j+1] - b[j])
    return result / Delta


def galactic(spectrum, b_max, l_max):
    """
    Calculate photon flux from a population of PBHs.

    Parameters
    ----------
    spectrum : Array-like
        Flux measured by an instrument.
    b_max : Float
        Maximum galactic latitude, in radians.
    l_max : Float
        Maximum galactic longitude, in radians.

    Returns
    -------
    Array-like
        Photon flux from a population of PBHs, in each energy bin from an instrument.

    """
    n_spec = len(spectrum)

    # Calculate J-factor
    j_factor = J_D(-l_max, l_max, -b_max, b_max)

    print(j_factor)

    galactic = []
    for i in range(n_spec):
        val = j_factor * spectrum[i] / (4*np.pi*m_pbh)
        galactic.append(val)

    return np.array(galactic)

monochromatic_MF = True

if monochromatic_MF:
    filename_append = "_monochromatic"
    m_pbh_mono = np.logspace(11, 21, 1000)

f_PBH_isatis = []
file_path_data = "./../../Downloads/version_finale/scripts/Isatis/constraints/photons/"

COMPTEL = False
INTEGRAL = False
EGRET = False
FermiLAT = True

exclude_last_bin = False
save_each_bin = True

if COMPTEL:
    append = "COMPTEL_1107.0200"
    b_max, l_max = np.radians(15), np.radians(30)

elif INTEGRAL:
    append = "INTEGRAL_1107.0200"
    b_max, l_max = np.radians(15), np.radians(30)

elif EGRET:
    append = "EGRET_9811211"
    b_max, l_max = np.radians(5), np.radians(30)

elif FermiLAT:
    append = "Fermi-LAT_1101.1381"
    b_max, l_max = np.radians(10), np.radians(30)

energies, energies_minus, energies_plus, flux, flux_minus, flux_plus = np.genfromtxt("%sflux_%s.txt"%(file_path_data, append), skip_header = 6).transpose()[0:6]

if not exclude_last_bin:
    append = "all_bins_%s"%(append)
    
if save_each_bin:
    append = "full_%s"%(append)

# Number of interpolation points
n_refined = 500


for i, m_pbh in enumerate(m_pbh_mono):
    # Load photon spectra from BlackHawk outputs
    exponent = np.floor(np.log10(m_pbh))
    coefficient = m_pbh / 10**exponent

    if monochromatic_MF:
        file_path_BlackHawk_data = "./../../Downloads/version_finale/results/GC_mono_{:.0f}/".format(i+1)

    print("{:.1f}e{:.0f}g/".format(coefficient, exponent))

    ener_spec, spectrum = read_blackhawk_spectra(file_path_BlackHawk_data + "instantaneous_secondary_spectra.txt", col=1)

    flux_galactic = galactic(spectrum, b_max, l_max)
    ener_refined = refined_energies(energies, n_refined)
    flux_refined = refined_flux(flux_galactic, ener_spec, n_refined)

    def binned_flux(flux_refined, ener_refined, ener_inst, ener_inst_minus, ener_inst_plus):
        """
        Calculate theoretical flux from PBHs in each bin of an instrument.

        Parameters
        ----------
        flux_refined : Array-like
            Flux, evaluated at energies evenly-spaced in log-space.
        ener_refined : Array-like
            DESCRIPTION.
        ener_inst : Array-like
            Energies measured by instrument (middle values).
        ener_inst_minus : Array-like
            Energies measured by instrument (upper error bar).
        ener_inst_plus : Array-like
            Energies measured by instrument (lower error bar).

        Returns
        -------
        Array-like
            Flux from a population of PBHs, sorted into energy bins measured by an instrument.

        """
        flux_binned = []
        nb_refined = len(flux_refined)
        nb_inst = len(ener_inst)

        if exclude_last_bin:
            nb_inst = nb_inst - 1

        for i in range(nb_inst):
            val_binned = 0
            c = 0
            while c < nb_refined and ener_refined[c] < ener_inst[i] - ener_inst_minus[i]:
                c += 1
            if c > 0 and c+1 < nb_refined:
                while c < nb_refined and ener_refined[c] < ener_inst[i] + ener_inst_plus[i]:
                    val_binned += (ener_refined[c+1] - ener_refined[c]) * (flux_refined[c+1] + flux_refined[c]) / 2
                    c += 1
            flux_binned.append(val_binned)
        return np.array(flux_binned)

    # Calculate constraint on f_PBH
    if exclude_last_bin:
        f_PBH = min(flux[:-1] * (energies_plus[:-1] + energies_minus[:-1]) / binned_flux(flux_refined, ener_refined, energies, energies_minus, energies_plus))

    else:
        f_PBH = min(flux * (energies_plus + energies_minus) / binned_flux(flux_refined, ener_refined, energies, energies_minus, energies_plus))
        if save_each_bin:
            f_PBH = flux * (energies_plus + energies_minus) / binned_flux(flux_refined, ener_refined, energies, energies_minus, energies_plus)
        
    f_PBH_isatis.append(f_PBH)

# Save calculated results for f_PBH
np.savetxt("./Data/fPBH_GC_%s.txt"%(append+filename_append), f_PBH_isatis, delimiter="\t", fmt="%s")
