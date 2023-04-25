#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 14:55:07 2023

@author: ppxmg2
"""

import numpy as np


# Unit conversions
g_to_solar_mass = 1 / 1.989e33    # g to solar masses
pc_to_cm = 3.0857e18    # conversion factor from pc to cm
GeV_to_g = 1.783e-24    # conversion factor from GeV to grams

r_odot = 8.5 * 1000 * pc_to_cm   # galactocentric solar radius, in cm

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


def rho_0_gNFW(rho_odot, r_odot, r_s, gamma=1):
    """
    Calculate the characteristic halo density, for the NFW density profile.

    Parameters
    ----------
    rho_odot : Float
        Dark matter density at the solar galactocentric radius, in g/cm^3.
    r_odot : Float
        Galactocentric radius of the Sun, in cm.
    r_s : Float
        Scale radius, in cm.
    gamma : Float, optional
        Slope parameter. The default is 1.

    Returns
    -------
    Float
        Characteristic halo density, for the NFW density profile, in g/cm^3.

    """
    return rho_odot * np.power(r_s/r_odot, -gamma) * np.power(1 + r_odot/r_s, 3-gamma)


def rho_0_Einasto(rho_odot, r_odot, r_s, alpha):
    """
    Calculate the characteristic halo density, for the Einasto density profile.
    

    Parameters
    ----------
    rho_odot : Float
        Dark matter density at the solar galactocentric radius, in g/cm^3.
    r_odot : TYPE
        Galactocentric radius of the Sun, in cm.
    r_s : Float
        Scale radius, in cm.
    gamma : Float, optional
        Slope parameter.

    Returns
    -------
    Float
        Characteristic halo density, for the Einasto density profile, in g/cm^3.

    """
    return rho_odot * np.exp((2/alpha) * (np.power(r_odot/r_s, alpha) - 1))


def rho_gNFW(los, b, l, gamma=1):
    """
    Generalised NFW density profile.

    Parameters
    ----------
    los : Float
        Line of sight distance, in cm.
    b : Float
        Galactic latitude, in radians.
    l : Float
        Galactic longitude, in radians.
    gamma : Float
        Slope parameter. The default is 1.

    Returns
    -------
    Array-like
        Dark matter density, in g / cm^3.

    """
    r = find_r(los, b, l)
    return rho_0 * (r_s / r) * (1 + (r/r_s))**(-2)


def rho_Einasto(los, b, l, alpha):
    """
    Einasto density profile.

    Parameters
    ----------
    los : Float
        Line of sight distance, in cm.
    b : Float
        Galactic latitude, in radians.
    l : Float
        Galactic longitude, in radians.
    alpha : Float
        Slope parameter.

    Returns
    -------
    Array-like
        Dark matter density, in g / cm^3.

    """
    r = find_r(los, b, l)
    return rho_0 * np.exp(-(2/alpha)*(np.power(r/r_s, alpha) - 1))


def J_D(l_min, l_max, b_min, b_max, rho, params_rho=None):
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
    rho : Function
        DM density profile.
    params_rho : Array-like
        DM density profile parameters. The default is Nome.

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
                result += metric * rho(s[k], b[j], l[i], *params_rho)

    Delta = 0
    for i in range(0, nb_angles-1):
        for j in range(0, nb_angles-1):
            Delta += abs(np.cos(b[i])) * (l[i+1] - l[i]) * (b[j+1] - b[j])
    return result / Delta


#%%
# Load density profile parameters, from Table II of 1906.06133

B1 = True
B2 = not B1

COMPTEL = False
INTEGRAL = False
EGRET = False
FermiLAT = True
KP23 = False

if B1:
    filename = "./Data/DM_params_1906.06133_B1.txt"
    
elif B2:
    filename = "./Data/DM_params_1906.06133_B2.txt"
    
    
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
    
elif KP23:
    append = "KP23"
    b_max, l_max = np.radians(47.5), np.radians(47.5)


[rho_odot_mean, rho_odot_plus, rho_odot_minus, r200_mean, r200_plus, r200_minus, rs_mean, rs_plus, rs_minus, slope_mean, slope_plus, slope_minus] = np.genfromtxt(filename, delimiter="\t", skip_header=1, unpack=True)



#%%

for i in range(0, 3):
    
    rho_odot_values = np.array([rho_odot_mean[i], rho_odot_mean[i]+rho_odot_plus[i], rho_odot_mean[i]-rho_odot_minus[i]]) * GeV_to_g
    r200_values = np.array([r200_mean[i], r200_mean[i]+r200_plus[i], r200_mean[i]-r200_minus[i]]) * 1000 * pc_to_cm
    rs_values = np.array([rs_mean[i], rs_mean[i]+rs_plus[i], rs_mean[i]-rs_minus[i]]) * 1000 * pc_to_cm
    slope_values = np.array([slope_mean[i], slope_mean[i]+slope_plus[i], slope_mean[i]-slope_minus[i]])
    
    if i == 0 or i == 1:
        rho_0_values = rho_0_gNFW(rho_odot_values, r_odot, rs_values, slope_values)
    elif i == 2:
        rho_0_values = rho_0_Einasto(rho_odot_values, r_odot, rs_values, slope_values)
    
    for rho_0 in [rho_0_values[1], rho_0_values[2]]: # limited range to calculate minimum and maximum values of the J-factor

        for r_max in r200_values:
            
            for r_s in rs_values:
                
                for slope in slope_values:
                    
                    params_rho = [slope]
                    
                    if i == 0 or i == 1:
                        print(J_D(-l_max, l_max, -b_max, b_max, rho_gNFW, params_rho))
                        
                    elif i == 2:
                        print(J_D(-l_max, l_max, -b_max, b_max, rho_Einasto, params_rho))

#%% Test methods, using the central values of the parameters, and comparing to those calculated in Isatis.

r_s = 17 * 1000 * pc_to_cm    # scale radius, in cm
r_max = 200 * 1000 * pc_to_cm  # maximum radius of Galactic halo in cm
r_odot = 8.5 * 1000 * pc_to_cm   # galactocentric solar radius, in cm
rho_0 = 8.5e-25	    # characteristic halo density in g/cm^3
params_rho = [1]
print(J_D(-l_max, l_max, -b_max, b_max, rho_gNFW, params_rho))