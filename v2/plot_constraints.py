#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:39:43 2023

@author: ppxmg2
"""

# Script plots results obtained for the extended PBH mass functions given by
# the fitting functions in 2009.03204.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from extended_MF_checks import envelope, load_results_Isatis
from preliminaries import load_data, m_max_SLN, LN, SLN, CC3, PL_MF

# Specify the plot style
mpl.rcParams.update({'font.size': 24, 'font.family':'serif'})
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
plt.style.use('tableau-colorblind10')


def frac_diff(y1, y2, x1, x2, interp_log = True):
    """
    Find the fractional difference between two arrays (y1, y2), evaluated
    at (x1, x2), of the form (y1/y2 - 1).
    
    In the calculation, interpolation (logarithmic or linear) is used to 
    evaluate the array y2 at x-axis values x1.

    Parameters
    ----------
    y1 : Array-like
        Array to find fractional difference of, evaluated at x1.
    y2 : Array-like
        Array to find fractional difference of, evaluated at x2.
    x1 : Array-like
        x-axis values that y1 is evaluated at.
    x2 : Array-like
        x-axis values that y2 is evaluated at.
    interp_log : Boolean, optional
        If True, use logarithmic interpolation to evaluate y1 at x2. The default is True.

    Returns
    -------
    Array-like
        Fractional difference between y1 and y2.

    """
    if interp_log:
        return y1 / 10**np.interp(np.log10(x1), np.log10(x2), np.log10(y2)) - 1
    else:
        return np.interp(x1, x2, y2) / y1 - 1


def load_data_GC_Isatis(Deltas, Delta_index, mf=None, params=None, evolved=True, exponent_PL_lower=2, approx=False):
    """
    Load extended MF constraints from Galactic Centre photons.

    Parameters
    ----------
    Deltas : Array-like
        Array of power-spectrum widths, which correspond to a given fitting function.
    Delta_index : Integer
        Index of the array Delta corresponding to the desired value of Delta.
    mf : Function, optional
        Fitting function to use. The default is None (delta-function).
    evolved : Boolean, optional
        If True, use the evolved form of the fitting function. The default is True.
    exponent_PL_lower : Float, optional
        Denotes the exponent of the power-law used to extrapolate the delta-function MF constraint. The default is 2.
    approx : Boolean, optional
        If True, load constraints obtained using f_max calculated from Isatis. Otherwise, load constraints calculated from the minimum constraint over each energy bin. The default is False.

    Returns
    -------
    mp_GC : Array-like
        Peak masses.
    f_PBH_GC : Array-like
        Constraint on f_PBH.

    """

    if mf == None:
        constraints_names_evap, f_PBHs_GC_delta = load_results_Isatis(modified=True)
        f_PBH_GC = envelope(f_PBHs_GC_delta)
        mp_GC = np.logspace(11, 21, 1000)
    
    else:
        constraints_names_short = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]

        if evolved:
            evolved_string = ""
            data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower)
        else:
            evolved_string = "_unevolved"
            data_folder = "./Data-tests/unevolved/PL_exp_{:.0f}".format(exponent_PL_lower)
        
        if mf==LN:
            mf_string = "LN"
        elif mf==SLN:
            mf_string = "SLN"
        elif mf==CC3:
            mf_string = "CC3"
        elif mf==PL_MF:
            mf_string = "PL"

            
        f_PBH_instrument = []
        
        for k in range(len(constraints_names_short)):
            # Load constraints for an evolved extended mass function obtained from each instrument
            
            if mf != PL_MF:
                if approx:
                    data_filename = data_folder + "/%s_GC_%s" % (mf_string, constraints_names_short[k]) + "_Carr_Delta={:.1f}".format(Deltas[Delta_index]) + "_approx%s.txt" % evolved_string
                else:
                    data_filename = data_folder + "/%s_GC_%s" % (mf_string, constraints_names_short[k]) + "_Carr_Delta={:.1f}".format(Deltas[Delta_index]) + "%s.txt" % evolved_string
            
            else:
                if approx:
                    data_filename = data_folder + "/%s_GC_%s" % (mf_string, constraints_names_short[k]) + "_Carr_approx%s.txt" % evolved_string
                else:
                    data_filename = data_folder + "/%s_GC_%s" % (mf_string, constraints_names_short[k]) + "_Carr%s.txt" % evolved_string
            
            mc_values, f_PBH_k = np.genfromtxt(data_filename, delimiter="\t")
    
            # Compile constraints from all instruments
            f_PBH_instrument.append(f_PBH_k)
     
        f_PBH_GC = envelope(f_PBH_instrument)
        
        if mf==LN:
            sigma_LN = params[0]
            mp_GC = mc_values * np.exp(-sigma_LN**2)
            print("\n data_filename [in load_data_GC_Isatis]")
            print(data_filename)
            print("approx = %s" % approx)

        elif mf==SLN:
            mp_GC = [m_max_SLN(m_c, *params, log_m_factor=3, n_steps=1000) for m_c in mc_values]

        elif mf==CC3 or mf == PL_MF:
            mp_GC = mc_values
            
    return mp_GC, f_PBH_GC

    
def load_data_KP23(Deltas, Delta_index, mf=None, evolved=True, exponent_PL_lower=2):
    """
    Load extended MF constraints from the delta-function MF constraints obtained by Korwar & Profumo (2023) [2302.04408].

    Parameters
    ----------
    Deltas : Array-like
        Array of power-spectrum widths, which correspond to a given fitting function.
    Delta_index : Integer
        Index of the array Delta corresponding to the desired value of Delta.
    mf : Function, optional
        Fitting function to use. The default is None (delta-function).
    evolved : Boolean, optional
        If True, use the evolved form of the fitting function. The default is True.
    exponent_PL_lower : Float, optional
        Denotes the exponent of the power-law used to extrapolate the delta-function MF constraint. The default is 2.

    Returns
    -------
    mp_KP23 : Array-like
        Peak masses.
    f_PBH_KP23 : Array-like
        Constraint on f_PBH.

    """
    
    # Path to extended MF constraints
    if evolved:
        data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower) 
    else:
        data_folder = "./Data-tests/unevolved/PL_exp_{:.0f}".format(exponent_PL_lower) 
    
    # Load data for the appropriate extended mass function (or delta-function MF):
    if mf == None:
        
        exponent_PL_upper = 2
        
        m_delta_values_loaded, f_max_loaded = load_data("./2302.04408/2302.04408_MW_diffuse_SPI.csv")
        
        m_delta_extrapolated_upper = np.logspace(15, 16, 11)
        m_delta_extrapolated_lower = np.logspace(11, 15, 41)
        
        f_max_extrapolated_upper = min(f_max_loaded) * np.power(m_delta_extrapolated_upper / min(m_delta_values_loaded), exponent_PL_upper)
        f_max_extrapolated_lower = min(f_max_extrapolated_upper) * np.power(m_delta_extrapolated_lower / min(m_delta_extrapolated_upper), exponent_PL_lower)
            
        f_PBH_KP23 = np.concatenate((f_max_extrapolated_lower, f_max_extrapolated_upper, f_max_loaded))
        mp_KP23 = np.concatenate((m_delta_extrapolated_lower, m_delta_extrapolated_upper, m_delta_values_loaded))

    elif mf == LN:
        data_filename = data_folder + "/LN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)
        mc_KP23, f_PBH_KP23 = np.genfromtxt(data_filename, delimiter="\t")
        mp_KP23 = mc_KP23 * np.exp(-sigmas_LN[Delta_index]**2)

    elif mf == SLN:
        data_filename = data_folder + "/SLN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)
        mc_KP23, f_PBH_KP23 = np.genfromtxt(data_filename, delimiter="\t")
        mp_KP23 = [m_max_SLN(m_c, sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index], log_m_factor=3, n_steps=1000) for m_c in mc_KP23]

    elif mf == CC3:
        data_filename = data_folder + "/CC3_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)
        mp_KP23, f_PBH_KP23 = np.genfromtxt(data_filename, delimiter="\t")

    elif mf == PL_MF:
        data_filename = data_folder + "/PL_2302.04408_Carr_extrapolated_exp{:.0f}.txt".format(exponent_PL_lower)
        mp_KP23, f_PBH_KP23 = np.genfromtxt(data_filename, delimiter="\t")

    return mp_KP23, f_PBH_KP23

    
def load_data_Voyager_BC19(Deltas, Delta_index, prop_A, with_bkg, mf=None, evolved=True, exponent_PL_lower=2):
    """
    Load extended MF constraints from the Voyager 1 delta-function MF constraints obtained by Boudaud & Cirelli (2019) [1807.03075].

    Parameters
    ----------
    Deltas : Array-like
        Array of power-spectrum widths, which correspond to a given fitting function.
    Delta_index : Integer
        Index of the array Delta corresponding to the desired value of Delta.
    prop_A : Boolean
        If True, load constraints obtained using propagation model prop A. If False, load constraints obtained using propagation model prop B.
    with_bkg : Boolean
        If True, load constraints obtained using background subtraction. If False, load constraints obtained without background subtraction.   
    mf : Function, optional
        Fitting function to use. The default is None (delta-function).
    evolved : Boolean, optional
        If True, use the evolved form of the fitting function. The default is True.
    exponent_PL_lower : Float, optional
        Denotes the exponent of the power-law used to extrapolate the delta-function MF constraint. The default is 2.

    Returns
    -------
    mp_BC19 : Array-like
        Peak masses.
    f_PBH_BC19 : Array-like
        Constraint on f_PBH.

    """
    
    if evolved:
        data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower)
    else:
        data_folder = "./Data-tests/unevolved/PL_exp_{:.0f}".format(exponent_PL_lower)
    
    if prop_A:
        prop_string = "prop_A"
    else:
        prop_string = "prop_B"
    if not with_bkg:
        prop_string += "_nobkg"
        
    if mf == None:
        if with_bkg:
            prop_string += "_bkg"
        mp_BC19, f_PBH_BC19 = load_data("1807.03075/1807.03075_" + prop_string + ".csv")
        
    elif mf == LN:
        data_filename = data_folder + "/LN_1807.03075_Carr_" + prop_string + "_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)
        mc_BC19, f_PBH_BC19 = np.genfromtxt(data_filename, delimiter="\t")
        mp_BC19 = mc_BC19 * np.exp(-sigmas_LN[Delta_index]**2)

    elif mf == SLN:
        data_filename = data_folder + "/SLN_1807.03075_Carr_" + prop_string + "_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)
        mc_BC19, f_PBH_BC19 = np.genfromtxt(data_filename, delimiter="\t")
        mp_BC19 = [m_max_SLN(m_c, sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index], log_m_factor=3, n_steps=1000) for m_c in mc_BC19]

    elif mf == CC3:
        data_filename = data_folder + "/CC3_1807.03075_Carr_" + prop_string + "_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)
        mp_BC19, f_PBH_BC19 = np.genfromtxt(data_filename, delimiter="\t")
 
    elif mf == PL_MF:
        data_filename = data_folder + "/PL_1807.03075_Carr_" + prop_string + "_extrapolated_exp{:.0f}.txt".format(exponent_PL_lower)
        mp_BC19, f_PBH_BC19 = np.genfromtxt(data_filename, delimiter="\t")
 
    return mp_BC19, f_PBH_BC19

    
def load_data_Subaru_Croon20(Deltas, Delta_index, mf=None):
    """
    Load extended MF constraints from the Subaru-HSC delta-function MF constraints obtained by Croon et al. (2020) [2007.12697].

    Parameters
    ----------
    Deltas : Array-like
        Array of power-spectrum widths, which correspond to a given fitting function.
    Delta_index : Integer
        Index of the array Delta corresponding to the desired value of Delta.
    mf : Function, optional
        Fitting function to use. The default is None (delta-function).

    Returns
    -------
    mp_Subaru : Array-like
        Peak masses.
    f_PBH_Subaru : Array-like
        Constraint on f_PBH.

    """
    
    if mf == None:
        mp_Subaru, f_PBH_Subaru = load_data("2007.12697/Subaru-HSC_2007.12697_dx=5.csv")

    elif mf == LN:
        mc_Subaru, f_PBH_Subaru = np.genfromtxt("./Data/LN_HSC_Carr_Delta={:.1f}.txt".format(Deltas[Delta_index]), delimiter="\t")
        mp_Subaru = mc_Subaru * np.exp(-sigmas_LN[Delta_index]**2)

    elif mf == SLN:
        mc_Subaru, f_PBH_Subaru = np.genfromtxt("./Data/SLN_HSC_Carr_Delta={:.1f}.txt".format(Deltas[Delta_index]), delimiter="\t")
        mp_Subaru = [m_max_SLN(m_c, sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index], log_m_factor=3, n_steps=1000) for m_c in mc_Subaru]

    elif mf == CC3:
        mp_Subaru, f_PBH_Subaru = np.genfromtxt("./Data/CC3_HSC_Carr_Delta={:.1f}.txt".format(Deltas[Delta_index]), delimiter="\t")
 
    elif mf == PL_MF:
        mp_Subaru, f_PBH_Subaru = np.genfromtxt("./Data/PL_HSC_Carr.txt", delimiter="\t")

    return mp_Subaru, f_PBH_Subaru


def load_data_GECCO(Deltas, Delta_index, mf=None, exponent_PL_lower=2, evolved=True, NFW=True):
    """
    Load extended MF constraints from the prospective GECCO delta-function MF constraints from Coogan et al. (2023) [2101.10370].

    Parameters
    ----------
    Deltas : Array-like
        Array of power-spectrum widths, which correspond to a given fitting function.
    Delta_index : Integer
        Index of the array Delta corresponding to the desired value of Delta.
    mf : Function, optional
        Fitting function to use. The default is None (delta-function).
    evolved : Boolean, optional
        If True, use the evolved form of the fitting function. The default is True.
    exponent_PL_lower : Float, optional
        Denotes the exponent of the power-law used to extrapolate the delta-function MF constraint. The default is 2.
    NFW : Boolean, optional
        If True, load constraints obtained using an NFW profile. If False, load constraints obtained using an Einasto profile.   

    Returns
    -------
    mp_GECCO : Array-like
        Peak masses.
    f_PBH_GECCO : Array-like
        Constraint on f_PBH.

    """
    
    if evolved:
        data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower)
    else:
        data_folder = "./Data-tests/unevolved/PL_exp_{:.0f}".format(exponent_PL_lower)
    
    if NFW:
        density_string = "NFW"
    else:
        density_string = "Einasto"

    if mf == None:
        mp_GECCO, f_PBH_GECCO = load_data("2101.01370/2101.01370_Fig9_GC_%s.csv" % density_string)

    elif mf == LN:
        mc_GECCO, f_PBH_GECCO = np.genfromtxt(data_folder + "/LN_2101.01370_Carr_Delta={:.1f}_".format(Deltas[Delta_index]) + "%s" % density_string + "_extrapolated_exp{:.0f}.txt".format(exponent_PL_lower))
        mp_GECCO = mc_GECCO * np.exp(-sigmas_LN[Delta_index]**2)

    elif mf == SLN:
        mc_GECCO, f_PBH_GECCO = np.genfromtxt(data_folder + "/SLN_2101.01370_Carr_Delta={:.1f}_".format(Deltas[Delta_index]) + "%s" % density_string + "_extrapolated_exp{:.0f}.txt".format(exponent_PL_lower))
        mp_GECCO = [m_max_SLN(m_c, sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index], log_m_factor=3, n_steps=1000) for m_c in mc_GECCO]

    elif mf == CC3:
        mp_GECCO, f_PBH_GECCO = np.genfromtxt(data_folder + "/CC3_2101.01370_Carr_Delta={:.1f}_".format(Deltas[Delta_index]) + "%s" % density_string + "_extrapolated_exp{:.0f}.txt".format(exponent_PL_lower))

    elif mf == PL_MF:
        mp_GECCO, f_PBH_GECCO = np.genfromtxt(data_folder + "/PL_2101.01370_Carr_%s" % density_string + "_extrapolated_exp{:.0f}.txt".format(exponent_PL_lower))
        
    return mp_GECCO, f_PBH_GECCO


def load_data_Sugiyama(Deltas, Delta_index, mf=None):
    """
    Load extended MF constraints from the prospective white dwarf microlensing delta-function MF constraints from Sugiyama et al. (2020) [1905.06066].

    Parameters
    ----------
    Deltas : Array-like
        Array of power-spectrum widths, which correspond to a given fitting function.
    Delta_index : Integer
        Index of the array Delta corresponding to the desired value of Delta.
    mf : Function, optional
        Fitting function to use. The default is None (delta-function).

    Returns
    -------
    mp_Subaru : Array-like
        Peak masses.
    f_PBH_Subaru : Array-like
        Constraint on f_PBH.

    """    

    if mf == None:
        mp_Sugiyama, f_PBH_Sugiyama = load_data("1905.06066/1905.06066_Fig8_finite+wave.csv")

    elif mf == LN:
        mc_Sugiyama, f_PBH_Sugiyama = np.genfromtxt("./Data/LN_Sugiyama20_Carr_Delta={:.1f}.txt".format(Deltas[Delta_index]), delimiter="\t") 
        mp_Sugiyama = mc_Sugiyama * np.exp(-sigmas_LN[Delta_index]**2)

    elif mf == SLN:
        mc_Sugiyama, f_PBH_Sugiyama = np.genfromtxt("./Data/SLN_Sugiyama20_Carr_Delta={:.1f}.txt".format(Deltas[Delta_index]), delimiter="\t") 
        mp_Sugiyama = [m_max_SLN(m_c, sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index], log_m_factor=3, n_steps=1000) for m_c in mc_Sugiyama]

    elif mf == CC3:
        mp_Sugiyama, f_PBH_Sugiyama = np.genfromtxt("./Data/CC3_Sugiyama20_Carr_Delta={:.1f}.txt".format(Deltas[Delta_index]), delimiter="\t") 
    
    elif mf == PL_MF:
        mp_Sugiyama, f_PBH_Sugiyama = np.genfromtxt("./Data/PL_Sugiyama20_Carr.txt", delimiter="\t")        
    
    return mp_Sugiyama, f_PBH_Sugiyama

    
def find_label(mf=None):
    """
    Generate a label for the fitting function that a constraint is calculated from.

    Parameters
    ----------
    mf : Function.
        Fitting function to use. The default is None (delta-function).

    Returns
    -------
    label : String
        Label denoting the fitting function used.

    """
    
    if mf == None:
        label="Delta function"
    elif mf == LN:
        label="LN"
    elif mf == SLN:
        label = "SLN"
    elif mf == CC3:
        label = "CC3"
    elif mf == PL_MF:
        label = "PL"
    return label


def set_ticks_grid(ax):
    """
    Set x-axis and y-axis ticks, and make grid.
    Follows https://stackoverflow.com/questions/30887920/how-to-show-minor-tick-labels-on-log-scale-with-matplotlib

    Parameters
    ----------
    ax : Matplotlib Axes object
        Axis to add ticks and grid to.

    Returns
    -------
    None.

    """
    
    x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
    ax.xaxis.set_major_locator(x_major)
    x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 5)
    ax.xaxis.set_minor_locator(x_minor)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    y_major = mpl.ticker.LogLocator(base = 10.0, numticks = 10)
    ax.yaxis.set_major_locator(y_major)
    y_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax.yaxis.set_minor_locator(y_minor)
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    
    ax.grid()


def plotter_GC_Isatis(Deltas, Delta_index, ax, color, mf=None, params=None, exponent_PL_lower=2, evolved=True, approx=True, show_label=False, linestyle="solid", linewidth=1, marker=None):
    """
    Plot extended MF constraints from Galactic Centre photons.    

    Parameters
    ----------
    Deltas : Array-like
        Array of power-spectrum widths, which correspond to a given fitting function.
    Delta_index : Integer
        Index of the array Delta corresponding to the desired value of Delta.
    ax : Matplotlib Axes object
        Axis to add ticks and grid to.
    color : String
        Color to use for plotting.
    mf : Function, optional
        Fitting function to use. The default is None (delta-function).
    exponent_PL_lower : Float, optional
        Denotes the exponent of the power-law used to extrapolate the delta-function MF. The default is 2.
    evolved : Boolean, optional
        If True, use the evolved form of the fitting function. The default is True.
    approx : Boolean, optional
        If True, plot constraints obtained using f_max calculated from Isatis. Otherwise, plot constraints calculated from the minimum constraint over each energy bin. The default is True.
    show_label : Boolean, optional
        If True, add a label denoting the fitting function used. The default is False.
    linestyle : String, optional
        Linestyle to use for pplotting. The default is "solid".
    linewidth : Float, optional
        Line width to use for pplotting. The default is 1.

    Returns
    -------
    None.

    """
    
    mp, f_PBH = load_data_GC_Isatis(Deltas, Delta_index, mf, params, evolved, exponent_PL_lower, approx)
    
    if not evolved:
        alpha=0.4
    else:
        alpha=1
    
    if show_label:
        label = find_label(mf)
        ax.plot(mp, f_PBH, color=color, linestyle=linestyle, label=label, alpha=alpha, marker=marker)
    else:
        ax.plot(mp, f_PBH, color=color, linestyle=linestyle, alpha=alpha, marker=marker)


def plotter_BC19(Deltas, Delta_index, ax, color, prop_A, with_bkg, mf=None, exponent_PL_lower=2, evolved=True, show_label=False, linestyle="solid", linewidth=1, marker=None):
    """
    Plot extended MF constraints from from the Voyager 1 delta-function MF constraints obtained by Boudaud & Cirelli (2019) [1807.03075].    

    Parameters
    ----------
    Deltas : Array-like
        Array of power-spectrum widths, which correspond to a given fitting function.
    Delta_index : Integer
        Index of the array Delta corresponding to the desired value of Delta.
    ax : Matplotlib Axes object
        Axis to add ticks and grid to.
    color : String
        Color to use for plotting.
    prop_A : Boolean
        If True, load constraints obtained using propagation model prop A. If False, load constraints obtained using propagation model prop B.
    with_bkg : Boolean
        If True, load constraints obtained using background subtraction. If False, load constraints obtained without background subtraction.   
    mf : Function, optional
        Fitting function to use. The default is None (delta-function).
    exponent_PL_lower : Float, optional
        Denotes the exponent of the power-law used to extrapolate the delta-function MF. The default is 2.
    evolved : Boolean, optional
        If True, use the evolved form of the fitting function. The default is True.
    show_label : Boolean, optional
        If True, add a label denoting the fitting function used. The default is False.
    linestyle : String, optional
        Linestyle to use for pplotting. The default is "solid".
    linewidth : Float, optional
        Line width to use for pplotting. The default is 1.
    marker : String, optional
        Marker to use for plotting. The default is None.

    Returns
    -------
    None.

    """

    mp, f_PBH = load_data_Voyager_BC19(Deltas, Delta_index, prop_A, with_bkg, mf, evolved, exponent_PL_lower)
    
    if not evolved:
        alpha=0.4
    else:
        alpha=1
    
    if show_label:
        label = find_label(mf)
        ax.plot(mp, f_PBH, color=color, linestyle=linestyle, label=label, alpha=alpha, marker=marker)
    else:
        ax.plot(mp, f_PBH, color=color, linestyle=linestyle, alpha=alpha, marker=marker)


def plotter_KP23(Deltas, Delta_index, ax, color, mf=None, exponent_PL_lower=2, evolved=True, show_label=False, linestyle="solid", linewidth=1, marker=None):
    """
    Plot extended MF constraints from the delta-function MF constraints obtained by Korwar & Profumo (2023) [2302.04408].    

    Parameters
    ----------
    Deltas : Array-like
        Array of power-spectrum widths, which correspond to a given fitting function.
    Delta_index : Integer
        Index of the array Delta corresponding to the desired value of Delta.
    ax : Matplotlib Axes object
        Axis to add ticks and grid to.
    color : String
        Color to use for plotting.
    mf : Function, optional
        Fitting function to use. The default is None (delta-function).
    exponent_PL_lower : Float, optional
        Denotes the exponent of the power-law used to extrapolate the delta-function MF. The default is 2.
    evolved : Boolean, optional
        If True, use the evolved form of the fitting function. The default is True.
    show_label : Boolean, optional
        If True, add a label denoting the fitting function used. The default is False.
    linestyle : String, optional
        Linestyle to use for pplotting. The default is "solid".
    linewidth : Float, optional
        Line width to use for pplotting. The default is 1.
    marker : String, optional
        Marker to use for plotting. The default is None.

    Returns
    -------
    None.

    """    
    
    mp, f_PBH = load_data_KP23(Deltas, Delta_index, mf, evolved, exponent_PL_lower)
    
    if not evolved:
        alpha=0.4
    else:
        alpha=1
    
    if show_label:
        label = find_label(mf)
        ax.plot(mp, f_PBH, color=color, linestyle=linestyle, label=label, alpha=alpha, marker=marker)
    else:
        ax.plot(mp, f_PBH, color=color, linestyle=linestyle, alpha=alpha, marker=marker)

    
def plotter_Subaru_Croon20(Deltas, Delta_index, ax, color, mf=None, show_label=True, linestyle="solid", linewidth=1, marker=None):
    """
    Plot extended MF constraints from the Subaru-HSC delta-function MF constraints obtained by Croon et al. (2020) [2007.12697].    

    Parameters
    ----------
    Deltas : Array-like
        Array of power-spectrum widths, which correspond to a given fitting function.
    Delta_index : Integer
        Index of the array Delta corresponding to the desired value of Delta.
    ax : Matplotlib Axes object
        Axis to add ticks and grid to.
    color : String
        Color to use for plotting.
    mf : Function, optional
        Fitting function to use. The default is None (delta-function).
    show_label : Boolean, optional
        If True, add a label denoting the fitting function used. The default is False.
    linestyle : String, optional
        Linestyle to use for pplotting. The default is "solid".
    linewidth : Float, optional
        Line width to use for pplotting. The default is 1.
    marker : String, optional
        Marker to use for plotting. The default is None.

    Returns
    -------
    None.

    """    
    
    mp_Subaru, f_PBH_Subaru = load_data_Subaru_Croon20(Deltas, Delta_index, mf)
    
    if show_label:
        label = find_label(mf)
        ax.plot(mp_Subaru, f_PBH_Subaru, color=color, linestyle=linestyle, label=label, marker=marker)
    else:
        ax.plot(mp_Subaru, f_PBH_Subaru, color=color, linestyle=linestyle, marker=marker)
        
        
def plotter_GECCO(Deltas, Delta_index, ax, color, mf=None, exponent_PL_lower=2, evolved=True, NFW=True, show_label=False, linestyle="solid", linewidth=1, marker=None):
    """
    Plot extended MF constraints from the prospective GECCO delta-function MF constraints from Coogan et al. (2023) [2101.10370].    

    Parameters
    ----------
    Deltas : Array-like
        Array of power-spectrum widths, which correspond to a given fitting function.
    Delta_index : Integer
        Index of the array Delta corresponding to the desired value of Delta.
    ax : Matplotlib Axes object
        Axis to add ticks and grid to.
    color : String
        Color to use for plotting.
    mf : Function, optional
        Fitting function to use. The default is None (delta-function).
    exponent_PL_lower : Float, optional
        Denotes the exponent of the power-law used to extrapolate the delta-function MF. The default is 2.
    evolved : Boolean, optional
        If True, use the evolved form of the fitting function. The default is True.
    NFW : Boolean, optional
        If True, load constraints obtained using an NFW profile. If False, load constraints obtained using an Einasto profile.   
    show_label : Boolean, optional
        If True, add a label denoting the fitting function used. The default is False.
    linestyle : String, optional
        Linestyle to use for pplotting. The default is "solid".
    linewidth : Float, optional
        Line width to use for pplotting. The default is 1.
    marker : String, optional
        Marker to use for plotting. The default is None.

    Returns
    -------
    None.

    """    
    
    mp_GECCO, f_PBH_GECCO = load_data_GECCO(Deltas, Delta_index, mf, exponent_PL_lower=exponent_PL_lower, evolved=evolved, NFW=NFW)
    
    if show_label:
        label = find_label(mf)
        ax.plot(mp_GECCO, f_PBH_GECCO, color=color, linestyle=linestyle, label=label, marker=marker)
    else:
        ax.plot(mp_GECCO, f_PBH_GECCO, color=color, linestyle=linestyle, marker=marker)


def plotter_Sugiyama(Deltas, Delta_index, ax, color, mf=None, show_label=True, linestyle="solid", linewidth=1, marker=None):
    """
    Plot extended MF constraints from the prospective white dwarf microlensing delta-function MF constraints from Sugiyama et al. (2020) [1905.06066].

    Parameters
    ----------
    Deltas : Array-like
        Array of power-spectrum widths, which correspond to a given fitting function.
    Delta_index : Integer
        Index of the array Delta corresponding to the desired value of Delta.
    ax : Matplotlib Axes object
        Axis to add ticks and grid to.
    color : String
        Color to use for plotting.
    mf : Function, optional
        Fitting function to use. The default is None (delta-function).
    show_label : Boolean, optional
        If True, add a label denoting the fitting function used. The default is False.
    linestyle : String, optional
        Linestyle to use for pplotting. The default is "solid".
    linewidth : Float, optional
        Line width to use for pplotting. The default is 1.
    marker : String, optional
        Marker to use for plotting. The default is None.

    Returns
    -------
    None.

    """    

    mp_Sugiyama, f_PBH_Sugiyama = load_data_Sugiyama(Deltas, Delta_index, mf)
    
    if show_label:
        label = find_label(mf)
        ax.plot(mp_Sugiyama, f_PBH_Sugiyama, color=color, linestyle=linestyle, label=label, marker=marker)
    else:
        ax.plot(mp_Sugiyama, f_PBH_Sugiyama, color=color, linestyle=linestyle, marker=marker)


#%% Tests of the method frac_diff:
    
if "__main__" == __name__:
    
    # Constant fractional difference
    y1, y2 = 1.5*np.ones(10), np.ones(10)
    x1 = x2 = np.linspace(0, 10, 10)
    print(frac_diff(y1, y2, x1, x2))
    
    y1, y2 = 1.5*np.ones(10), np.ones(30)
    x1 = np.linspace(0, 10, 10)
    x2 = np.linspace(-10, 20, 30)
    print(frac_diff(y1, y2, x1, x2))
    
    # x-dependent fractional difference
    x1 = x2 = np.linspace(1, 10, 10)
    y1 = 1 + x1**2
    y2 = np.ones(10)
    print(frac_diff(y1, y2, x1, x2))
       

#%% Existing constraints

if "__main__" == __name__:
    
    # If True, plot the evaporation constraints used by Isatis (from COMPTEL, INTEGRAL, EGRET and Fermi-LAT)
    plot_GC_Isatis = False
    # If True, plot the evaporation constraints shown in Korwar & Profumo (2023) [2302.04408]
    plot_KP23 = True
    # If True, plot the evaporation constraints from Boudaud & Cirelli (2019) [1807.03075]
    plot_BC19 = False
    # If True, plot unevolved MF constraint
    plot_unevolved = False
    # If True, plot the fractional difference between evolved and unevolved MF results
    plot_fracdiff = True
    
    # Choose colors to match those from Fig. 5 of 2009.03204
    colors = ['silver', 'tab:red', 'tab:blue', 'k', 'k']
    
    # Parameters used for convergence tests in Galactic Centre constraints.
    cutoff = 1e-4
    delta_log_m = 1e-3
    E_number = 500    
    
    if E_number < 1e3:
        energies_string = "E{:.0f}".format(E_number)
    else:
        energies_string = "E{:.0f}".format(np.log10(E_number))
            
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
        
    # Load Subaru-HSC delta-function MF constraint
    m_delta_Subaru, f_PBH_delta_Subaru = load_data("2007.12697/Subaru-HSC_2007.12697_dx=5.csv")
    
    for i in range(len(Deltas)):
                        
        fig, ax = plt.subplots(figsize=(9, 5))
        
        if plot_GC_Isatis:

            # If required, plot unevolved MF constraints.
            if plot_unevolved:
                
                plotter_GC_Isatis(Deltas, i, ax, mf=LN, evolved=False, color=colors[1])
                plotter_GC_Isatis(Deltas, i, ax, mf=SLN, evolved=False, color=colors[2])
                plotter_GC_Isatis(Deltas, i, ax, mf=CC3, evolved=False, color=colors[3])

            #plt.suptitle("Existing constraints (showing Galactic Centre photon constraints (Isatis)), $\Delta={:.1f}$".format(Deltas[i]), fontsize="small")
            
            ax.set_xlabel("$m_p~[\mathrm{g}]$")                
            ax.set_ylabel("$f_\mathrm{PBH}$")
            ax.set_xscale("log")
            ax.set_yscale("log")

            plotter_GC_Isatis(Deltas, i, ax, mf=None, color=colors[0])
            plotter_GC_Isatis(Deltas, i, ax, mf=LN, color=colors[1], linestyle=(0, (5, 1)))
            plotter_GC_Isatis(Deltas, i, ax, mf=SLN, color=colors[2], linestyle=(0, (5, 7)))
            plotter_GC_Isatis(Deltas, i, ax, mf=CC3, color=colors[3], linestyle="dashed")

            
            if plot_unevolved and plot_fracdiff:
                fig1, ax1a = plt.subplots(figsize=(6,6))
                
                mp_LN_evolved, f_PBH_LN_evolved = load_data_GC_Isatis(Deltas, i, mf=LN, evolved=True)
                mp_SLN_evolved, f_PBH_SLN_evolved = load_data_GC_Isatis(Deltas, i, mf=SLN, evolved=True)
                mp_CC3_evolved, f_PBH_CC3_evolved = load_data_GC_Isatis(Deltas, i, mf=CC3, evolved=True)
               
                mp_LN_unevolved, f_PBH_LN_unevolved = load_data_GC_Isatis(Deltas, i, mf=LN, evolved=False)
                mp_SLN_unevolved, f_PBH_SLN_unevolved = load_data_GC_Isatis(Deltas, i, mf=SLN, evolved=False)
                mp_CC3_unevolved, f_PBH_CC3_unevolved = load_data_GC_Isatis(Deltas, i, mf=CC3, evolved=False)
                
                ax1a.plot(mp_LN_evolved, np.abs(frac_diff(f_PBH_LN_unevolved, f_PBH_LN_evolved, mp_LN_unevolved, mp_LN_evolved)), label="LN", color="r")
                ax1a.plot(mp_SLN_evolved, np.abs(frac_diff(f_PBH_SLN_unevolved, f_PBH_SLN_evolved, mp_SLN_unevolved, mp_SLN_evolved)), label="SLN", color="b")
                ax1a.plot(mp_CC3_evolved, np.abs(frac_diff(f_PBH_CC3_unevolved, f_PBH_CC3_evolved, mp_CC3_unevolved, mp_CC3_evolved)), label="CC3", color="g")
                ax1a.set_ylabel("$|\Delta f_\mathrm{PBH} / f_\mathrm{PBH}|$")
                ax1a.set_xlabel("$m_p~[\mathrm{g}]$")
                ax1a.set_xscale("log")
                ax1a.set_yscale("log")
                ax1a.set_title("$\Delta={:.1f}$".format(Deltas[i]))
                ax1a.legend(title="Unevolved/evolved - 1", fontsize="x-small")
                ax1a.set_xlim(xmin=1e16)
                ax1a.set_ylim(ymax=1e2)
                ax1a.grid()
                fig1.tight_layout()
            
            
        elif plot_KP23:
            ax.set_xlabel("$m_p~[\mathrm{g}]$")
            #fig.suptitle("Existing constraints (showing Korwar \& Profumo 2023 constraints), $\Delta={:.1f}$".format(Deltas[i]), fontsize="small")

            ax.set_ylabel("$f_\mathrm{PBH}$")
            ax.set_xscale("log")
            ax.set_yscale("log")
            
            plotter_KP23(Deltas, i, ax, color=colors[0], linestyle="solid", linewidth=2)
            plotter_KP23(Deltas, i, ax, color=colors[1], mf=LN, linestyle=(0, (5, 1)))
            plotter_KP23(Deltas, i, ax, color=colors[2], mf=SLN, linestyle=(0, (5, 7)))
            plotter_KP23(Deltas, i, ax, color=colors[3], mf=CC3, linestyle="dashed")
            
            
            # If required, plot the fractional difference from the delta-function MF constraint
            if plot_fracdiff:
                
                m_delta, f_PBH_delta =  load_data_KP23(Deltas, i, evolved=True)
                mp_LN, f_PBH_LN = load_data_KP23(Deltas, i, mf=LN, evolved=True)
                mp_SLN, f_PBH_SLN = load_data_KP23(Deltas, i, mf=SLN, evolved=True)
                mp_CC3, f_PBH_CC3 = load_data_KP23(Deltas, i, mf=CC3, evolved=True)
               
                fig1, ax1a = plt.subplots(figsize=(6,6))
                ax1a.plot(mp_LN, frac_diff(f_PBH_LN, f_PBH_delta, mp_LN, m_delta), label="LN", color="r")
                ax1a.plot(mp_SLN, frac_diff(f_PBH_SLN, f_PBH_delta, mp_SLN, m_delta), label="SLN", color="b")
                ax1a.plot(mp_CC3, frac_diff(f_PBH_CC3, f_PBH_delta, mp_CC3, m_delta), label="CC3", color="g")
                ax1a.set_ylabel("$\Delta f_\mathrm{PBH} / f_\mathrm{PBH}$")
                ax1a.set_xlabel("$m_p~[\mathrm{g}]$")
                ax1a.set_xscale("log")
                #ax1a.set_yscale("log")
                ax1a.set_title("$\Delta={:.1f}$".format(Deltas[i]))
                ax1a.legend(title="Delta func./extended MF - 1", fontsize="x-small")
                ax1a.set_xlim(xmin=5e14)
                ax1a.set_ylim(-0.1, 0.1)
                ax1a.grid()
                fig1.tight_layout()
            
            # If required, plot constraints obtained with unevolved MF
            if plot_unevolved:
                plotter_KP23(Deltas, i, ax, color=colors[0], linestyle="solid", linewidth=2)
                plotter_KP23(Deltas, i, ax, color=colors[1], mf=LN, evolved=False)
                plotter_KP23(Deltas, i, ax, color=colors[2], mf=SLN, evolved=False)
                plotter_KP23(Deltas, i, ax, color=colors[3], mf=CC3, evolved=False)

                
        elif plot_BC19:
            
            """
            # Boolean determines which propagation model to load data from
            for prop_A in [True, False]:
                prop_B = not prop_A
            
                # Boolean determines whether to load constraint obtained with a background or without a background
                for with_bkg in [True, False]:

                    if not with_bkg:
                        linestyle = "dashed"

                    else:
                        linestyle = "dotted"                                                    
                                       
                    plotter_BC19(Deltas, i, ax, colors[0], prop_A, with_bkg, mf=None, linestyle=linestyle)
                    plotter_BC19(Deltas, i, ax, colors[1], prop_A, with_bkg, mf=LN, linestyle=linestyle)
                    plotter_BC19(Deltas, i, ax, colors[2], prop_A, with_bkg, mf=SLN, linestyle=linestyle)
                    plotter_BC19(Deltas, i, ax, colors[3], prop_A, with_bkg, mf=CC3, linestyle=linestyle)
                """
            #plt.suptitle("Existing constraints (showing Voyager 1 constraints), $\Delta={:.1f}$".format(Deltas[i]), fontsize="small")
            
            # For Delta = 5, tightest constraint comes from the Prop A model with background subtraction
            prop_A = False
            with_bkg = True

            plotter_BC19(Deltas, i, ax, colors[0], prop_A, with_bkg)
            plotter_BC19(Deltas, i, ax, colors[1], prop_A, with_bkg, mf=LN, linestyle=(0, (5, 1)))
            plotter_BC19(Deltas, i, ax, colors[2], prop_A, with_bkg, mf=SLN, linestyle=(0, (5, 7)))
            plotter_BC19(Deltas, i, ax, colors[3], prop_A, with_bkg, mf=CC3, linestyle="dashed")

            ax.set_xlabel("$m_p~[\mathrm{g}]$")
            ax.set_ylabel("$f_\mathrm{PBH}$")
            ax.set_xscale("log")
            ax.set_yscale("log")

        if Deltas[i] < 5:
            show_label_Subaru = False
        else:
            show_label_Subaru = True
        # Plot Subaru-HSC constraints        
        plotter_Subaru_Croon20(Deltas, i, ax, color=colors[0], linestyle="solid", linewidth=2, show_label=show_label_Subaru)
        plotter_Subaru_Croon20(Deltas, i, ax, color=colors[1], mf=LN,  linestyle=(0, (5, 1)), show_label=show_label_Subaru)
        plotter_Subaru_Croon20(Deltas, i, ax, color=colors[2], mf=SLN, linestyle=(0, (5, 7)), show_label=show_label_Subaru)
        plotter_Subaru_Croon20(Deltas, i, ax, color=colors[3], mf=CC3, linestyle="dashed", show_label=show_label_Subaru)

        set_ticks_grid(ax)
        
        # Set axis limits
        if Deltas[i] < 5:
            xmin_evap, xmax_evap = 1e16, 2.5e17
            xmin_HSC, xmax_HSC = 1e21, 1e29
            
            if plot_KP23:
                xmin_evap, xmax_evap = 1e16, 7e17
                
            elif plot_BC19:
                xmin_evap, xmax_evap = 1e16, 3e17
        else:
            xmin_evap, xmax_evap = 1e16, 7e17
            xmin_HSC, xmax_HSC = 9e18, 1e29

            if plot_KP23:
                xmin_evap, xmax_evap = 1e16, 2e18
                
            elif plot_BC19:
                xmin_evap, xmax_evap = 1e16, 1e19
                
        ymin, ymax = 1e-3, 1

        ax.set_xlim(xmin_evap, 1e24)
        ax.set_ylim(ymin, ymax)
       
        ax.legend(fontsize="xx-small", title="$\Delta={:.0f}$".format(Deltas[i]), loc="upper right")
        fig.tight_layout()
        
        if plot_GC_Isatis:
            fig.savefig("./Results/Figures/fPBH_Delta={:.1f}_existing_GC_Isatis.pdf".format(Deltas[i]))
            fig.savefig("./Results/Figures/fPBH_Delta={:.1f}_existing_GC_Isatis.png".format(Deltas[i]))
            
        elif plot_KP23:
            fig.savefig("./Results/Figures/fPBH_Delta={:.1f}_existing_KP23.pdf".format(Deltas[i]))
            fig.savefig("./Results/Figures/fPBH_Delta={:.1f}_existing_KP23.png".format(Deltas[i]))
        
        elif plot_BC19:
            fig.savefig("./Results/Figures/fPBH_Delta={:.1f}_existing_BC19.pdf".format(Deltas[i]))
            fig.savefig("./Results/Figures/fPBH_Delta={:.1f}_existing_BC19.png".format(Deltas[i]))
        
                
#%% Prospective constraints

if "__main__" == __name__:
        
    # Choose colors to match those from Fig. 5 of 2009.03204
    colors = ['silver', 'tab:red', 'tab:blue', 'k', 'k']
                    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
        
    for i in range(len(Deltas)):
                        
        fig, ax = plt.subplots(figsize=(9, 5))
        # Plot prospective extended MF constraints from the white dwarf microlensing survey proposed in Sugiyama et al. (2020) [1905.06066].
        
        ax.set_xlabel("$m_p~[\mathrm{g}]$")
        ax.set_ylabel("$f_\mathrm{PBH}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        
        NFW = False
        
        # Set axis limits
        if Deltas[i] < 5:
            xmin_evap, xmax_evap = 1e16, 2e18
            xmin_micro, xmax_micro = 2e20, 5e23
            show_label = False
        else:
            xmin_evap, xmax_evap = 1e16, 5e18
            xmin_micro, xmax_micro = 2e17, 5e23
            show_label = True

        # plot Einasto profile results            
        plotter_GECCO(Deltas, i, ax, color=colors[0], NFW=NFW, linestyle="solid", linewidth=2)
        plotter_GECCO(Deltas, i, ax, color=colors[1], NFW=NFW, mf=LN, linestyle=(0, (5, 1)))
        plotter_GECCO(Deltas, i, ax, color=colors[2], NFW=NFW, mf=SLN, linestyle=(0, (5, 7)))
        plotter_GECCO(Deltas, i, ax, color=colors[3], NFW=NFW, mf=CC3, linestyle="dashed")

        plotter_Sugiyama(Deltas, i, ax, color=colors[0], linestyle="solid", linewidth=2, show_label=show_label)
        plotter_Sugiyama(Deltas, i, ax, color=colors[1], mf=LN, linestyle=(0, (5, 1)), show_label=show_label)
        plotter_Sugiyama(Deltas, i, ax, color=colors[2], mf=SLN, linestyle=(0, (5, 7)), show_label=show_label)
        plotter_Sugiyama(Deltas, i, ax, color=colors[3], mf=CC3, linestyle="dashed", show_label=show_label)

        set_ticks_grid(ax)
        ymin, ymax = 1e-3, 1

        ax.set_xlim(xmin_evap, xmax_micro)
        ax.set_ylim(ymin, ymax)
        ax.legend(fontsize="xx-small", title="$\Delta={:.0f}$".format(Deltas[i]), loc="upper right")
        
        #plt.suptitle("Prospective constraints, $\Delta={:.1f}$".format(Deltas[i]), fontsize="small")
        fig.tight_layout()
        fig.savefig("./Results/Figures/fPBH_Delta={:.1f}_prospective.pdf".format(Deltas[i]))
        fig.savefig("./Results/Figures/fPBH_Delta={:.1f}_prospective.png".format(Deltas[i]))
            

#%% Plot constraints for different Delta on the same plot

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
            
    exponent_PL_lower = 2
    data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower)
    
    plot_LN = False
    plot_SLN = True
    plot_CC3 = False
    
    plot_unevolved = True
    
    fig, ax = plt.subplots(figsize=(6,6))
    fig1, ax1 = plt.subplots(figsize=(6,6))
       
    # Delta-function MF constraints
    
    # Power-law exponent to use between 1e11g and 1e15g.
    exponent_PL_lower = 2.0
    m_delta, f_PBH_delta = load_data_KP23(Deltas, i=0, evolved=True, exponent_PL_lower=exponent_PL_lower)
    ax.plot(m_delta, f_PBH_delta, color="tab:gray", label="Delta func.", linewidth=2)

    colors=["tab:blue", "tab:orange", "tab:green", "tab:red"]
        
    for i, Delta_index in enumerate([0, 5, 6]):
    #for i, Delta_index in enumerate([1, 2, 3, 4]):
        
        data_filename_LN = data_folder + "/LN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)
        data_filename_SLN = data_folder + "/SLN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)
        data_filename_CC3 = data_folder + "/CC3_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)
    
               
        if plot_LN:
            mp_LN, f_PBH_LN = load_data_KP23(Deltas, Delta_index, mf=LN, evolved=True)
            ax.plot(mp_LN, f_PBH_LN, color=colors[i], dashes=[6, 2], label="{:.1f}".format(Deltas[Delta_index]))
            ax1.plot(mp_LN, np.abs(frac_diff(f_PBH_LN, f_PBH_delta, mp_LN, m_delta)), color=colors[i], label="{:.1f}".format(Deltas[Delta_index]))
            
        elif plot_SLN:
            mp_SLN, f_PBH_SLN = load_data_KP23(Deltas, Delta_index, mf=SLN, evolved=True)
            ax.plot(mp_SLN, f_PBH_SLN, color=colors[i], linestyle=(0, (5, 7)), label="{:.1f}".format(Deltas[Delta_index]))
            ax1.plot(mp_SLN, np.abs(frac_diff(f_PBH_SLN, f_PBH_delta, mp_SLN, m_delta)), color=colors[i], label="{:.1f}".format(Deltas[Delta_index]))

        elif plot_CC3:
            mp_CC3, f_PBH_CC3 = load_data_KP23(Deltas, Delta_index, mf=CC3, evolved=True)
            ax.plot(mp_CC3, f_PBH_CC3, color=colors[i], linestyle="dashed", label="{:.1f}".format(Deltas[Delta_index]))
            ax1.plot(mp_CC3, np.abs(frac_diff(f_PBH_CC3, f_PBH_delta, mp_CC3, m_delta)), color=colors[i], label="{:.1f}".format(Deltas[Delta_index]))
           
        # Plot constraint obtained with unevolved MF
        if plot_unevolved:
            
            if plot_LN:
                plotter_KP23(Deltas, Delta_index, ax, color=colors[i], mf=LN, evolved=False, exponent_PL_lower=exponent_PL_lower)                
            elif plot_SLN:
                plotter_KP23(Deltas, Delta_index, ax, color=colors[i], mf=SLN, evolved=False, exponent_PL_lower=exponent_PL_lower)
            elif plot_CC3:
                plotter_KP23(Deltas, Delta_index, ax, color=colors[i], mf=CC3, evolved=False, exponent_PL_lower=exponent_PL_lower)

    ax.set_ylabel("$f_\mathrm{PBH}$")
    ax1.set_ylabel("$|f_\mathrm{PBH} / f_\mathrm{max} - 1|$")
    
    for a in [ax, ax1]:
        a.set_xlabel("$m_p~[\mathrm{g}]$")
        a.legend(title="$\Delta$", fontsize="x-small")
        a.set_xscale("log")
        a.set_yscale("log")
        
    ax.set_xlim(1e16, 2e18)
    ax.set_ylim(1e-5, 1)
    ax1.set_xlim(1e15, max(m_delta))
    ax1.set_ylim(1e-2, 2)
    
    for f in [fig, fig1]:
        if plot_LN:
            f.suptitle("Korwar \& Profumo 2023 constraints (LN)", fontsize="small")
        elif plot_SLN:
            f.suptitle("Korwar \& Profumo 2023 constraints (SLN)", fontsize="small")
        elif plot_CC3:
            f.suptitle("Korwar \& Profumo 2023 constraints (CC3)", fontsize="small")
        f.tight_layout()
 
    x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
    ax1.xaxis.set_major_locator(x_major)
    x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 5)
    ax1.xaxis.set_minor_locator(x_minor)
    ax1.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())


#%% Plot constraints for different Delta on the same plot

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
            
    exponent_PL_lower = 2
    data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower)
            
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, tight_layout = {'pad': 0}, figsize=(10,15))
       
    # Linestyles for different constraints
    linestyles = ["dashdot", "dashed", "dotted"]
       
    # Delta-function MF constraints
    m_delta_GC = np.logspace(11, 21, 1000)
    constraints_names_GC, f_PBHs_GC_delta = load_results_Isatis(modified=True)
    f_PBH_delta_GC = envelope(f_PBHs_GC_delta)
    m_delta_Subaru, f_PBH_delta_Subaru = load_data("2007.12697/Subaru-HSC_2007.12697_dx=5.csv")

    m_delta_values_loaded, f_max_loaded = load_data("2302.04408/2302.04408_MW_diffuse_SPI.csv")
    # Power-law exponent to use between 1e15g and 1e16g.
    exponent_PL_upper = 2.0
    # Power-law exponent to use between 1e11g and 1e15g.
    exponent_PL_lower = 2.0
    m_delta_extrapolated_upper = np.logspace(15, 16, 11)
    m_delta_extrapolated_lower = np.logspace(11, 15, 41)
    f_max_extrapolated_upper = min(f_max_loaded) * np.power(m_delta_extrapolated_upper / min(m_delta_values_loaded), exponent_PL_upper)
    f_max_extrapolated_lower = min(f_max_extrapolated_upper) * np.power(m_delta_extrapolated_lower / min(m_delta_extrapolated_upper), exponent_PL_lower)
    m_pbh_values_upper = np.concatenate((m_delta_extrapolated_upper, m_delta_values_loaded))
    f_max_upper = np.concatenate((f_max_extrapolated_upper, f_max_loaded))
    f_PBH_delta_KP23 = np.concatenate((f_max_extrapolated_lower, f_max_extrapolated_upper, f_max_loaded))
    m_delta_KP23 = np.concatenate((m_delta_extrapolated_lower, m_delta_extrapolated_upper, m_delta_values_loaded))

    colors_LN = ["k", "tab:red", "pink"]
    colors_SLN = ["k", "tab:blue", "deepskyblue"]
    colors_CC3 = ["k", "tab:green", "lime"]
    
    # Opacities and line widths for different Delta:
    linewidth_values = [2, 1.5, 1]
    
    for ax in [ax0, ax1, ax2]:
        ax.plot(0, 0, color="tab:gray", linewidth=2, label="$\delta$ func.")
        ax.plot(m_delta_Subaru, f_PBH_delta_Subaru, color="tab:gray", linewidth=2, linestyle=linestyles[0])
        ax.plot(m_delta_GC, f_PBH_delta_GC, color="tab:gray", linewidth=2, linestyle=linestyles[1])
        ax.plot(m_delta_KP23, f_PBH_delta_KP23, color="tab:gray", linewidth=2, linestyle=linestyles[2])

    
    for i, Delta_index in enumerate([0, 5, 6]):
        
        # Loading constraints from Subaru-HSC
        mc_Subaru_SLN, f_PBH_Subaru_SLN = np.genfromtxt("./Data/SLN_HSC_Carr_Delta={:.1f}.txt".format(Deltas[Delta_index]), delimiter="\t")
        mp_Subaru_CC3, f_PBH_Subaru_CC3 = np.genfromtxt("./Data/CC3_HSC_Carr_Delta={:.1f}.txt".format(Deltas[Delta_index]), delimiter="\t")
        mc_Subaru_LN, f_PBH_Subaru_LN = np.genfromtxt("./Data/LN_HSC_Carr_Delta={:.1f}.txt".format(Deltas[Delta_index]), delimiter="\t")
        
        # Galactic Centre photon constraints, from Isatis
        mc_values_GC = np.logspace(14, 20, 120)

        exponent_PL_lower = 2
        constraints_names_short = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
        data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower)
        
        f_PBH_instrument_LN = []
        f_PBH_instrument_SLN = []
        f_PBH_instrument_CC3 = []

        for k in range(len(constraints_names_short)):
            # Load constraints for an evolved extended mass function obtained from each instrument
            data_filename_LN = data_folder + "/LN_GC_%s" % constraints_names_short[k] + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[Delta_index])
            data_filename_SLN = data_folder + "/SLN_GC_%s" % constraints_names_short[k]  + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[Delta_index])
            data_filename_CC3 = data_folder + "/CC3_GC_%s" % constraints_names_short[k]  + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[Delta_index])
                
            mc_LN_evolved, f_PBH_LN_evolved = np.genfromtxt(data_filename_LN, delimiter="\t")
            mc_SLN_evolved, f_PBH_SLN_evolved = np.genfromtxt(data_filename_SLN, delimiter="\t")
            mp_CC3_evolved, f_PBH_CC3_evolved = np.genfromtxt(data_filename_CC3, delimiter="\t")
            
            # Compile constraints from all instruments
            f_PBH_instrument_LN.append(f_PBH_LN_evolved)
            f_PBH_instrument_SLN.append(f_PBH_SLN_evolved)
            f_PBH_instrument_CC3.append(f_PBH_CC3_evolved)
 
        f_PBH_GC_LN = envelope(f_PBH_instrument_LN)
        f_PBH_GC_SLN = envelope(f_PBH_instrument_SLN)
        f_PBH_GC_CC3 = envelope(f_PBH_instrument_CC3)

        # Korwar & Profumo (2023) constraints
        data_filename_LN = data_folder + "/LN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)
        data_filename_SLN = data_folder + "/SLN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)
        data_filename_CC3 = data_folder + "/CC3_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)
        mc_KP23_LN, f_PBH_KP23_LN = np.genfromtxt(data_filename_LN, delimiter="\t")
        mc_KP23_SLN, f_PBH_KP23_SLN = np.genfromtxt(data_filename_SLN, delimiter="\t")
        mp_KP23_CC3, f_PBH_KP23_CC3 = np.genfromtxt(data_filename_CC3, delimiter="\t")

        # Plot LN MF results
        mp_KP23_LN = mc_KP23_LN * np.exp(-sigmas_LN[Delta_index]**2)
        mp_GC_LN = mc_values_GC * np.exp(-sigmas_LN[Delta_index]**2)
        mp_Subaru_LN = mc_Subaru_SLN * np.exp(-sigmas_LN[Delta_index]**2) 
        ax0.plot(mp_Subaru_LN, f_PBH_Subaru_LN, color=colors_LN[i], linestyle=linestyles[0], linewidth=linewidth_values[i])
        ax0.plot(mp_GC_LN, f_PBH_GC_LN, color=colors_LN[i], linestyle=linestyles[1], linewidth=linewidth_values[i])
        ax0.plot(mp_KP23_LN, f_PBH_KP23_LN, color=colors_LN[i], linestyle=linestyles[2], linewidth=linewidth_values[i])
        ax0.plot(0,0, color=colors_LN[i], label="$\Delta={:.0f}$".format(Deltas[Delta_index]))
        ax0a = ax0.twinx()
        ax0a.legend(title="LN", loc="upper right", fontsize="x-small")
        ax0a.axis("off")
        
        # Plot SLN MF results
        mp_KP23_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index], log_m_factor=3, n_steps=1000) for m_c in mc_KP23_SLN]
        mp_GC_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index], log_m_factor=3, n_steps=1000) for m_c in mc_values_GC]
        mp_Subaru_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index], log_m_factor=3, n_steps=1000) for m_c in mc_Subaru_SLN]
        ax1.plot(mp_Subaru_SLN, f_PBH_Subaru_SLN, color=colors_SLN[i], linestyle=linestyles[0], linewidth=linewidth_values[i])
        ax1.plot(mp_GC_SLN, f_PBH_GC_SLN, color=colors_SLN[i], linestyle=linestyles[1], linewidth=linewidth_values[i])
        ax1.plot(mp_KP23_SLN, f_PBH_KP23_SLN, color=colors_SLN[i], linestyle=linestyles[2], linewidth=linewidth_values[i])
        ax1.plot(0,0, color=colors_SLN[i], label="$\Delta={:.0f}$".format(Deltas[Delta_index]))
        ax1a = ax1.twinx()
        ax1a.legend(title="SLN", loc="upper right", fontsize="x-small")
        ax1a.axis("off")
        
        # Plot CC3 MF results
        mp_GC_CC3 = mc_values_GC
        ax2.plot(mp_Subaru_CC3, f_PBH_Subaru_CC3, color=colors_CC3[i], linestyle=linestyles[0], linewidth=linewidth_values[i])
        ax2.plot(mp_GC_CC3, f_PBH_GC_CC3, color=colors_CC3[i], linestyle=linestyles[1], linewidth=linewidth_values[i])
        ax2.plot(mp_KP23_CC3, f_PBH_KP23_CC3, color=colors_CC3[i], linestyle=linestyles[2], linewidth=linewidth_values[i])
        ax2.plot(0,0, color=colors_CC3[i], label="$\Delta={:.0f}$".format(Deltas[Delta_index]))
        ax2a = ax2.twinx()
        ax2a.legend(title="CC3", loc="upper right", fontsize="x-small")
        ax2a.axis("off")
        
    for ax in[ax0, ax1, ax2]:
        ax.set_ylabel("$f_\mathrm{PBH}$")

        ax.set_xscale("log")
        ax.set_yscale("log")  
        ax.set_xlim(1e16, 1e24)
        ax.set_ylim(1e-3, 1)
        ax.legend(fontsize="x-small")
            
        x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 10)
        ax.xaxis.set_major_locator(x_major)
        x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
        ax.xaxis.set_minor_locator(x_minor)
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.grid()
        
    ax2.set_xlabel("$m_p~[\mathrm{g}]$")
    #fig.tight_layout()
 
    
#%% Plot the most stringent GC photon constraint (calculated using the method from 1705.05567 with the delta-function MF constraint from Isatis).

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    colors_evap = ["tab:orange", "tab:green", "tab:red", "tab:blue"]
    constraints_names_short = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
    
    # if True, use the constraints obtained for evolved MFs. Otherwise, use constraints obtained using the unevolved MFs.
    evolved = False
    
    # Parameters used for convergence tests in Galactic Centre constraints.
    cutoff = 1e-4
    delta_log_m = 1e-3
    E_number = 500    
    
    if E_number < 1e3:
        energies_string = "E{:.0f}".format(E_number)
    else:
        energies_string = "E{:.0f}".format(np.log10(E_number))
    
    plot_LN = False
    plot_SLN = True
    plot_CC3 = False
        
    for i in range(len(Deltas)):
        
        fig, ax = plt.subplots(figsize=(8, 8))
        # Load constraints from Galactic Centre photons.
        
        exponent_PL_lower = 2
        constraints_names_short = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
        if evolved:
            data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower)
        else:
            data_folder = "./Data-tests/unevolved/PL_exp_{:.0f}".format(exponent_PL_lower)
        f_PBH_instrument_LN = []
        f_PBH_instrument_SLN = []
        f_PBH_instrument_CC3 = []

        for k in range(len(constraints_names_short)):
            # Load constraints for an evolved extended mass function obtained from each instrument
            if evolved:
                data_filename_LN = data_folder + "/LN_GC_%s" % constraints_names_short[k] + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[i])
                data_filename_SLN = data_folder + "/SLN_GC_%s" % constraints_names_short[k]  + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[i])
                data_filename_CC3 = data_folder + "/CC3_GC_%s" % constraints_names_short[k]  + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[i])
            else:
                data_filename_LN = data_folder + "/LN_GC_%s" % constraints_names_short[k] + "_Carr_Delta={:.1f}_approx_unevolved.txt".format(Deltas[i])
                data_filename_SLN = data_folder + "/SLN_GC_%s" % constraints_names_short[k]  + "_Carr_Delta={:.1f}_approx_unevolved.txt".format(Deltas[i])
                data_filename_CC3 = data_folder + "/CC3_GC_%s" % constraints_names_short[k]  + "_Carr_Delta={:.1f}_approx_unevolved.txt".format(Deltas[i])

            mc_LN_evolved, f_PBH_LN_evolved = np.genfromtxt(data_filename_LN, delimiter="\t")
            mc_SLN_evolved, f_PBH_SLN_evolved = np.genfromtxt(data_filename_SLN, delimiter="\t")
            mp_CC3_evolved, f_PBH_CC3_evolved = np.genfromtxt(data_filename_CC3, delimiter="\t")
            
            # Estimate peak mass of skew-lognormal MF
            mp_SLN_evap = [m_max_SLN(m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i], log_m_factor=3, n_steps=1000) for m_c in mc_SLN_evolved]
            
            if plot_LN:
                ax.plot(mc_LN_evolved * np.exp(-sigmas_LN[i]**2), f_PBH_LN_evolved, color=colors_evap[k], marker="x", label=constraints_names_short[k])
            elif plot_SLN:
                ax.plot(mp_SLN_evap, f_PBH_SLN_evolved, color=colors_evap[k], marker="x", label=constraints_names_short[k])
            elif plot_CC3:
                ax.plot(mp_CC3_evolved, f_PBH_CC3_evolved, color=colors_evap[k], marker="x", label=constraints_names_short[k])

        ax.set_xlim(1e16, 1e18)
        ax.set_ylim(10**(-6), 1)
        ax.set_xlabel("$m_p~[\mathrm{g}]$")
        ax.set_ylabel("$f_\mathrm{PBH}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize="x-small")
        fig.tight_layout()
        fig.suptitle("$\Delta={:.1f}$".format(Deltas[i]))

        
#%% Plot the GC photon, Korwar & Profumo (2023) constraints and Boudaud & Cirelli (2019) constraints on the same plot

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
                
    plot_LN = False
    plot_SLN = False
    plot_CC3 = True
    
    exponent_PL_lower = 3
    data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower)
    
    # For Boudaud & Cirelli (2019) constraints
    BC19_colours = ["b", "r"]
    linestyles = ["solid", "dashed"]
    
    prop_A = True
    with_bkg = False
    
    # For Galactic Centre photon constraints
    constraints_names_short = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
    mc_values_evap = np.logspace(14, 20, 120)


    for Delta_index in range(len(Deltas)):
        
        fig, ax = plt.subplots(figsize=(8, 8))
                
        for colour_index, prop in enumerate([True, True]):
            
            prop_B = not prop_A
            
            for linestyle_index, with_bkg in enumerate([False, True]):
            
                label=""
                
                if prop_A:
                    prop_string = "prop_A"
                    label = "Prop A"
        
                elif prop_B:
                    prop_string = "prop_B"
                    label = "Prop B"
                    
                if not with_bkg:
                    prop_string += "_nobkg"
                    label += " w/o bkg subtraction "
                else:
                    label += " w/ bkg subtraction"
                    
                data_filename_LN = data_folder + "/LN_1807.03075_Carr_" + prop_string + "_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)
                data_filename_SLN = data_folder + "/SLN_1807.03075_Carr_" + prop_string + "_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)
                data_filename_CC3 = data_folder + "/CC3_1807.03075_Carr_" + prop_string + "_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)
                
                mc_LN_evolved, f_PBH_LN_evolved = np.genfromtxt(data_filename_LN, delimiter="\t")
                mc_SLN_evolved, f_PBH_SLN_evolved = np.genfromtxt(data_filename_SLN, delimiter="\t")
                mp_CC3_evolved, f_PBH_CC3_evolved = np.genfromtxt(data_filename_CC3, delimiter="\t")
                
                if plot_LN:
                    mp_LN = mc_LN_evolved * np.exp(-sigmas_LN[Delta_index]**2)
                    ax.plot(mp_LN, f_PBH_LN_evolved, color=BC19_colours[colour_index], linestyle=linestyles[linestyle_index], label=label)
                elif plot_SLN:
                    mp_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index], log_m_factor=3, n_steps=1000) for m_c in mc_SLN_evolved]
                    ax.plot(mp_SLN, f_PBH_SLN_evolved, color=BC19_colours[colour_index], linestyle=linestyles[linestyle_index], label=label)
                elif plot_CC3:
                    ax.plot(mp_CC3_evolved, f_PBH_CC3_evolved, color=BC19_colours[colour_index], linestyle=linestyles[linestyle_index], label="BC '19 (" + label + ")")
                    
                with_bkg = not with_bkg
                   
            prop_A = not prop_A

        data_filename_LN = data_folder + "/LN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)
        data_filename_SLN = data_folder + "/SLN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)
        data_filename_CC3 = data_folder + "/CC3_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)
    
        mc_KP23_LN, f_PBH_KP23_LN = np.genfromtxt(data_filename_LN, delimiter="\t")
        mc_KP23_SLN, f_PBH_KP23_SLN = np.genfromtxt(data_filename_SLN, delimiter="\t")
        mp_KP23_CC3, f_PBH_KP23_CC3 = np.genfromtxt(data_filename_CC3, delimiter="\t")
            
        """
        # Load constraints from Galactic Centre photons.
        f_PBH_instrument_LN = []
        f_PBH_instrument_SLN = []
        f_PBH_instrument_CC3 = []

        for k in range(len(constraints_names_short)):
            # Load constraints for an evolved extended mass function obtained from each instrument
            data_filename_LN = data_folder + "/LN_GC_%s" % constraints_names_short[k] + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[Delta_index])
            data_filename_SLN = data_folder + "/SLN_GC_%s" % constraints_names_short[k]  + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[Delta_index])
            data_filename_CC3 = data_folder + "/CC3_GC_%s" % constraints_names_short[k]  + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[Delta_index])
                
            mc_GC_LN, f_PBH_LN_evolved = np.genfromtxt(data_filename_LN, delimiter="\t")
            mc_GC_SLN, f_PBH_SLN_evolved = np.genfromtxt(data_filename_SLN, delimiter="\t")
            mp_GC_CC3, f_PBH_CC3_evolved = np.genfromtxt(data_filename_CC3, delimiter="\t")
            
            # Compile constraints from all instruments
            f_PBH_instrument_LN.append(f_PBH_LN_evolved)
            f_PBH_instrument_SLN.append(f_PBH_SLN_evolved)
            f_PBH_instrument_CC3.append(f_PBH_CC3_evolved)
 
        f_PBH_GC_LN = envelope(f_PBH_instrument_LN)
        f_PBH_GC_SLN = envelope(f_PBH_instrument_SLN)
        f_PBH_GC_CC3 = envelope(f_PBH_instrument_CC3)
        """
        if plot_LN:
            ax.plot(mc_KP23_LN * np.exp(-sigmas_LN[Delta_index]**2), f_PBH_KP23_LN, color="k", label="KP '23")
            #ax.plot(mc_values_evap * np.exp(-sigmas_LN[Delta_index]**2), f_PBH_GC_LN, color="tab:grey", label="GC photons")
                
        elif plot_SLN:
            # Estimate peak mass of skew-lognormal MF
            mp_KP23_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index], log_m_factor=3, n_steps=1000) for m_c in mc_KP23_SLN]
            ax.plot(mp_KP23_SLN, f_PBH_KP23_SLN, color="k", label="KP '23")
            #mp_GC_SLN = [m_max_SLN(m_c, sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index], log_m_factor=3, n_steps=1000) for m_c in mc_values_evap]
            #ax.plot(mp_GC_SLN, f_PBH_GC_SLN, color="tab:grey", label="GC photons")
        
        elif plot_CC3:
            ax.plot(mc_values_evap, f_PBH_KP23_CC3, color="k", label="KP '23")
            #ax.plot(mp_GC_CC3, f_PBH_GC_CC3, color="tab:grey", label="GC photons")

        ax.set_xlim(1e16, 5e18)
        ax.set_ylim(10**(-6), 1)
        ax.set_xlabel("$m_p~[\mathrm{g}]$")
        ax.set_ylabel("$f_\mathrm{PBH}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize="x-small")
        fig.tight_layout()
        fig.suptitle("$\Delta={:.1f}$".format(Deltas[Delta_index]))
    
        
#%% Plot extended MF constraints shown in Fig. 20 of 2002.12778 on the same axes as in that figure.
from preliminaries import constraint_Carr, LN


def Solmass_to_g(m):
    """Convert a mass m (in solar masses) to grams."""
    return 1.989e33 * m


def g_to_Solmass(m):
    """Convert a mass m (in grams) to solar masses."""
    return m / 1.989e33
    

def f_PBH_beta_prime(m_values, beta_prime):
    """
    Calcualte f_PBH from the initial PBH fraction beta_prime, using Eq. 57 of 2002.12778.

    Parameters
    ----------
    m_values : Array-like
        PBH masses, in grams.
    beta_prime : Array-like / float
        Scaled fraction of the energy density of the Universe in PBHs at their formation time (see Eq. 8 of 2002.12778), at each mass in m_values.

    Returns
    -------
    Array-like
        Value of f_PBH evaluated at each mass in m_values.

    """
    return 3.81e8 * beta_prime * np.power(m_values / 1.989e33, -1/2)
 
    
def beta_prime_gamma_rays(m_values, epsilon=0.4):
    """
    Calculate values of beta prime allowed from extragalactic gamma-rays, using the simple power-law expressions in 2002.12778.

    Parameters
    ----------
    m_values : Array-like
        PBH masses, in grams.
    epsilon : Float, optional
        Parameter describing the power-law dependence of x-ray and gamma-ray spectra on photon energy. The default is 0.4.

    Returns
    -------
    Array-like
        Scaled fraction of the energy density of the Universe in PBHs at their formation time (see Eq. 8 of 2002.12778), at each mass in m_values.

    """
    beta_prime_values = []
    
    for m in m_values:
        if m < m_star:
            beta_prime_values.append(5e-28 * np.power(m/m_star, -5/2-2*epsilon))
        else:
            beta_prime_values.append(5e-26 * np.power(m/m_star, 7/2+epsilon))
    
    return np.array(beta_prime_values)


if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
            
    # Power-law exponent to use between 1e15g and 1e16g.
    exponent_PL_upper = 2.0
    # Power-law exponent to use between 1e11g and 1e15g.
    exponent_PL_lower = 2.0
    
    data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower)
    
    plot_unevolved = True
    plot_against_mc = False
    
    fig, ax = plt.subplots(figsize=(7,6))
       
    
    # Delta-function MF constraints from Korwar & Profumo (2023)
    m_delta_KP23_loaded, f_max_KP23_loaded = load_data("2302.04408/2302.04408_MW_diffuse_SPI.csv")
        
    m_delta_extrapolated_upper = np.logspace(15, 16, 11)
    m_delta_extrapolated_lower = np.logspace(11, 15, 41)
    
    f_max_extrapolated_upper = min(f_max_KP23_loaded) * np.power(m_delta_extrapolated_upper / min(m_delta_KP23_loaded), exponent_PL_upper)
    f_max_extrapolated_lower = min(f_max_extrapolated_upper) * np.power(m_delta_extrapolated_lower / min(m_delta_extrapolated_upper), exponent_PL_lower)

    m_delta_Carr21_upper = np.concatenate((m_delta_extrapolated_upper, m_delta_KP23_loaded))
    f_max_upper = np.concatenate((f_max_extrapolated_upper, f_max_KP23_loaded))
    
    f_max_KP23 = np.concatenate((f_max_extrapolated_lower, f_max_extrapolated_upper, f_max_KP23_loaded))
    m_delta_KP23 = np.concatenate((m_delta_extrapolated_lower, m_delta_extrapolated_upper, m_delta_KP23_loaded))
    
    
    # Width of log-normal MF used in Carr et al. (2021)
    sigma_Carr21 = 2
    # Range of characteristic masses for obtaining constraints.
    mc_Carr21 = np.logspace(14, 22, 1000)

    colors=["tab:blue", "tab:orange"]
    
    ax.plot(m_delta_KP23, f_max_KP23, color="tab:gray", label="$\delta$ func.", linewidth=2)
    ax1 = ax.secondary_xaxis('top', functions=(g_to_Solmass, Solmass_to_g)) 
    
    
    # Calculate extended MF constraints obtained for a log-normal with the delta-function MF constraint from Korwar & Profumo (2023).
    for i, sigma in enumerate([sigma_Carr21]):
        
        f_PBH_evolved = constraint_Carr(mc_Carr21, m_delta_KP23, f_max_KP23, LN, [sigma], evolved=True)
        f_PBH_unevolved = constraint_Carr(mc_Carr21, m_delta_KP23, f_max_KP23, LN, [sigma], evolved=False)
        
        if plot_against_mc:
            ax.plot(mc_Carr21, f_PBH_evolved, color=colors[i], dashes=[6, 2], label="LN ($\sigma={:.2f})$".format(sigma))
        else:
            ax.plot(mc_Carr21 * np.exp(-sigma**2), f_PBH_evolved, color=colors[i], dashes=[6, 2], label="LN ($\sigma={:.2f})$".format(sigma))
                        
        # Plot constraint obtained with unevolved MF
        if plot_unevolved:

            if plot_against_mc:
                ax.plot(mc_Carr21, f_PBH_unevolved, color=colors[i], alpha=0.4)
            else:
                ax.plot(mc_Carr21 * np.exp(-sigma**2), f_PBH_unevolved, color=colors[i], alpha=0.4)
            
    # Calculate constraints shown in Fig. 20 of 2002.12778
    m_min = 1e11
    m_max = 1e20
    epsilon = 0.4
    m_star = 5.1e14

    # Calculate delta-function MF constraints, using Eqs. 32-33 of 2002.12778 for beta_prime, and Eq. 57 to convert to f_PBH
    m_delta_Carr21 = 10**np.arange(np.log10(m_min), np.log10(m_max), 0.1)
    f_max_Carr21 = f_PBH_beta_prime(m_delta_Carr21, beta_prime_gamma_rays(m_delta_Carr21))
                
    f_PBH_Carr21 = constraint_Carr(mc_Carr21, m_delta_Carr21, f_max_Carr21, LN, params=[sigma_Carr21], evolved=False)
    
    m_delta_Carr21_loaded, f_max_Carr21_loaded = load_data("./2002.12778/Carr+21_mono_RH.csv")
    mc_Carr21_LN_loaded, fPBH_Carr21_LN_loaded = load_data("./2002.12778/Carr+21_Gamma_ray_LN_RH.csv")
    
    if plot_against_mc:
        ax.set_xlabel("$m_c = m_p\exp(\sigma^2)~[\mathrm{g}]$")
        #ax1.set_xlabel("$m_c~[M_\odot]$")
        
        ax2 = ax.twinx()        
        #ax2.plot(Solmass_to_g(m_delta_Carr21_loaded), f_max_Carr21_loaded, color="k", label="$\delta$ func.")
        ax2.plot(m_delta_Carr21, f_max_Carr21, color="k", label="$\delta$ func.", linestyle="dotted")
        #ax2.plot(Solmass_to_g(mc_Carr21_LN_loaded), fPBH_Carr21_LN_loaded, color="lime", label="LN ($\sigma={:.1f}$)".format(sigma_Carr21))
        ax2.plot(mc_Carr21, f_PBH_Carr21, color="tab:green", label="LN ($\sigma={:.2f}$)".format(sigma_Carr21))
        ax2.legend(title="Carr+ '21", fontsize="x-small", loc=(0.55, 0.02))
        
    else:
        ax.set_xlabel("$m_p~[\mathrm{g}]$")
        #ax1.set_xlabel("$m_p~[M_\odot]$")
        
        ax2 = ax.twinx()        
        #ax2.plot(Solmass_to_g(m_delta_Carr21_loaded), f_max_Carr21_loaded, color="k", label="$\delta$ func.")
        ax2.plot(m_delta_Carr21, f_max_Carr21, color="k", label="$\delta$ func.", linestyle="dotted")
        #ax2.plot(Solmass_to_g(mc_Carr21_LN_loaded) / np.exp(sigma_Carr21**2), fPBH_Carr21_LN_loaded, color="lime", label="LN ($\sigma={:.1f}$)".format(sigma_Carr21))
        ax2.plot(mc_Carr21 * np.exp(-sigma_Carr21**2), f_PBH_Carr21, color="tab:green", label="LN ($\sigma={:.2f}$)".format(sigma_Carr21))
        ax2.legend(title="Carr+ '21", fontsize="x-small", loc=(0.55, 0.02))
            
    ax.set_ylabel("$f_\mathrm{PBH}$")
    
    ax.legend(title="KP' 23", fontsize="x-small", loc=(0.55, 0.35))
    
    for a in [ax, ax2]:
        a.set_xscale("log")
        a.set_yscale("log")
        
        if plot_against_mc:
            a.set_xlim(1e16, 1e20)
        else:
            a.set_xlim(1e16, 3e18)
        a.set_ylim(1e-6, 1)
        
        x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 10)
        a.xaxis.set_major_locator(x_major)
        x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
        a.xaxis.set_minor_locator(x_minor)
        a.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    
    #ax1.set_axis_off()
    ax2.set_axis_off()
    fig.tight_layout()
    
    #%%
    # Plot psi_N, f_max and the integrand in the same figure window
    for sigma_Carr21 in [sigmas_LN[5], 2]:
        m_p = 1e17
        m_c = m_p * np.exp(sigma_Carr21**2) 
        
        fig1, axes = plt.subplots(3, 1, figsize=(5, 14))
        ax1 = axes[0]
        ax2 = axes[1]
        ax3 = axes[2]
        
        ax1.plot(m_delta_KP23, LN(m_delta_KP23, m_c=m_c, sigma=sigma_Carr21) / max(LN(m_delta_KP23, m_c=m_c, sigma=sigma_Carr21)))
        ax1.set_ylabel("$\psi / \psi_\mathrm{max}$")
        ax1.set_ylim(1e-5, 2)
    
        ax2.plot(m_delta_KP23, 1/f_max_KP23, color=(0.5294, 0.3546, 0.7020), label="KP '23")
        ax2.plot(m_delta_Carr21, 1/f_max_Carr21, color="k", label="Carr+ '21")
        ax2.legend(fontsize="x-small")
        ax2.set_ylabel("$1 / f_\mathrm{max}$")
        ax2.set_ylim(1e-2, 1e10)
       
        ax3.plot(m_delta_KP23, LN(m_delta_KP23, m_c=m_c, sigma=sigma_Carr21)/f_max_KP23, color=(0.5294, 0.3546, 0.7020), label="KP '23")
        ax3.plot(m_delta_Carr21, LN(m_delta_Carr21, m_c=m_c, sigma=sigma_Carr21)/f_max_Carr21, color="k", label="Carr+ '21")
        ax3.legend(fontsize="x-small")
        ax3.set_ylabel("$\psi_\mathrm{N}/f_\mathrm{max}~[\mathrm{g}^{-1}]$")
        ax3.set_ylim(1e-20, 1e-8)
    
        for ax in [ax1, ax2, ax3]:
            ax.set_xlabel("$m~[\mathrm{g}]$")
            ax.set_xlim(1e14, 1e17)
            ax.set_xscale("log")
            ax.set_yscale("log")
         
        # Plot the integrand on a linear scale
        fig2, ax4 = plt.subplots(figsize=(6,5))
        ax4.plot(m_delta_KP23, LN(m_delta_KP23, m_c=m_c, sigma=sigma_Carr21)/f_max_KP23, color=(0.5294, 0.3546, 0.7020), label="KP '23")
        ax4.plot(m_delta_Carr21, LN(m_delta_Carr21, m_c=m_c, sigma=sigma_Carr21)/f_max_Carr21, color="k", label="Carr+ '21")
        ax4.legend(fontsize="x-small")
        ax4.set_ylabel("$\psi_\mathrm{N}/f_\mathrm{max}~[\mathrm{g}^{-1}]$")
        ax4.set_xlabel("$m~[\mathrm{g}]$")
        ax4.set_xlim(0, 1e15)
        
        for fig in [fig1, fig2]:
            fig.suptitle("$m_p = {:.1e}".format(m_p) + "~\mathrm{g}$" + ", $\sigma={:.2f}$".format(sigma_Carr21), fontsize="small")
            fig.tight_layout()
        
#%%
    # Plot psi_N, f_max and the integrand in the same figure window (sigma=2 case), using constraints from Auffinger (2022) [2201.01265] Fig. 3 RH panel
    m_delta_A22, f_max_A22 = load_data("./2201.01265/2201.01265_Fig3_EGXB.csv")   

    sigma_A22 = 2
    m_p = 1e16
    m_c = m_p * np.exp(sigma_A22**2) 
    
    fig1, axes = plt.subplots(3, 1, figsize=(5, 14))
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]
    
    from preliminaries import psi_evolved, mass_evolved
    t_0 = 13.8e9 * 365.25 * 86400    # Age of Universe, in seconds
    m_pbh_values_formation = np.concatenate((np.logspace(np.log10(m_star) - 3, np.log10(m_star)), np.arange(m_star, m_star*(1+1e-11), 5e2), np.arange(m_star*(1+1e-11), m_star*(1+1e-6), 1e7), np.logspace(np.log10(m_star*(1+1e-4)), np.log10(max(m_delta_KP23))+4, 1000)))
    m_pbh_values_evolved = mass_evolved(m_pbh_values_formation, t_0)
    psi_initial = LN(m_pbh_values_formation, m_c, sigma_A22)
    psi_evolved = psi_evolved(psi_initial, m_pbh_values_evolved, m_pbh_values_formation)
    psi_evolved_interp_A22 = 10**np.interp(np.log10(m_delta_A22), np.log10(m_pbh_values_evolved), np.log10(psi_evolved))
    psi_evolved_interp_KP23 = 10**np.interp(np.log10(m_delta_KP23), np.log10(m_pbh_values_evolved), np.log10(psi_evolved))
   
    ax1.plot(m_delta_KP23, psi_evolved_interp_KP23)
    ax1.set_ylabel("$\psi_\mathrm{N}~[\mathrm{g}^{-1}]$")

    ax2.plot(m_delta_KP23, 1/f_max_KP23, color=(0.5294, 0.3546, 0.7020), label="KP '23")
    ax2.plot(m_delta_A22, 1/f_max_A22, color="k", label="Auffinger '22")
    ax2.legend(fontsize="x-small")
    ax2.set_ylabel("$1 / f_\mathrm{max}$")
    
    ax3.plot(m_delta_KP23, psi_evolved_interp_KP23/f_max_KP23, color=(0.5294, 0.3546, 0.7020), label="KP '23")
    ax3.plot(m_delta_A22, psi_evolved_interp_A22/f_max_A22, color="k", label="Auffinger '22")
    ax3.legend(fontsize="x-small")
    ax3.set_ylabel("$\psi_\mathrm{N}/f_\mathrm{max}~[\mathrm{g}^{-1}]$")

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel("$m~[\mathrm{g}]$")
        ax.set_xlim(1e14, 1e17)
        ax.set_xscale("log")
        ax.set_yscale("log")
     
    # Plot the integrand on a linear scale
    fig2, ax4 = plt.subplots(figsize=(6,5))
    ax4.plot(m_delta_KP23, psi_evolved_interp_KP23/f_max_KP23, color=(0.5294, 0.3546, 0.7020), label="KP '23")
    ax4.plot(m_delta_A22, psi_evolved_interp_A22/f_max_A22, color="k", label="Auffinger '22")
    ax4.legend(fontsize="x-small")
    ax4.set_ylabel("$\psi_\mathrm{N}/f_\mathrm{max}~[\mathrm{g}^{-1}]$")
    ax4.set_xlim(0, 5e16)
    
    for fig in [fig1, fig2]:
        fig.suptitle("$m_p = {:.1e}".format(m_p) + "~\mathrm{g}$", fontsize="small")
        fig.tight_layout()

        
#%% Test: log-normal plots. Aim is to understand why the extended MF constraints shown in Fig. 20 of 2002.12778 differ so much from the delta-function MF constraints compared to the versions Im using.
    
    # Plot the constraints shown in Fig. 20 of 2002.12778
    m_min = 1e11
    m_max = 1e20
    epsilon = 0.4
    m_star = 5.1e14
    
    m_pbh_values = 10**np.arange(np.log10(m_min), np.log10(m_max), 0.1)
    f_max_values = f_PBH_beta_prime(m_pbh_values, beta_prime_gamma_rays(m_pbh_values))
    #f_max_values /= 2
    mc_values = np.logspace(15, 22, 70)
    
    sigma = 2
    
    fig, ax = plt.subplots(figsize=(8,7))
    ax1 = ax.secondary_xaxis('top', functions=(g_to_Solmass, Solmass_to_g)) 

    f_PBH_values = constraint_Carr(mc_values, m_pbh_values, f_max_values, LN, [sigma], evolved=False)

    m_delta_values_loaded, f_max_loaded = load_data("./2002.12778/Carr+21_mono_RH.csv")
    mc_LN_values_loaded, f_PBH_loaded = load_data("./2002.12778/Carr+21_Gamma_ray_LN_RH.csv")
    
    ax.plot(m_delta_values_loaded * 1.989e33, f_max_loaded, color="tab:grey", label="Delta func.")
    ax.plot(m_pbh_values, f_max_values, color="k", label="Delta func. [repr.]", linestyle="dashed")
    ax.plot(mc_LN_values_loaded * 1.989e33, f_PBH_loaded, color="lime", label="LN ($\sigma={:.1f}$)".format(sigma))
    ax.plot(mc_values, f_PBH_values, color="tab:green", label="LN ($\sigma={:.1f}$) [repr.]".format(sigma), linestyle="dashed")

    ax.set_ylabel("$f_\mathrm{PBH}$")
    ax.set_xlabel("$m_c~[\mathrm{g}]$")
    ax1.set_xlabel("$m_c~[M_\odot]$")
    ax.legend(title="$\sigma$", fontsize="x-small")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-18*1.989e33, 5*1.989e33)
    ax.set_ylim(1e-4, 1)
    fig.tight_layout()    
    
    
    fig, ax = plt.subplots(figsize=(8,7))
    ax1 = ax.secondary_xaxis('top', functions=(g_to_Solmass, Solmass_to_g)) 

    f_PBH_values = constraint_Carr(mc_values, m_pbh_values, f_max_values, LN, [sigma], evolved=False)

    m_delta_values_loaded, f_max_loaded = load_data("./2002.12778/Carr+21_mono_RH.csv")
    mc_LN_values_loaded, f_PBH_loaded = load_data("./2002.12778/Carr+21_Gamma_ray_LN_RH.csv")
    
    ax.plot(m_delta_values_loaded * 1.989e33, f_max_loaded, color="tab:grey", label="Delta func.")
    ax.plot(m_pbh_values, f_max_values, color="k", label="Delta func. [repr.]", linestyle="dashed")
    ax.plot(mc_LN_values_loaded * 1.989e33 * np.exp(-sigma**2), f_PBH_loaded, color="lime", label="LN ($\sigma={:.1f}$)".format(sigma))
    ax.plot(mc_values  * np.exp(-sigma**2), f_PBH_values, color="tab:green", label="LN ($\sigma={:.1f}$) [repr.]".format(sigma), linestyle="dashed")

    ax.set_ylabel("$f_\mathrm{PBH}$")
    ax.set_xlabel("$m_p~[\mathrm{g}]$")
    ax1.set_xlabel("$m_p~[M_\odot]$")
    ax.legend(title="$\sigma$", fontsize="x-small")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-18*1.989e33, 5*1.989e33)
    ax.set_ylim(1e-4, 1)
    fig.tight_layout()    

    
