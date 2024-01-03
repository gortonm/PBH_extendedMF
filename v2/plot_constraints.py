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
from preliminaries import load_data, m_max_SLN, LN, SLN, CC3, frac_diff, constraint_Carr

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

# Load mass function parameters
[Deltas, sigmas_LN, ln_mc_SL, mp_SL, sigmas_SLN, alphas_SLN, mp_CC, alphas_CC, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)


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
    params : Array-like, optional
        Parameters of the mass function. Not required for a delta function, required for extended MFs.
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
        constraints_names_evap, f_PBHs_GC_delta = load_results_Isatis(mf_string="GC_mono", wide=True)
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

            
        f_PBH_instrument = []
        
        for k in range(len(constraints_names_short)):
            # Load constraints for an evolved extended mass function obtained from each instrument
                    
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

        elif mf==SLN:

            mp_GC = [m_max_SLN(m_c, *params, log_m_factor=3, n_steps=1000) for m_c in mc_values]

        elif mf==CC3:
            mp_GC = mc_values
            
    return np.array(mp_GC), np.array(f_PBH_GC)

    
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

    return np.array(mp_KP23), np.array(f_PBH_KP23)

    
def load_data_Voyager_BC19(Deltas, Delta_index, prop_A, with_bkg_subtr, mf=None, evolved=True, exponent_PL_lower=2, prop_B_lower=False):
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
    with_bkg_subtr : Boolean
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
        prop_B_lower = False
    else:
        prop_string = "prop_B"
        
    if not with_bkg_subtr:
        prop_string += "_nobkg"
                    
    if mf == None:
        if with_bkg_subtr:
            prop_string += "_bkg"
    
    if not prop_A :
        if prop_B_lower:
            prop_string += "_lower"
        else:
            prop_string += "_upper"        
    
    if mf == None:
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
        
    return np.array(mp_BC19), np.array(f_PBH_BC19)

    
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
 
    return np.array(mp_Subaru), np.array(f_PBH_Subaru)


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
        
    return np.array(mp_GECCO), np.array(f_PBH_GECCO)


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
        
    return np.array(mp_Sugiyama), np.array(f_PBH_Sugiyama)

    
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
        label="Delta func."
    elif mf == LN:
        label="LN"
    elif mf == SLN:
        label = "SLN"
    elif mf == CC3:
        label = "CC3"
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


def plotter_GC_Isatis(Deltas, Delta_index, ax, color, mf=None, params=None, exponent_PL_lower=2, evolved=True, approx=False, show_label=False, linestyle="solid", linewidth=1, marker=None, alpha=1):
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
        If True, plot constraints obtained using f_max calculated from Isatis. Otherwise, plot constraints calculated from the minimum constraint over each energy bin. The default is False.
    show_label : Boolean, optional
        If True, add a label denoting the fitting function used. The default is False.
    linestyle : String, optional
        Linestyle to use for pplotting. The default is "solid".
    linewidth : Float, optional
        Line width to use for pplotting. The default is 1.
    alpha : Float, optional
        Transparency of the line. The default is 1 (zero transparency).

    Returns
    -------
    None.

    """
    
    mp, f_PBH = load_data_GC_Isatis(Deltas, Delta_index, mf, params, evolved, exponent_PL_lower, approx)
    """
    if not evolved:
        alpha=0.4
    else:
        alpha=1
    """
    if show_label:
        label = find_label(mf)
        ax.plot(mp, f_PBH, color=color, linestyle=linestyle, linewidth=linewidth, label=label, alpha=alpha, marker=marker)
    else:
        ax.plot(mp, f_PBH, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, marker=marker)


def plotter_BC19(Deltas, Delta_index, ax, color, prop_A, with_bkg_subtr, mf=None, exponent_PL_lower=2, evolved=True, show_label=False, prop_B_lower=False, linestyle="solid", linewidth=1, marker=None, alpha=1):
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
    with_bkg_subtr : Boolean
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
    alpha : Float, optional
        Transparency of the line. The default is 1 (zero transparency).

    Returns
    -------
    None.

    """
    mp, f_PBH = load_data_Voyager_BC19(Deltas, Delta_index, prop_A, with_bkg_subtr, mf, evolved, exponent_PL_lower, prop_B_lower)
    """
    if not evolved:
        alpha=0.4
    else:
        alpha=1
    """
    if show_label:
        label = find_label(mf)
        ax.plot(mp, f_PBH, color=color, linestyle=linestyle, linewidth=linewidth, label=label, alpha=alpha, marker=marker)
    else:
        ax.plot(mp, f_PBH, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, marker=marker)


def plotter_BC19_range(Deltas, Delta_index, ax, color, with_bkg_subtr, mf=None, exponent_PL_lower=2, evolved=True, alpha=1):
    """
    Plot extended MF constraints from from the Voyager 1 delta-function MF constraints obtained by Boudaud & Cirelli (2019) [1807.03075].
    Unlike plotter_BC19(), show the range of constraints that is possible due to uncertainties arising from the electron/positron propagation model.       

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
    with_bkg_subtr : Boolean
        If True, load constraints obtained using background subtraction. If False, load constraints obtained without background subtraction.   
    mf : Function, optional
        Fitting function to use. The default is None (delta-function).
    exponent_PL_lower : Float, optional
        Denotes the exponent of the power-law used to extrapolate the delta-function MF. The default is 2.
    evolved : Boolean, optional
        If True, use the evolved form of the fitting function. The default is True.
    alpha : Float, optional
        Transparency of the line. The default is 1 (zero transparency).

    Returns
    -------
    None.

    """
    
    mp_propA, f_PBH_propA = load_data_Voyager_BC19(Deltas, Delta_index, prop_A=True, with_bkg_subtr=with_bkg_subtr, mf=mf, evolved=evolved, exponent_PL_lower=exponent_PL_lower)
    mp_propB_upper, f_PBH_propB_upper = load_data_Voyager_BC19(Deltas, Delta_index, prop_A=False, with_bkg_subtr=with_bkg_subtr, mf=mf, evolved=evolved, exponent_PL_lower=exponent_PL_lower)
    mp_propB_lower, f_PBH_propB_lower = load_data_Voyager_BC19(Deltas, Delta_index, prop_A=False, with_bkg_subtr=with_bkg_subtr, mf=mf, evolved=evolved, exponent_PL_lower=exponent_PL_lower, prop_B_lower=True)
    
    """
    ax.fill_between(mp_propA, f_PBH_propA, np.interp(mp_propA, mp_propB_upper, f_PBH_propB_upper), color=color, linewidth=0, alpha=alpha)
    ax.fill_between(mp_propA, f_PBH_propA, np.interp(mp_propA, mp_propB_lower, f_PBH_propB_lower), color=color, linewidth=0, alpha=alpha)
    ax.fill_between(mp_propB_upper, f_PBH_propB_upper, np.interp(mp_propB_upper, mp_propB_lower, f_PBH_propB_lower), color=color, linewidth=0, alpha=alpha)
    """
    ax.fill_between(mp_propB_upper, f_PBH_propB_upper, np.interp(mp_propB_upper, mp_propB_lower, f_PBH_propB_lower), color=color, linewidth=0, alpha=alpha)
   

def plotter_KP23(Deltas, Delta_index, ax, color, mf=None, exponent_PL_lower=2, evolved=True, show_label=False, linestyle="solid", linewidth=1, marker=None, alpha=1):
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
    alpha : Float, optional
        Transparency of the line. The default is 1 (zero transparency).

    Returns
    -------
    None.

    """    
    
    mp, f_PBH = load_data_KP23(Deltas, Delta_index, mf, evolved, exponent_PL_lower)
    """
    if not evolved:
        alpha=0.4
    else:
        alpha=1
    """
    if show_label:
        label = find_label(mf)
        ax.plot(mp, f_PBH, color=color, linestyle=linestyle, linewidth=linewidth, label=label, alpha=alpha, marker=marker)
    else:
        ax.plot(mp, f_PBH, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, marker=marker)

    
def plotter_Subaru_Croon20(Deltas, Delta_index, ax, color, mf=None, show_label=True, linestyle="solid", linewidth=1, marker=None, alpha=1):
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
    alpha : Float, optional
        Transparency of the line. The default is 1 (zero transparency).

    Returns
    -------
    None.

    """    
    
    mp_Subaru, f_PBH_Subaru = load_data_Subaru_Croon20(Deltas, Delta_index, mf)
    
    if show_label:
        label = find_label(mf)
        ax.plot(mp_Subaru, f_PBH_Subaru, color=color, linewidth=linewidth, linestyle=linestyle, label=label, alpha=alpha, marker=marker)
    else:
        ax.plot(mp_Subaru, f_PBH_Subaru, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha, marker=marker)
        
        
def plotter_GECCO(Deltas, Delta_index, ax, color, mf=None, exponent_PL_lower=2, evolved=True, NFW=True, show_label=False, linestyle="solid", linewidth=1, marker=None, alpha=1):
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
    alpha : Float, optional
        Transparency of the line. The default is 1 (zero transparency).

    Returns
    -------
    None.

    """    
    
    mp_GECCO, f_PBH_GECCO = load_data_GECCO(Deltas, Delta_index, mf, exponent_PL_lower=exponent_PL_lower, evolved=evolved, NFW=NFW)
    
    if show_label:
        label = find_label(mf)
        ax.plot(mp_GECCO, f_PBH_GECCO, color=color, linewidth=linewidth, linestyle=linestyle, label=label, alpha=alpha, marker=marker)
    else:
        ax.plot(mp_GECCO, f_PBH_GECCO, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha, marker=marker)


def plotter_Sugiyama(Deltas, Delta_index, ax, color, mf=None, show_label=True, linestyle="solid", linewidth=1, marker=None, alpha=1):
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
    alpha : Float, optional
        Transparency of the line. The default is 1 (zero transparency).

    Returns
    -------
    None.

    """    

    mp_Sugiyama, f_PBH_Sugiyama = load_data_Sugiyama(Deltas, Delta_index, mf)
    
    if show_label:
        label = find_label(mf)
        ax.plot(mp_Sugiyama, f_PBH_Sugiyama, color=color, linewidth=linewidth, linestyle=linestyle, label=label, alpha=alpha, marker=marker)
    else:
        ax.plot(mp_Sugiyama, f_PBH_Sugiyama, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha, marker=marker)


def Solmass_to_g(m):
    """Convert a mass m (in solar masses) to grams."""
    return 1.989e33 * m


def g_to_Solmass(m):
    """Convert a mass m (in grams) to solar masses."""
    return m / 1.989e33


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

#%% Plot all delta-function MF constraints

if "__main__" == __name__:
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    plot_existing = True
    plot_prospective = True
    
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    Delta_index = 0
    
    if plot_existing:
        plotter_GC_Isatis(Deltas, Delta_index, ax, color="b")
        ax.text(5e16, 0.6,"GC photons", fontsize="xx-small", color="b")
        
        #mp_propA, f_PBH_propA = load_data_Voyager_BC19(Deltas, Delta_index, prop_A=True, with_bkg_subtr=False, mf=None)
        mp_propB_upper, f_PBH_propB_upper = load_data_Voyager_BC19(Deltas, Delta_index, prop_A=False, with_bkg_subtr=False, mf=None)
        #mp_propB_lower, f_PBH_propB_lower = load_data_Voyager_BC19(Deltas, Delta_index, prop_A=False, with_bkg_subtr=False, mf=None, prop_B_lower=True)
        """
        ax.fill_between(mp_propA, f_PBH_propA, np.interp(mp_propA, mp_propB_upper, f_PBH_propB_upper), color="r", linewidth=0, alpha=1)
        ax.fill_between(mp_propA, f_PBH_propA, np.interp(mp_propA, mp_propB_lower, f_PBH_propB_lower), color="r", linewidth=0, alpha=1)
        ax.fill_between(mp_propB_upper, f_PBH_propB_upper, np.interp(mp_propB_upper, mp_propB_lower, f_PBH_propB_lower), color="r", linewidth=0, alpha=1)
        """
        #ax.plot(mp_propA, f_PBH_propA, color="r")
        ax.plot(mp_propB_upper, f_PBH_propB_upper, color="r")
        #ax.plot(mp_propB_lower, f_PBH_propB_lower, color="r")
       
        #mp_propA, f_PBH_propA = load_data_Voyager_BC19(Deltas, Delta_index, prop_A=True, with_bkg_subtr=True, mf=None)
        mp_propB_upper, f_PBH_propB_upper = load_data_Voyager_BC19(Deltas, Delta_index, prop_A=False, with_bkg_subtr=True, mf=None)
        mp_propB_lower, f_PBH_propB_lower = load_data_Voyager_BC19(Deltas, Delta_index, prop_A=False, with_bkg_subtr=True, mf=None, prop_B_lower=True)
        """
        ax.fill_between(mp_propA, f_PBH_propA, np.interp(mp_propA, mp_propB_upper, f_PBH_propB_upper), color="r", alpha=0., linestyle="dashed")
        ax.fill_between(mp_propA, f_PBH_propA, np.interp(mp_propA, mp_propB_lower, f_PBH_propB_lower), color="r", alpha=0, linestyle="dashed")
        ax.fill_between(mp_propB_upper, f_PBH_propB_upper, np.interp(mp_propB_upper, mp_propB_lower, f_PBH_propB_lower), color="r", alpha=0., linestyle="dashed")
        """
        
        #ax.plot(mp_propA, f_PBH_propA, color="r", linestyle="dashed")
        #ax.plot(mp_propB_upper, f_PBH_propB_upper, color="r", linestyle="dashed")
        ax.plot(mp_propB_lower, f_PBH_propB_lower, color="r", linestyle="dashed")
    
        ax.text(1.5e15, 0.3,"Voyager 1", fontsize="xx-small", color="r")    
        
        plotter_KP23(Deltas, Delta_index, ax, color="orange", linestyle="dashdot")
        ax.text(1.4e17, 0.004, "Photons \n (from $e^+ e^-$ annihilation)", fontsize="xx-small", color="orange")
    
        plotter_Subaru_Croon20(Deltas, Delta_index, ax, color="tab:grey")
        ax.text(2.5e22, 0.4,"Subaru-HSC", fontsize="xx-small", color="tab:grey")
        """
        m_delta_values_Berteaud, f_max_Berteaud = load_data("2202.07483/2202.07483_Fig3.csv")
        ax.plot(m_delta_values_Berteaud, f_max_Berteaud, linestyle="dashdot", color="brown")
        ax.text(1e17, 0.02, "GC photons \n (template fit)", fontsize="xx-small", color="brown")
        """
    if plot_prospective:
        plotter_GECCO(Deltas, Delta_index, ax, color="#5F9ED1", linestyle="dotted")
        ax.text(4e17, 0.1,"Future MeV \n gamma-rays", fontsize="xx-small", color="#5F9ED1")
    
        plotter_Sugiyama(Deltas, Delta_index, ax, color="k", linestyle="dotted")
        ax.text(1e21, 0.002,"WD microlensing", fontsize="xx-small", color="k")

    ax.tick_params("x", pad=7)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylabel("$f_\mathrm{PBH}$")
    ax.set_xlabel("$m~[\mathrm{g}]$")
    ax.set_ylim(1e-3, 1)
    ax.set_xlim(1e15, 1e24)
    
    ax1 = ax.secondary_xaxis('top', functions=(g_to_Solmass, Solmass_to_g))
    ax1.set_xlabel("$m~[M_\odot]$", labelpad=14)
    ax1.tick_params("x")
    
    ax2 = ax.secondary_yaxis('right')
    ax2.set_yticklabels([])
      
    fig.tight_layout(pad=0.1)
    
#%% Plot all extended MF constraints

    if "__main__" == __name__:
        # Load mass function parameters.
        [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
        
        plot_existing = True
        plot_prospective = False
        
        mf = SLN
        Delta_index = 6
        
        if mf == LN:
            params = [sigmas_LN[Delta_index]]
            mf_string = "LN"
        elif mf == SLN:
            params = [sigmas_SLN[Delta_index], alphas_SLN[Delta_index]]
            mf_string = "SLN"
        elif mf == CC3:
            params = [alphas_CC3[Delta_index], betas[Delta_index]]      
            mf_string = "CC3"
        
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        
        if plot_existing:
            plotter_GC_Isatis(Deltas, Delta_index, ax, color="b", mf=mf, params=params)
            ax.text(3e17, 0.6,"GC photons", fontsize="xx-small", color="b")
            
            #mp_propA, f_PBH_propA = load_data_Voyager_BC19(Deltas, Delta_index, prop_A=True, with_bkg_subtr=False, mf=mf)
            mp_propB_upper, f_PBH_propB_upper = load_data_Voyager_BC19(Deltas, Delta_index, prop_A=False, with_bkg_subtr=False, mf=mf)
            #mp_propB_lower, f_PBH_propB_lower = load_data_Voyager_BC19(Deltas, Delta_index, prop_A=False, with_bkg_subtr=False, mf=mf, prop_B_lower=True)

            #ax.plot(mp_propA, f_PBH_propA, color="r")
            ax.plot(mp_propB_upper, f_PBH_propB_upper, color="r")
            #ax.plot(mp_propB_lower, f_PBH_propB_lower, color="r")
           
            #mp_propA, f_PBH_propA = load_data_Voyager_BC19(Deltas, Delta_index, prop_A=True, with_bkg_subtr=True, mf=mf)
            mp_propB_upper, f_PBH_propB_upper = load_data_Voyager_BC19(Deltas, Delta_index, prop_A=False, with_bkg_subtr=True, mf=mf)
            #mp_propB_lower, f_PBH_propB_lower = load_data_Voyager_BC19(Deltas, Delta_index, prop_A=False, with_bkg_subtr=True, mf=mf, prop_B_lower=True)
            
            #ax.plot(mp_propA, f_PBH_propA, color="r", linestyle="dashed")
            ax.plot(mp_propB_upper, f_PBH_propB_upper, color="r", linestyle="dashed")
            #ax.plot(mp_propB_lower, f_PBH_propB_lower, color="r", linestyle="dashed")
        
            ax.text(1.3e15, 0.3,"Voyager 1", fontsize="xx-small", color="r")    
            
            plotter_KP23(Deltas, Delta_index, ax, color="orange", linestyle="dashdot", mf=mf)
            ax.text(1.2e17, 0.002, "Photons \n (from $e^+/e^-$ annihilation)", fontsize="xx-small", color="orange")
        
            plotter_Subaru_Croon20(Deltas, Delta_index, ax, color="tab:grey", mf=mf)
            ax.text(2.5e22, 0.4,"Subaru-HSC", fontsize="xx-small", color="tab:grey")
            
            # Calculate uncertainty in 511 keV line from the propagation model
            
            # Density profile parameters
            rho_odot = 0.4 # Local DM density, in GeV / cm^3
            # Maximum distance (from Galactic Centre) where positrons are injected that annhihilate within 1.5kpc of the Galactic Centre, in kpc
            R_large = 3.5
            R_small = 1.5
            # Scale radius, in kpc (values from Ng+ '14 [1310.1915])
            r_s_Iso = 3.5
            r_s_NFW = 20
            r_odot = 8.5 # Galactocentric distance of Sun, in kpc
            annihilation_factor = 0.8 # For most stringgent constraints, consider the scenario where 80% of all positrons injected within R_NFW of the Galactic Centre annihilate
            
            density_integral_NFW = annihilation_factor * rho_odot * r_odot * (r_s_NFW + r_odot)**2 * (np.log(1 + (R_large / r_s_NFW)) - R_large / (R_large + r_s_NFW))
            density_integral_Iso = rho_odot * (r_s_Iso**2 + r_odot**2) * (R_small - r_s_Iso * np.arctan(R_small/r_s_Iso))
            
            print(density_integral_NFW / density_integral_Iso)
            
            mc_511keV, f_PBH_511keV = np.genfromtxt("./Data-tests/PL_exp_-2/%s" % mf_string + "_1912.01014_Carr_Delta={:.1f}_extrapolated_exp-2.txt".format(Deltas[Delta_index]))
            mc_CMB, f_PBH_CMB = np.genfromtxt("./Data-tests/unevolved/PL_exp_-2/%s" % mf_string + "_2108.13256_Carr_Delta={:.1f}_extrapolated_exp-2.txt".format(Deltas[Delta_index]))
            mc_Berteaud, f_PBH_Berteaud = np.genfromtxt("./Data-tests/PL_exp_2/%s" % mf_string +"_2202.07483_Carr_Delta={:.1f}_extrapolated_exp2.txt".format(Deltas[Delta_index]))
            
            if mf == LN:
                mp_CMB = mc_CMB * np.exp(-sigmas_LN[Delta_index]**2)
                mp_511keV = mc_511keV * np.exp(-sigmas_LN[Delta_index]**2)
                mp_Berteaud = mc_Berteaud * np.exp(-sigmas_LN[Delta_index]**2)
            elif mf == SLN:
                mp_CMB = [m_max_SLN(m_c, sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index], log_m_factor=3, n_steps=1000) for m_c in mc_CMB]
                mp_511keV = [m_max_SLN(m_c, sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index], log_m_factor=3, n_steps=1000) for m_c in mc_511keV]
                mp_Berteaud = [m_max_SLN(m_c, sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index], log_m_factor=3, n_steps=1000) for m_c in mc_Berteaud]
            elif mf == CC3:
                mp_CMB = mc_CMB
                mp_511keV = mc_511keV
                mp_Berteaud = mc_Berteaud
                                 
            ax.plot(mp_511keV, f_PBH_511keV, color="g")
            ax.fill_between(mp_511keV, f_PBH_511keV, (density_integral_NFW / density_integral_Iso)*f_PBH_511keV, color="g", alpha=0.3)        
            ax.text(3e17, 5e-2,"511 keV line", fontsize="xx-small", color="g")

            ax.plot(mp_CMB, f_PBH_CMB, linestyle="dashed", color="cyan")
            ax.text(3e15, 0.5,"CMB \n anisotropies", fontsize="xx-small", color="cyan")

            ax.plot(mp_Berteaud, f_PBH_Berteaud, linestyle="dashdot", color="brown")
            ax.text(3e17, 0.1,"GC photons \n (template fit)", fontsize="xx-small", color="brown")

            constraints_names_short = ["COMPTEL_1502.06116", "COMPTEL_1107.0200", "EGRET_0405441", "EGRET_9811211", "Fermi-LAT_1410.3696", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200", "HEAO+balloon_9903492"]
            exponent_PL_lower = -2
            data_folder_EXGB = "./Data-tests/unevolved/PL_exp_{:.0f}".format(exponent_PL_lower)
            constraints_names, f_max_Isatis = load_results_Isatis(mf_string="EXGB_Hazma")
            m_delta_values_loaded = np.logspace(14, 17, 32)

            f_PBH_instrument = []
            
            for i in range(len(constraints_names_short)):
                
                if i in (0, 2, 4, 7):

                    # Load constraints for an evolved extended mass function obtained from each instrument
                    data_filename_EXGB = data_folder_EXGB + "/%s_EXGB_%s" % (mf_string, constraints_names_short[i])  + "_Carr_Delta={:.1f}_approx_unevolved.txt".format(Deltas[Delta_index])
                    mc_EXGB, f_PBH_k = np.genfromtxt(data_filename_EXGB, delimiter="\t")
                    f_PBH_instrument.append(f_PBH_k)
                
            f_PBH_EXGB = envelope(f_PBH_instrument)
            
            if mf == LN:
                mp_EXGB = mc_EXGB * np.exp(-sigmas_LN[Delta_index]**2)
            elif mf == SLN:
                mp_EXGB = [m_max_SLN(m_c, sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index], log_m_factor=3, n_steps=1000) for m_c in mc_EXGB]
            elif mf == CC3:
                mp_EXGB = mc_EXGB

            ax.plot(mp_EXGB, f_PBH_EXGB, color="pink")
            ax.fill_between(mp_EXGB, f_PBH_EXGB, 2*f_PBH_EXGB, color="pink", alpha=0.3)
            ax.text(6e16, 1e-2,"EXGB (Isatis)", fontsize="xx-small", color="pink")
                   
        if plot_prospective:
            plotter_GECCO(Deltas, Delta_index, ax, color="#5F9ED1", linestyle="dotted", mf=mf)
            ax.text(4e17, 0.1,"Future MeV \n gamma-rays", fontsize="xx-small", color="#5F9ED1")
        
            plotter_Sugiyama(Deltas, Delta_index, ax, color="k", linestyle="dotted", mf=mf)
            ax.text(1e21, 0.002,"WD microlensing", fontsize="xx-small", color="k")

        ax.tick_params("x", pad=7)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_ylabel("$f_\mathrm{PBH}$")
        ax.set_xlabel("$m_\mathrm{p}~[\mathrm{g}]$")
        ax.set_ylim(1e-3, 1)
        ax.set_xlim(1e15, 1e24)
        ax.set_title("%s" % mf_string + ", $\Delta={:.1f}$".format(Deltas[Delta_index]))
        
        ax1 = ax.secondary_xaxis('top', functions=(g_to_Solmass, Solmass_to_g))
        ax1.set_xlabel("$m_\mathrm{p}~[M_\odot]$", labelpad=14)
        ax1.tick_params("x")
        
        ax2 = ax.secondary_yaxis('right')
        ax2.set_yticklabels([])
          
        fig.tight_layout(pad=0.1)


#%% Existing constraints

if "__main__" == __name__:
    
    # If True, plot the evaporation constraints used by Isatis (from COMPTEL, INTEGRAL, EGRET and Fermi-LAT)
    plot_GC_Isatis = False
    # If True, plot the evaporation constraints shown in Korwar & Profumo (2023) [2302.04408]
    plot_KP23 = False
    # If True, plot the evaporation constraints from Boudaud & Cirelli (2019) [1807.03075]
    plot_BC19 = True
    # If True, plot unevolved MF constraint
    plot_unevolved = False
    # If True, plot the fractional difference between evolved and unevolved MF results
    plot_fracdiff = False
    # If True, plot the fractional difference between the different fitting functions
    plot_fracdiff_fits = True
    
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
        
    # Load Subaru-HSC delta-fubhnction MF constraint
    m_delta_Subaru, f_PBH_delta_Subaru = load_data("2007.12697/Subaru-HSC_2007.12697_dx=5.csv")
    
    for i in range(len(Deltas)):
        
        if Deltas[i] in [0]:
                        
            fig, ax = plt.subplots(figsize=(9, 5))
            
            if plot_GC_Isatis:
    
                # If required, plot unevolved MF constraints.
                if plot_unevolved:
                    
                    plotter_GC_Isatis(Deltas, i, ax, mf=LN, evolved=False, params=[sigmas_LN[i]], color=colors[1])
                    plotter_GC_Isatis(Deltas, i, ax, mf=SLN, evolved=False, params=[sigmas_SLN[i], alphas_SLN[i]], color=colors[2])
                    plotter_GC_Isatis(Deltas, i, ax, mf=CC3, evolved=False, params=[alphas_CC3[i], betas[i]], color=colors[3])
    
                #plt.suptitle("Existing constraints (showing Galactic Centre photon constraints (Isatis)), $\Delta={:.1f}$".format(Deltas[i]), fontsize="small")
                
                ax.set_xlabel("$m_\mathrm{p}~[\mathrm{g}]$")                
                ax.set_ylabel("$f_\mathrm{PBH}$")
                ax.set_xscale("log")
                ax.set_yscale("log")
    
                plotter_GC_Isatis(Deltas, i, ax, mf=None, color=colors[0])
                plotter_GC_Isatis(Deltas, i, ax, mf=LN, color=colors[1], params=[sigmas_LN[i]], linestyle=(0, (5, 1)))
                plotter_GC_Isatis(Deltas, i, ax, mf=SLN, color=colors[2], params=[sigmas_SLN[i], alphas_SLN[i]], linestyle=(0, (5, 7)))
                plotter_GC_Isatis(Deltas, i, ax, mf=CC3, color=colors[3], params=[alphas_CC3[i], betas[i]], linestyle="dashed")
    
                
                if plot_unevolved and plot_fracdiff:
                    fig1, ax1a = plt.subplots(figsize=(6,6))
                    
                    mp_LN_evolved, f_PBH_LN_evolved = load_data_GC_Isatis(Deltas, i, mf=LN, params=[sigmas_LN[i]], evolved=True)
                    mp_SLN_evolved, f_PBH_SLN_evolved = load_data_GC_Isatis(Deltas, i, mf=SLN, params=[sigmas_SLN[i], alphas_SLN[i]], evolved=True)
                    mp_CC3_evolved, f_PBH_CC3_evolved = load_data_GC_Isatis(Deltas, i, mf=CC3, params=[alphas_CC3[i], betas[i]], evolved=True)
                   
                    mp_LN_unevolved, f_PBH_LN_unevolved = load_data_GC_Isatis(Deltas, i, mf=LN, params=[sigmas_LN[i]], evolved=False)
                    mp_SLN_unevolved, f_PBH_SLN_unevolved = load_data_GC_Isatis(Deltas, i, mf=SLN, params=[sigmas_SLN[i], alphas_SLN[i]], evolved=False)
                    mp_CC3_unevolved, f_PBH_CC3_unevolved = load_data_GC_Isatis(Deltas, i, mf=CC3, params=[alphas_CC3[i], betas[i]], evolved=False)
                    
                    ax1a.plot(mp_LN_evolved, np.abs(frac_diff(f_PBH_LN_evolved, f_PBH_LN_unevolved, mp_LN_evolved, mp_LN_unevolved)), label="LN", color="r")
                    ax1a.plot(mp_SLN_evolved, np.abs(frac_diff(f_PBH_SLN_evolved, f_PBH_SLN_unevolved, mp_SLN_evolved, mp_SLN_unevolved)), label="SLN", color="b")
                    ax1a.plot(mp_CC3_evolved, np.abs(frac_diff(f_PBH_CC3_evolved, f_PBH_CC3_unevolved, mp_CC3_evolved, mp_CC3_unevolved)), label="CC3", color="g")
                    ax1a.set_ylabel("$|\Delta f_\mathrm{PBH} / f_\mathrm{PBH}|$")
                    ax1a.set_xlabel("$m_\mathrm{p}~[\mathrm{g}]$")
                    ax1a.set_xscale("log")
                    ax1a.set_yscale("log")
                    ax1a.set_title("$\Delta={:.1f}$".format(Deltas[i]))
                    ax1a.legend(title="Evolved/unevolved - 1", fontsize="x-small")
                    ax1a.set_xlim(xmin=1e16)
                    ax1a.set_ylim(ymax=1e2)
                    ax1a.grid()
                    fig1.tight_layout()
                
                if plot_fracdiff_fits:
                    fig2, ax2 = plt.subplots(figsize=(5,5))                    
                    mp_LN_evolved, f_PBH_LN_evolved = load_data_GC_Isatis(Deltas, i, mf=LN, params=[sigmas_LN[i]], evolved=True)
                    mp_SLN_evolved, f_PBH_SLN_evolved = load_data_GC_Isatis(Deltas, i, mf=SLN, params=[sigmas_SLN[i], alphas_SLN[i]], evolved=True)
                    mp_CC3_evolved, f_PBH_CC3_evolved = load_data_GC_Isatis(Deltas, i, mf=CC3, params=[alphas_CC3[i], betas[i]], evolved=True)
                                       
                    ax2.plot(mp_SLN_evolved, np.abs(frac_diff(f_PBH_SLN_evolved, f_PBH_LN_evolved, mp_SLN_evolved, mp_LN_evolved)), label="LN vs SLN", color="purple")
                    ax2.plot(mp_LN_evolved, np.abs(frac_diff(f_PBH_LN_evolved, f_PBH_CC3_evolved, mp_LN_evolved, mp_CC3_evolved)), label="LN vs CC3", color="brown")
                    ax2.plot(mp_SLN_evolved, np.abs(frac_diff(f_PBH_SLN_evolved, f_PBH_CC3_evolved, mp_SLN_evolved, mp_CC3_evolved)), label="SLN vs CC3", color="turquoise")
                    ax2.set_ylabel("$|\Delta f_\mathrm{PBH} / f_\mathrm{PBH}|$")
                    ax2.set_xlabel("$m_\mathrm{p}~[\mathrm{g}]$")
                    ax2.set_xscale("log")
                    ax2.set_yscale("log")
                    ax2.set_title("$\Delta={:.1f}$ (GC photons)".format(Deltas[i]), fontsize="small")
                    ax2.legend(fontsize="xx-small")
                    # Set upper x-axis limit to the maximum m_p where f_PBH = 1 is allowed for any of the fitting functions
                    ax2.set_xlim(xmin=1e16, xmax=max([min(mp_CC3_evolved[f_PBH_CC3_evolved > 1]), min(mp_SLN_evolved[f_PBH_SLN_evolved > 1]), min(mp_LN_evolved[f_PBH_LN_evolved > 1])]))
                    ax2.set_ylim(ymin=1e-2, ymax=1e2)
                    
                    y_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
                    ax2.yaxis.set_major_locator(y_major)
                    y_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 5)
                    ax2.yaxis.set_minor_locator(y_minor)
                    ax2.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
                    
                    ax2.grid()
                    fig2.tight_layout()
                    fig2.savefig("./Tests/Figures/Fracdiff_fits/GC_Delta={:.1f}.png".format(Deltas[i]))
                
            if plot_KP23:
                ax.set_xlabel("$m_\mathrm{p}~[\mathrm{g}]$")
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
                    
                    mp_LN_evolved, f_PBH_LN_evolved = load_data_KP23(Deltas, i, mf=LN, evolved=True)
                    mp_SLN_evolved, f_PBH_SLN_evolved = load_data_KP23(Deltas, i, mf=SLN, evolved=True)
                    mp_CC3_evolved, f_PBH_CC3_evolved = load_data_KP23(Deltas, i, mf=CC3, evolved=True)
                    
                    mp_LN_unevolved, f_PBH_LN_unevolved = load_data_KP23(Deltas, i, mf=LN, evolved=False)
                    mp_SLN_unevolved, f_PBH_SLN_unevolved = load_data_KP23(Deltas, i, mf=SLN, evolved=False)
                    mp_CC3_unevolved, f_PBH_CC3_unevolved = load_data_KP23(Deltas, i, mf=CC3, evolved=False)
                   
                    fig1, ax1a = plt.subplots(figsize=(6,6))
                    ax1a.plot(mp_LN_evolved, np.abs(frac_diff(f_PBH_LN_evolved, f_PBH_LN_unevolved, mp_LN_evolved, mp_LN_unevolved)), label="LN", color="r")
                    ax1a.plot(mp_SLN_evolved, np.abs(frac_diff(f_PBH_SLN_evolved, f_PBH_SLN_unevolved, mp_SLN_evolved, mp_SLN_unevolved)), label="SLN", color="b")
                    ax1a.plot(mp_CC3_evolved, np.abs(frac_diff(f_PBH_CC3_evolved, f_PBH_CC3_unevolved, mp_CC3_evolved, mp_CC3_unevolved)), label="CC3", color="g")

                    ax1a.set_ylabel("$\Delta f_\mathrm{PBH} / f_\mathrm{PBH}$")
                    ax1a.set_xlabel("$m_\mathrm{p}~[\mathrm{g}]$")
                    ax1a.set_xscale("log")
                    ax1a.set_yscale("log")
                    ax1a.set_title("$\Delta={:.0f}$".format(Deltas[i]))
                    ax1a.legend(title="Evolved/unevolved - 1", fontsize="x-small")
                    ax1a.set_xlim(xmin=1e16)
                    ax1a.set_ylim(ymax=1e2)
                    ax1a.grid()
                    fig1.tight_layout()
                    
                if plot_fracdiff_fits:
                    fig2, ax2 = plt.subplots(figsize=(5,5))                    
                    mp_LN_evolved, f_PBH_LN_evolved = load_data_KP23(Deltas, i, mf=LN, evolved=True)
                    mp_SLN_evolved, f_PBH_SLN_evolved = load_data_KP23(Deltas, i, mf=SLN, evolved=True)
                    mp_CC3_evolved, f_PBH_CC3_evolved = load_data_KP23(Deltas, i, mf=CC3, evolved=True)
                                       
                    ax2.plot(mp_SLN_evolved, np.abs(frac_diff(f_PBH_SLN_evolved, f_PBH_LN_evolved, mp_SLN_evolved, mp_LN_evolved)), label="LN vs SLN", color="purple")
                    ax2.plot(mp_LN_evolved, np.abs(frac_diff(f_PBH_LN_evolved, f_PBH_CC3_evolved, mp_LN_evolved, mp_CC3_evolved)), label="LN vs CC3", color="brown")
                    ax2.plot(mp_SLN_evolved, np.abs(frac_diff(f_PBH_SLN_evolved, f_PBH_CC3_evolved, mp_SLN_evolved, mp_CC3_evolved)), label="SLN vs CC3", color="turquoise")
                    ax2.set_ylabel("$|\Delta f_\mathrm{PBH} / f_\mathrm{PBH}|$")
                    ax2.set_xlabel("$m_\mathrm{p}~[\mathrm{g}]$")
                    ax2.set_xscale("log")
                    ax2.set_yscale("log")
                    ax2.set_title("$\Delta={:.1f}$ (KP '23)".format(Deltas[i]), fontsize="small")
                    ax2.legend(fontsize="xx-small")
                    # Set upper x-axis limit to the maximum m_p where f_PBH = 1 is allowed for any of the fitting functions
                    ax2.set_xlim(xmin=1e16, xmax=max([min(mp_CC3_evolved[f_PBH_CC3_evolved > 1]), min(mp_SLN_evolved[f_PBH_SLN_evolved > 1]), min(mp_LN_evolved[f_PBH_LN_evolved > 1])]))
                    ax2.set_ylim(ymin=1e-2, ymax=1e2)
                                        
                    y_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
                    ax2.yaxis.set_major_locator(y_major)
                    y_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 5)
                    ax2.yaxis.set_minor_locator(y_minor)
                    ax2.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
                    
                    ax2.grid()
                    fig2.tight_layout()
                    fig2.savefig("./Tests/Figures/Fracdiff_fits/KP23_Delta={:.1f}.png".format(Deltas[i]))
               
                # If required, plot constraints obtained with unevolved MF
                if plot_unevolved:
                    plotter_KP23(Deltas, i, ax, color=colors[0], linestyle="solid")
                    plotter_KP23(Deltas, i, ax, color=colors[1], mf=LN, evolved=False)
                    plotter_KP23(Deltas, i, ax, color=colors[2], mf=SLN, evolved=False)
                    plotter_KP23(Deltas, i, ax, color=colors[3], mf=CC3, evolved=False)                
    
            if plot_BC19:
                
                prop_A = False
                with_bkg_subtr = True
                prop_B_lower = True
                
                mp_propA, f_PBH_propA = load_data_Voyager_BC19(Deltas, i, prop_A=True, with_bkg_subtr=with_bkg_subtr, mf=None)
                mp_propB_upper, f_PBH_propB_upper = load_data_Voyager_BC19(Deltas, i, prop_A=False, with_bkg_subtr=with_bkg_subtr, mf=None)
                mp_propB_lower, f_PBH_propB_lower = load_data_Voyager_BC19(Deltas, i, prop_A=False, with_bkg_subtr=with_bkg_subtr, mf=None, prop_B_lower=True)
                ax.fill_between(mp_propA, f_PBH_propA, np.interp(mp_propA, mp_propB_upper, f_PBH_propB_upper), color="silver", linewidth=0, alpha=1)
                ax.fill_between(mp_propA, f_PBH_propA, np.interp(mp_propA, mp_propB_lower, f_PBH_propB_lower), color="silver", linewidth=0, alpha=1)
                ax.fill_between(mp_propB_upper, f_PBH_propB_upper, np.interp(mp_propB_upper, mp_propB_lower, f_PBH_propB_lower), color="silver", linewidth=0, alpha=1)
                
                for evolved in [True, False]:
                    if evolved == False:
                        alpha = 0.4
                    else:
                        alpha = 1
                    plotter_BC19(Deltas, i, ax, color=colors[1], mf=LN, prop_A=prop_A, with_bkg_subtr=with_bkg_subtr, prop_B_lower=prop_B_lower, alpha=alpha, evolved=evolved)
                    plotter_BC19(Deltas, i, ax, color=colors[1], mf=LN, prop_A=prop_A, with_bkg_subtr=with_bkg_subtr, prop_B_lower=prop_B_lower, alpha=alpha, evolved=evolved)
                    plotter_BC19(Deltas, i, ax, color=colors[2], mf=SLN, prop_A=prop_A, with_bkg_subtr=with_bkg_subtr, prop_B_lower=prop_B_lower, alpha=alpha, evolved=evolved)
                    plotter_BC19(Deltas, i, ax, color=colors[2], mf=SLN, prop_A=prop_A, with_bkg_subtr=with_bkg_subtr, prop_B_lower=prop_B_lower, alpha=alpha, evolved=evolved)
                    plotter_BC19(Deltas, i, ax, color=colors[3], mf=CC3, prop_A=prop_A, with_bkg_subtr=with_bkg_subtr, prop_B_lower=prop_B_lower, alpha=alpha, evolved=evolved)
                    plotter_BC19(Deltas, i, ax, color=colors[3], mf=CC3, prop_A=prop_A, with_bkg_subtr=with_bkg_subtr, prop_B_lower=prop_B_lower, alpha=alpha, evolved=evolved)
    
                ax.set_xlabel("$m_\mathrm{p}~[\mathrm{g}]$")
                ax.set_ylabel("$f_\mathrm{PBH}$")
                ax.set_xscale("log")
                ax.set_yscale("log")
                
                # If required, plot the fractional difference from the delta-function MF constraint
                if plot_fracdiff:
                    
                    mp_LN_evolved, f_PBH_LN_evolved = load_data_Voyager_BC19(Deltas, i, mf=LN, prop_A=prop_A, with_bkg_subtr=with_bkg_subtr, prop_B_lower=prop_B_lower, evolved=True)
                    mp_SLN_evolved, f_PBH_SLN_evolved = load_data_Voyager_BC19(Deltas, i, mf=SLN, prop_A=prop_A, with_bkg_subtr=with_bkg_subtr, prop_B_lower=prop_B_lower, evolved=True)
                    mp_CC3_evolved, f_PBH_CC3_evolved = load_data_Voyager_BC19(Deltas, i, mf=CC3, prop_A=prop_A, with_bkg_subtr=with_bkg_subtr, prop_B_lower=prop_B_lower, evolved=True)
                    
                    mp_LN_unevolved, f_PBH_LN_unevolved = load_data_Voyager_BC19(Deltas, i, mf=LN, prop_A=prop_A, with_bkg_subtr=with_bkg_subtr, prop_B_lower=prop_B_lower, evolved=False)
                    mp_SLN_unevolved, f_PBH_SLN_unevolved = load_data_Voyager_BC19(Deltas, i, mf=SLN, prop_A=prop_A, with_bkg_subtr=with_bkg_subtr, prop_B_lower=prop_B_lower, evolved=False)
                    mp_CC3_unevolved, f_PBH_CC3_unevolved = load_data_Voyager_BC19(Deltas, i, mf=CC3, prop_A=prop_A, with_bkg_subtr=with_bkg_subtr, prop_B_lower=prop_B_lower, evolved=False)
                   
                    fig1, ax1a = plt.subplots(figsize=(6,6))
                    ax1a.plot(mp_LN_evolved, np.abs(frac_diff(f_PBH_LN_evolved, f_PBH_LN_unevolved, mp_LN_evolved, mp_LN_unevolved)), label="LN", color="r")
                    ax1a.plot(mp_SLN_evolved, np.abs(frac_diff(f_PBH_SLN_evolved, f_PBH_SLN_unevolved, mp_SLN_evolved, mp_SLN_unevolved)), label="SLN", color="b")
                    ax1a.plot(mp_CC3_evolved, np.abs(frac_diff(f_PBH_CC3_evolved, f_PBH_CC3_unevolved, mp_CC3_evolved, mp_CC3_unevolved)), label="CC3", color="g")
                    ax1a.set_ylabel("$\Delta f_\mathrm{PBH} / f_\mathrm{PBH}$")
                    ax1a.set_xlabel("$m_\mathrm{p}~[\mathrm{g}]$")
                    ax1a.set_xscale("log")
                    ax1a.set_yscale("log")
                    ax1a.set_title("$\Delta={:.0f}$".format(Deltas[i]))
                    ax1a.legend(title="Evolved/unevolved - 1", fontsize="x-small")
                    ax1a.set_xlim(xmin=1e16)
                    ax1a.set_ylim(ymax=1e2)
                    ax1a.grid()
                    fig1.tight_layout()
                    
                if plot_fracdiff_fits:
                    fig2, ax2 = plt.subplots(figsize=(5,5))                    
                    mp_LN_evolved, f_PBH_LN_evolved = load_data_Voyager_BC19(Deltas, i, mf=LN, evolved=True, prop_A=False, with_bkg_subtr=True)
                    mp_SLN_evolved, f_PBH_SLN_evolved = load_data_Voyager_BC19(Deltas, i, mf=SLN, evolved=True, prop_A=False, with_bkg_subtr=True)
                    mp_CC3_evolved, f_PBH_CC3_evolved = load_data_Voyager_BC19(Deltas, i, mf=CC3, evolved=True, prop_A=False, with_bkg_subtr=True)
                                       
                    ax2.plot(mp_SLN_evolved, np.abs(frac_diff(f_PBH_SLN_evolved, f_PBH_LN_evolved, mp_SLN_evolved, mp_LN_evolved)), label="LN vs SLN", color="purple")
                    ax2.plot(mp_LN_evolved, np.abs(frac_diff(f_PBH_LN_evolved, f_PBH_CC3_evolved, mp_LN_evolved, mp_CC3_evolved)), label="LN vs CC3", color="brown")
                    ax2.plot(mp_SLN_evolved, np.abs(frac_diff(f_PBH_SLN_evolved, f_PBH_CC3_evolved, mp_SLN_evolved, mp_CC3_evolved)), label="SLN vs CC3", color="turquoise")
                    ax2.set_ylabel("$|\Delta f_\mathrm{PBH} / f_\mathrm{PBH}|$")
                    ax2.set_xlabel("$m_\mathrm{p}~[\mathrm{g}]$")
                    ax2.set_xscale("log")
                    ax2.set_yscale("log")
                    ax2.set_title("$\Delta={:.1f}$ (Voyager 1)".format(Deltas[i]), fontsize="small")
                    ax2.legend(fontsize="xx-small")
                    # Set upper x-axis limit to the maximum m_p where f_PBH = 1 is allowed for any of the fitting functions
                    ax2.set_xlim(xmin=1e16, xmax=max([min(mp_CC3_evolved[f_PBH_CC3_evolved > 1]), min(mp_SLN_evolved[f_PBH_SLN_evolved > 1]), min(mp_LN_evolved[f_PBH_LN_evolved > 1])]))
                    ax2.set_ylim(ymin=1e-2, ymax=1e2)
                                        
                    y_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
                    ax2.yaxis.set_major_locator(y_major)
                    y_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 5)
                    ax2.yaxis.set_minor_locator(y_minor)
                    ax2.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
                    
                    ax2.grid()
                    fig2.tight_layout()
                    fig2.savefig("./Tests/Figures/Fracdiff_fits/BC19_Delta={:.1f}.png".format(Deltas[i]))
   
            if plot_fracdiff_fits:
                fig2, ax2 = plt.subplots(figsize=(5,5))
    
                mp_LN_evolved, f_PBH_LN_evolved = load_data_Subaru_Croon20(Deltas, i, mf=LN)
                mp_SLN_evolved, f_PBH_SLN_evolved = load_data_Subaru_Croon20(Deltas, i, mf=SLN)
                mp_CC3_evolved, f_PBH_CC3_evolved = load_data_Subaru_Croon20(Deltas, i, mf=CC3)
                                   
                ax2.plot(mp_SLN_evolved, np.abs(frac_diff(f_PBH_LN_evolved, f_PBH_SLN_evolved, mp_LN_evolved, mp_SLN_evolved)), label="LN vs SLN", color="purple")
                ax2.plot(mp_LN_evolved, np.abs(frac_diff(f_PBH_CC3_evolved, f_PBH_LN_evolved, mp_CC3_evolved, mp_LN_evolved)), label="LN vs CC3", color="brown")
                ax2.plot(mp_SLN_evolved, np.abs(frac_diff(f_PBH_CC3_evolved, f_PBH_SLN_evolved, mp_CC3_evolved, mp_SLN_evolved)), label="SLN vs CC3", color="turquoise")
                ax2.set_ylabel("$|\Delta f_\mathrm{PBH} / f_\mathrm{PBH}|$")
                ax2.set_xlabel("$m_\mathrm{p}~[\mathrm{g}]$")
                ax2.set_xscale("log")
                ax2.set_yscale("log")
                ax2.set_title("$\Delta={:.1f}$ (Subaru-HSC)".format(Deltas[i]), fontsize="small")
                ax2.legend(fontsize="xx-small")
                # Set lower x-axis limit to the minimum m_p where f_PBH = 1 is allowed for any of the fitting functions
                ax2.set_xlim(xmin=min([min(mp_CC3_evolved[f_PBH_CC3_evolved < 1]), min(mp_SLN_evolved[f_PBH_SLN_evolved < 1]), min(mp_LN_evolved[f_PBH_LN_evolved < 1])]), xmax=1e24)
                ax2.set_ylim(ymin=1e-2, ymax=1e3)
                                    
                y_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
                ax2.yaxis.set_major_locator(y_major)
                y_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 5)
                ax2.yaxis.set_minor_locator(y_minor)
                ax2.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
                    
                ax2.grid()
                fig2.tight_layout()
                fig2.savefig("./Tests/Figures/Fracdiff_fits/Subaru_Delta={:.1f}.png".format(Deltas[i]))
 
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
# Version showing constraints obtained using different observations in different colours, and using line style to distinguish between fitting functions.

if "__main__" == __name__:
        
    #colors = ['silver', 'tab:red', 'tab:blue', 'k', 'k']
    colors = ['tab:blue', 'k']
                    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
        
    for i in range(len(Deltas)):
                        
        fig, ax = plt.subplots(figsize=(9, 5))
        # Plot prospective extended MF constraints from the white dwarf microlensing survey proposed in Sugiyama et al. (2020) [1905.06066].
        
        ax.set_xlabel("$m_\mathrm{p}~[\mathrm{g}]$")
        ax.set_ylabel("$f_\mathrm{PBH}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        
        NFW = False
        
        # If True, plot unevolved MF constraint
        plot_unevolved = True
        # If True, plot the fractional difference between evolved and unevolved MF results
        plot_fracdiff = True
        # If True, plot the fractional difference between the different fitting functions
        plot_fracdiff_fits = False
        
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
        plotter_GECCO(Deltas, i, ax, color=colors[0], NFW=NFW, linestyle="solid", linewidth=1)
        plotter_GECCO(Deltas, i, ax, color=colors[0], NFW=NFW, mf=LN, linestyle="dotted")
        plotter_GECCO(Deltas, i, ax, color=colors[0], NFW=NFW, mf=SLN, linestyle="dashdot")
        plotter_GECCO(Deltas, i, ax, color=colors[0], NFW=NFW, mf=CC3, linestyle="dashed")

        plotter_Sugiyama(Deltas, i, ax, color=colors[1], linestyle="solid", linewidth=1, show_label=show_label)
        plotter_Sugiyama(Deltas, i, ax, color=colors[1], mf=LN, linestyle="dotted", show_label=show_label)
        plotter_Sugiyama(Deltas, i, ax, color=colors[1], mf=SLN, linestyle="dashdot", show_label=show_label)
        plotter_Sugiyama(Deltas, i, ax, color=colors[1], mf=CC3, linestyle="dashed", show_label=show_label)

        set_ticks_grid(ax)
        ymin, ymax = 1e-3, 1

        ax.set_xlim(xmin_evap, xmax_micro)
        ax.set_ylim(ymin, ymax)
        ax.legend(fontsize="xx-small", title="$\Delta={:.0f}$".format(Deltas[i]), loc="upper right")
        
        #plt.suptitle("Prospective constraints, $\Delta={:.1f}$".format(Deltas[i]), fontsize="small")
        fig.tight_layout()
        fig.savefig("./Results/Figures/fPBH_Delta={:.1f}_prospective.pdf".format(Deltas[i]))
        fig.savefig("./Results/Figures/fPBH_Delta={:.1f}_prospective.png".format(Deltas[i]))
        
        if plot_unevolved and plot_fracdiff:
            fig1, ax1a = plt.subplots(figsize=(6,6))
            
            mp_LN_evolved, f_PBH_LN_evolved = load_data_GECCO(Deltas, i, mf=LN, evolved=True)
            mp_SLN_evolved, f_PBH_SLN_evolved = load_data_GECCO(Deltas, i, mf=SLN, evolved=True)
            mp_CC3_evolved, f_PBH_CC3_evolved = load_data_GECCO(Deltas, i, mf=CC3, evolved=True)
           
            mp_LN_unevolved, f_PBH_LN_unevolved = load_data_GECCO(Deltas, i, mf=LN, evolved=False)
            mp_SLN_unevolved, f_PBH_SLN_unevolved = load_data_GECCO(Deltas, i, mf=SLN, evolved=False)
            mp_CC3_unevolved, f_PBH_CC3_unevolved = load_data_GECCO(Deltas, i, mf=CC3, evolved=False)
            
            ax1a.plot(mp_LN_evolved, np.abs(frac_diff(f_PBH_LN_evolved, f_PBH_LN_unevolved, mp_LN_evolved, mp_LN_unevolved)), label="LN", color="r")
            ax1a.plot(mp_SLN_evolved, np.abs(frac_diff(f_PBH_SLN_evolved, f_PBH_SLN_unevolved, mp_SLN_evolved, mp_SLN_unevolved)), label="SLN", color="b")
            ax1a.plot(mp_CC3_evolved, np.abs(frac_diff(f_PBH_CC3_evolved, f_PBH_CC3_unevolved, mp_CC3_evolved, mp_CC3_unevolved)), label="CC3", color="g")
            ax1a.set_ylabel("$|\Delta f_\mathrm{PBH} / f_\mathrm{PBH}|$")
            ax1a.set_xlabel("$m_\mathrm{p}~[\mathrm{g}]$")
            ax1a.set_xscale("log")
            ax1a.set_yscale("log")
            ax1a.set_title("$\Delta={:.1f}$".format(Deltas[i]))
            ax1a.legend(title="Evolved/unevolved - 1", fontsize="x-small")
            ax1a.set_xlim(xmin=1e16)
            ax1a.set_ylim(ymax=1e2)
            ax1a.grid()
            fig1.tight_layout()
         
        
        if plot_fracdiff_fits:
            
            if Deltas[i] in (0, 2, 5):
                fig2, ax2 = plt.subplots(figsize=(5,5))                    
                mp_LN_evolved, f_PBH_LN_evolved = load_data_GECCO(Deltas, i, mf=LN, evolved=True)
                mp_SLN_evolved, f_PBH_SLN_evolved = load_data_GECCO(Deltas, i, mf=SLN, evolved=True)
                mp_CC3_evolved, f_PBH_CC3_evolved = load_data_GECCO(Deltas, i, mf=CC3, evolved=True)
                                   
                ax2.plot(mp_SLN_evolved, np.abs(frac_diff(f_PBH_SLN_evolved, f_PBH_LN_evolved, mp_SLN_evolved, mp_LN_evolved)), label="LN vs SLN", color="purple")
                ax2.plot(mp_LN_evolved, np.abs(frac_diff(f_PBH_LN_evolved, f_PBH_CC3_evolved, mp_LN_evolved, mp_CC3_evolved)), label="LN vs CC3", color="brown")
                ax2.plot(mp_SLN_evolved, np.abs(frac_diff(f_PBH_SLN_evolved, f_PBH_CC3_evolved, mp_SLN_evolved, mp_CC3_evolved)), label="SLN vs CC3", color="turquoise")
                ax2.set_ylabel("$|\Delta f_\mathrm{PBH} / f_\mathrm{PBH}|$")
                ax2.set_xlabel("$m_\mathrm{p}~[\mathrm{g}]$")
                ax2.set_xscale("log")
                ax2.set_yscale("log")
                ax2.set_title("$\Delta={:.1f}$ (future MeV telescopes)".format(Deltas[i]), fontsize="small")
                ax2.legend(fontsize="xx-small")
                # Set upper x-axis limit to the maximum m_p where f_PBH = 1 is allowed for any of the fitting functions
                ax2.set_xlim(xmin=1e16, xmax=max([min(mp_CC3_evolved[f_PBH_CC3_evolved > 1]), min(mp_SLN_evolved[f_PBH_SLN_evolved > 1]), min(mp_LN_evolved[f_PBH_LN_evolved > 1])]))
                ax2.set_ylim(ymin=1e-2, ymax=1e2)
                
                y_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
                ax2.yaxis.set_major_locator(y_major)
                y_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 5)
                ax2.yaxis.set_minor_locator(y_minor)
                ax2.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
                
                ax2.grid()
                fig2.tight_layout()
                fig2.savefig("./Tests/Figures/Fracdiff_fits/GECCO_Delta={:.1f}.png".format(Deltas[i]))
    
                fig3, ax3 = plt.subplots(figsize=(5,5))                    
                mp_LN_evolved, f_PBH_LN_evolved = load_data_Sugiyama(Deltas, i, mf=LN)
                mp_SLN_evolved, f_PBH_SLN_evolved = load_data_Sugiyama(Deltas, i, mf=SLN)
                mp_CC3_evolved, f_PBH_CC3_evolved = load_data_Sugiyama(Deltas, i, mf=CC3)
                                   
                ax3.plot(mp_SLN_evolved, np.abs(frac_diff(f_PBH_SLN_evolved, f_PBH_LN_evolved, mp_SLN_evolved, mp_LN_evolved)), label="LN vs SLN", color="purple")
                ax3.plot(mp_LN_evolved, np.abs(frac_diff(f_PBH_LN_evolved, f_PBH_CC3_evolved, mp_LN_evolved, mp_CC3_evolved)), label="LN vs CC3", color="brown")
                ax3.plot(mp_SLN_evolved, np.abs(frac_diff(f_PBH_SLN_evolved, f_PBH_CC3_evolved, mp_SLN_evolved, mp_CC3_evolved)), label="SLN vs CC3", color="turquoise")
                ax3.set_ylabel("$|\Delta f_\mathrm{PBH} / f_\mathrm{PBH}|$")
                ax3.set_xlabel("$m_\mathrm{p}~[\mathrm{g}]$")
                ax3.set_xscale("log")
                ax3.set_yscale("log")
                ax3.set_title("$\Delta={:.1f}$ (future WD microlensing)".format(Deltas[i]), fontsize="small")
                ax3.legend(fontsize="xx-small")
                # Set upper x-axis limit to the maximum m_p where f_PBH = 1 is allowed for any of the fitting functions
                ax3.set_xlim(xmin=min([min(mp_CC3_evolved[f_PBH_CC3_evolved < 1]), min(mp_SLN_evolved[f_PBH_SLN_evolved < 1]), min(mp_LN_evolved[f_PBH_LN_evolved < 1])]), xmax=1e24)
                ax3.set_ylim(ymin=1e-2, ymax=1e2)
                
                y_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
                ax3.yaxis.set_major_locator(y_major)
                y_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 5)
                ax3.yaxis.set_minor_locator(y_minor)
                ax3.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
                
                ax3.grid()
                fig3.tight_layout()
                fig3.savefig("./Tests/Figures/Fracdiff_fits/WD_Delta={:.1f}.png".format(Deltas[i]))

#%% Plot fractional difference between extended MF constraints from GECCO obtained with the NFW and Einasto profiles

if "__main__" == __name__:
        
    for Delta_index in range(len(Deltas)):
        fig, ax = plt.subplots(figsize=(6, 5))
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        
        colors = ["r", "b", "g"]
        
        print(Deltas[Delta_index])
        
        for i, mf in enumerate([LN, SLN, CC3]):
            mp_NFW, f_PBH_NFW = load_data_GECCO(Deltas, Delta_index, mf=mf, NFW=True)
            mp_Einasto, f_PBH_Einasto = load_data_GECCO(Deltas, Delta_index, mf=mf, NFW=False)

            ax.plot(mp_Einasto, f_PBH_Einasto, color=colors[i])
            ax.plot(mp_NFW, f_PBH_NFW, color=colors[i], linestyle="dashed")
            ax1.plot(mp_NFW, frac_diff(f_PBH_NFW, f_PBH_Einasto, mp_NFW, mp_Einasto), color=colors[i])

        ax.set_ylabel("$f_\mathrm{PBH}$")
        ax.set_xlabel("$m_\mathrm{p}~[\mathrm{g}]$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1)
        fig.tight_layout()
            
        ax1.set_ylabel("$|\Delta f_\mathrm{PBH} / f_\mathrm{PBH}|$")
        ax1.set_xlabel("$m_\mathrm{p}~[\mathrm{g}]$")
        ax1.set_xlim(min(mp_NFW), max(mp_NFW[f_PBH_NFW < 1]))
        ax1.set_ylim(1.5, 2)
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        fig1.tight_layout()
     
        
#%% Existing constraints
# Version showing constraints obtained using different observations in different colours, and using line style to distinguish between fitting functions.

if "__main__" == __name__:
    
    # If True, plot constraints obtained with background subtraction
    with_bkg_subtr = True
    # If True, load the more stringent "prop B" constraint
    prop_B_lower = True
    
    colors = ['r', 'b', 'orange', 'tab:grey']
    linestyles = ['solid', 'dotted', 'dashdot', 'dashed']
                    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)

    Delta_indices = [0, 5, 6]
    
    linewidth = 1

    if len(Delta_indices) == 4:
        plt.figure(figsize=(12, 12))
        ax0 = plt.subplot(2, 2, 1)    
        
    elif len(Delta_indices) == 3:
        plt.figure(figsize=(14, 5.5))
        ax = plt.subplot(1, 3, 1)
        
    for axis_index, Delta_index in enumerate(Delta_indices):
        
        if len(Delta_indices) == 4:
            ax = plt.subplot(2, 2, axis_index + 1, sharex=ax)
        elif len(Delta_indices) == 3:
            ax = plt.subplot(1, 3, axis_index + 1, sharex=ax)
                        
        if with_bkg_subtr:
            plotter_KP23(Deltas, Delta_index, ax, color=colors[2], linestyle=linestyles[0], linewidth=linewidth)
            plotter_KP23(Deltas, Delta_index, ax, color=colors[2], mf=LN, linestyle=linestyles[1], linewidth=linewidth)
            plotter_KP23(Deltas, Delta_index, ax, color=colors[2], mf=SLN, linestyle=linestyles[2], linewidth=linewidth)
            plotter_KP23(Deltas, Delta_index, ax, color=colors[2], mf=CC3, linestyle=linestyles[3], linewidth=linewidth)

        else:
            plotter_GC_Isatis(Deltas, Delta_index, ax, mf=None, evolved=False, color=colors[1], linestyle=linestyles[0], linewidth=linewidth)
            plotter_GC_Isatis(Deltas, Delta_index, ax, mf=LN, params=[sigmas_LN[Delta_index]], color=colors[1], linestyle=linestyles[1], linewidth=linewidth)
            plotter_GC_Isatis(Deltas, Delta_index, ax, mf=SLN, params=[sigmas_SLN[Delta_index], alphas_SLN[Delta_index]], color=colors[1], linestyle=linestyles[2], linewidth=linewidth)
            plotter_GC_Isatis(Deltas, Delta_index, ax, mf=CC3, params=[alphas_CC3[Delta_index], betas[Delta_index]], color=colors[1], linestyle=linestyles[3], linewidth=linewidth)

        mp_propA, f_PBH_propA = load_data_Voyager_BC19(Deltas, Delta_index, prop_A=True, with_bkg_subtr=with_bkg_subtr, mf=None)
        mp_propB_upper, f_PBH_propB_upper = load_data_Voyager_BC19(Deltas, Delta_index, prop_A=False, with_bkg_subtr=with_bkg_subtr, mf=None)
        mp_propB_lower, f_PBH_propB_lower = load_data_Voyager_BC19(Deltas, Delta_index, prop_A=False, with_bkg_subtr=with_bkg_subtr, mf=None, prop_B_lower=True)

        plotter_BC19(Deltas, Delta_index, ax, color=colors[0], mf=None, prop_A=False, with_bkg_subtr=with_bkg_subtr, prop_B_lower=prop_B_lower)
        plotter_BC19(Deltas, Delta_index, ax, color=colors[0], mf=LN, prop_A=False, with_bkg_subtr=with_bkg_subtr, prop_B_lower=prop_B_lower, linestyle=linestyles[1], linewidth=linewidth)
        plotter_BC19(Deltas, Delta_index, ax, color=colors[0], mf=SLN, prop_A=False, with_bkg_subtr=with_bkg_subtr, prop_B_lower=prop_B_lower, linestyle=linestyles[2], linewidth=linewidth)
        plotter_BC19(Deltas, Delta_index, ax, color=colors[0], mf=CC3, prop_A=False, with_bkg_subtr=with_bkg_subtr, prop_B_lower=prop_B_lower, linestyle=linestyles[3], linewidth=linewidth)
                                
        xmin, xmax = 1e16, 5e23
        ymin, ymax = 1e-3, 1
        
        show_label_Subaru = False
            
        # Plot Subaru-HSC constraints        
        plotter_Subaru_Croon20(Deltas, Delta_index, ax, color=colors[3], linestyle=linestyles[0], linewidth=linewidth, show_label=show_label_Subaru)
        plotter_Subaru_Croon20(Deltas, Delta_index, ax, color=colors[3], mf=LN,  linestyle=linestyles[1], linewidth=linewidth, show_label=show_label_Subaru)
        plotter_Subaru_Croon20(Deltas, Delta_index, ax, color=colors[3], mf=SLN, linestyle=linestyles[2], linewidth=linewidth, show_label=show_label_Subaru)
        plotter_Subaru_Croon20(Deltas, Delta_index, ax, color=colors[3], mf=CC3, linestyle=linestyles[3], linewidth=linewidth, show_label=show_label_Subaru)

        ax.plot(0, 0, color="k", linestyle=linestyles[0], label="Delta func.")
        ax.plot(0, 0, color="k", linestyle=linestyles[1], label="LN")
        ax.plot(0, 0, color="k", linestyle=linestyles[2], label="SLN")
        ax.plot(0, 0, color="k", linestyle=linestyles[3], label="CC3")

        ax.tick_params("x", pad=7)
        ax.set_xlabel("$m_\mathrm{p}~[\mathrm{g}]$")
        ax.set_ylabel("$f_\mathrm{PBH}$")           
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        
        ax1 = ax.secondary_xaxis('top', functions=(g_to_Solmass, Solmass_to_g))
        ax1.set_xlabel("$m_\mathrm{p}~[M_\odot]$", labelpad=14)
        ax1.tick_params("x")
        
        ax2 = ax.secondary_yaxis('right')
        ax2.set_yticklabels([])
        
        if Deltas[Delta_index] in (1,2):
            ax.legend(fontsize="xx-small", loc="lower center")
                
        ax.set_title("$\Delta={:.0f}$".format(Deltas[Delta_index]), pad=25)
        
    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(wspace=0.3)
    
#%% Prospective constraints
# Version showing constraints obtained using different observations in different colours, and using line style to distinguish between fitting functions.

if "__main__" == __name__:
        
    colors = ['tab:blue', 'k']
    linestyles = ['solid', 'dotted', 'dashdot', 'dashed']
                    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)

    Delta_indices = [0, 5, 6]
    
    if len(Delta_indices) == 4:
        plt.figure(figsize=(12, 12))
        ax0 = plt.subplot(2, 2, 1)

    elif len(Delta_indices) == 3:
        plt.figure(figsize=(14, 5.5))
        ax = plt.subplot(1, 3, 1)
                  
    for axis_index, Delta_index in enumerate(Delta_indices):
        
        if len(Delta_indices) == 4:
            ax = plt.subplot(2, 2, axis_index + 1, sharex=ax)
        elif len(Delta_indices) == 3:
            ax = plt.subplot(1, 3, axis_index + 1, sharex=ax)
           
        # Plot prospective extended MF constraints from the white dwarf microlensing survey proposed in Sugiyama et al. (2020) [1905.06066].        
        NFW = False
        show_label = False
        
        # Set axis limits
        xmin, xmax = 1e16, 5e23
        ymin, ymax = 1e-3, 1

        # plot Einasto profile results            
        plotter_GECCO(Deltas, Delta_index, ax, color=colors[0], NFW=NFW, linestyle=linestyles[0])
        plotter_GECCO(Deltas, Delta_index, ax, color=colors[0], NFW=NFW, mf=LN, linestyle=linestyles[1])
        plotter_GECCO(Deltas, Delta_index, ax, color=colors[0], NFW=NFW, mf=CC3, linestyle=linestyles[3])
        plotter_GECCO(Deltas, Delta_index, ax, color=colors[0], NFW=NFW, mf=SLN, linestyle=linestyles[2])

        plotter_Sugiyama(Deltas, Delta_index, ax, color=colors[1], linestyle=linestyles[0], show_label=show_label)
        plotter_Sugiyama(Deltas, Delta_index, ax, color=colors[1], mf=LN, linestyle=linestyles[1], show_label=show_label)
        plotter_Sugiyama(Deltas, Delta_index, ax, color=colors[1], mf=CC3, linestyle=linestyles[3], show_label=show_label)
        plotter_Sugiyama(Deltas, Delta_index, ax, color=colors[1], mf=SLN, linestyle=linestyles[2], show_label=show_label)
                
        ax.plot(0, 0, color="k", linestyle=linestyles[0], label="Delta func.")
        ax.plot(0, 0, color="k", linestyle=linestyles[1], label="LN")
        ax.plot(0, 0, color="k", linestyle=linestyles[2], label="SLN")
        ax.plot(0, 0, color="k", linestyle=linestyles[3], label="CC3")

        ax.tick_params("x", pad=7)
        ax.set_xlabel("$m_\mathrm{p}~[\mathrm{g}]$")
        ax.set_ylabel("$f_\mathrm{PBH}$")           
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        
        ax1 = ax.secondary_xaxis('top', functions=(g_to_Solmass, Solmass_to_g))
        ax1.set_xlabel("$m_\mathrm{p}~[M_\odot]$", labelpad=14)
        ax1.tick_params("x")
        
        ax2 = ax.secondary_yaxis('right')
        ax2.set_yticklabels([])
        
        if Deltas[Delta_index] in (1,2):
            ax.legend(fontsize="xx-small", loc=[0.21, 0.05])
                
        ax.set_title("$\Delta={:.0f}$".format(Deltas[Delta_index]), pad=15)
   
    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(wspace=0.3)
        
#%% Plot the evaporation constraints on the same plot

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)

    Delta_indices = [0, 5, 6]
    
    if len(Delta_indices) == 4:
        plt.figure(figsize=(9, 13))
        ax = plt.subplot(2, 2, 1)

    elif len(Delta_indices) == 3:
        plt.figure(figsize=(14, 5))
        ax = plt.subplot(1, 3, 1)
                  
    for axis_index, Delta_index in enumerate(Delta_indices):
        
        if len(Delta_indices) == 4:
            ax = plt.subplot(2, 2, axis_index + 1, sharex=ax)
        elif len(Delta_indices) == 3:
            ax = plt.subplot(1, 3, axis_index + 1, sharex=ax)
    
        if Deltas[Delta_index] < 5:
            mf = CC3
            params = [alphas_CC3[Delta_index], betas[Delta_index]]
            mf_label="CC3"
        else:
            mf = SLN
            params = [sigmas_SLN[Delta_index], alphas_SLN[Delta_index]]
            mf_label="SLN"
           
        # Present constraints obtained using background subtraction with dashed lines
        plotter_KP23(Deltas, Delta_index, ax, color="tab:orange", mf=mf, linestyle="dashed")
         
        # Present constraints obtained without background subtraction with solid lines        
        plotter_GC_Isatis(Deltas, Delta_index, ax, color="b", mf=mf, params=params)
        
        ax.plot(0, 0, color="b", label="GC photons")
        ax.plot(0, 0, color="tab:orange", linestyle="dashed", label="Photons \n (from $e^+$ annihilation)")
        
        #plotter_BC19(Deltas, Delta_index, ax, color="r", mf=mf, prop_A=False, with_bkg_subtr=False, prop_B_lower=True)
        plotter_BC19(Deltas, Delta_index, ax, color="r", mf=mf, prop_A=False, with_bkg_subtr=False, prop_B_lower=False)
        plotter_BC19_range(Deltas, Delta_index, ax, color="r", mf=mf, with_bkg_subtr=False, alpha=0.3)
      
        # Present constraints obtained using background subtraction with dashed lines                    
        #plotter_BC19(Deltas, Delta_index, ax, color="r", mf=mf, prop_A=False, with_bkg_subtr=True, prop_B_lower=True, linestyle="dashed")
        plotter_BC19(Deltas, Delta_index, ax, color="r", mf=mf, prop_A=False, with_bkg_subtr=True, prop_B_lower=False, linestyle="dashed")
        plotter_BC19_range(Deltas, Delta_index, ax, color="r", mf=mf, with_bkg_subtr=True, alpha=0.3)
        
        ax.plot(0, 0, color="r", label="Voyager 1 \n w/o bkg. \n subtraction")
        ax.plot(0, 0, color="r", linestyle="dashed", label="Voyager 1 \n w/ bkg. \n subtraction")
        
        if Deltas[Delta_index] == 0:
            ax.legend(fontsize="xx-small")
            
        ax.set_title("$\Delta={:.0f}$".format(Deltas[Delta_index]) + " (%s)" % mf_label, fontsize="small")
        ax.tick_params("x", pad=7)
        ax.set_xlabel("$m_\mathrm{p}~[\mathrm{g}]$")
        ax.set_ylabel("$f_\mathrm{PBH}$")           
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e16, 1e19)
        ax.set_ylim(1e-5, 1)
        
        x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
        ax.xaxis.set_major_locator(x_major)
        x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 5)
        ax.xaxis.set_minor_locator(x_minor)
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    plt.tight_layout(pad=0.1)        
    plt.subplots_adjust(wspace=0.3)

#%% Plot constraints for different Delta on the same plot (Korwar & Profumo 2023)

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
            
    exponent_PL_lower = 2
    data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower)
    
    plot_LN = False
    plot_SLN = False
    plot_CC3 = True
    
    plot_unevolved = True
    
    fig, ax = plt.subplots(figsize=(6,6))
    fig1, ax1 = plt.subplots(figsize=(6,6))
       
    # Delta-function MF constraints
    
    # Power-law exponent to use between 1e11g and 1e15g.
    exponent_PL_lower = 2.0
    m_delta, f_PBH_delta = load_data_KP23(Deltas, Delta_index=0, evolved=True, exponent_PL_lower=exponent_PL_lower)
    ax.plot(m_delta, f_PBH_delta, color="tab:gray", label="Delta func.", linewidth=2)

    colors=["tab:blue", "tab:orange", "tab:green", "tab:red"]
        
    for i, Delta_index in enumerate([0, 5, 6]):
    #for i, Delta_index in enumerate([1, 2, 3, 4]):
        
        data_filename_LN = data_folder + "/LN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)
        data_filename_SLN = data_folder + "/SLN_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)
        data_filename_CC3 = data_folder + "/CC3_2302.04408_Carr_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL_lower)
    
               
        if plot_LN:
            mp_LN, f_PBH_LN = load_data_KP23(Deltas, Delta_index, mf=LN, evolved=True)
            ax.plot(mp_LN, f_PBH_LN, color=colors[i], dashes=[6, 2], label="{:.0f}".format(Deltas[Delta_index]))
            ax1.plot(mp_LN, np.abs(frac_diff(f_PBH_LN, f_PBH_delta, mp_LN, m_delta)), color=colors[i], label="{:.0f}".format(Deltas[Delta_index]))
            
        elif plot_SLN:
            mp_SLN, f_PBH_SLN = load_data_KP23(Deltas, Delta_index, mf=SLN, evolved=True)
            ax.plot(mp_SLN, f_PBH_SLN, color=colors[i], linestyle=(0, (5, 7)), label="{:.0f}".format(Deltas[Delta_index]))
            ax1.plot(mp_SLN, np.abs(frac_diff(f_PBH_SLN, f_PBH_delta, mp_SLN, m_delta)), color=colors[i], label="{:.0f}".format(Deltas[Delta_index]))

        elif plot_CC3:
            mp_CC3, f_PBH_CC3 = load_data_KP23(Deltas, Delta_index, mf=CC3, evolved=True)
            ax.plot(mp_CC3, f_PBH_CC3, color=colors[i], linestyle="dashed", label="{:.0f}".format(Deltas[Delta_index]))
            ax1.plot(mp_CC3, np.abs(frac_diff(f_PBH_CC3, f_PBH_delta, mp_CC3, m_delta)), color=colors[i], label="{:.0f}".format(Deltas[Delta_index]))
           
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
        a.set_xlabel("$m_\mathrm{p}~[\mathrm{g}]$")
        a.legend(title="$\Delta$", fontsize="x-small")
        a.set_xscale("log")
        a.set_yscale("log")
        
    ax.set_xlim(1e15, 2e18)
    ax.set_ylim(1e-7, 1)
    ax1.set_xlim(1e15, max(m_delta))
    ax1.set_ylim(1e-2, 2)
    
    x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 10)
    ax.xaxis.set_major_locator(x_major)
    x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax.xaxis.set_minor_locator(x_minor)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.tick_params("x", pad=7)
    
    for f in [fig, fig1]:
        if plot_LN:
            f.suptitle("Korwar \& Profumo 2023 constraints (LN)", fontsize="small")
        elif plot_SLN:
            f.suptitle("Korwar \& Profumo 2023 constraints (SLN)", fontsize="small")
        elif plot_CC3:
            f.suptitle("Korwar \& Profumo 2023 constraints (CC3)", fontsize="small")
        f.tight_layout()
 

#%% Plot constraints for different Delta on the same plot (Boudaud & Cirelli 2019)

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
            
    exponent_PL_lower = 2
    data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower)
    
    plot_LN = False
    plot_SLN = False
    plot_CC3 = True
        
    fig, ax = plt.subplots(figsize=(6,6))       
    
    # Power-law exponent to use between 1e11g and 1e15g.
    exponent_PL_lower = 2.0

    colors=["tab:blue", "tab:orange", "tab:green", "tab:red"]
    with_bkg_subtr = True
        
    for i, Delta_index in enumerate([0, 5, 6]):
    #for i, Delta_index in enumerate([1, 2, 3, 4]):
    
        ax.plot(0, 0, color=colors[i], label="{:.0f}".format(Deltas[Delta_index]))

        if plot_LN:
            plotter_BC19_range(Deltas, Delta_index, ax, color=colors[i], mf=LN, evolved=True, exponent_PL_lower=exponent_PL_lower, with_bkg_subtr=with_bkg_subtr)             
            #plotter_BC19(Deltas, Delta_index, ax, color="k", mf=LN, evolved=True, exponent_PL_lower=exponent_PL_lower, with_bkg_subtr=with_bkg_subtr, prop_A=False, prop_B_lower=True)
            
        elif plot_SLN:
            plotter_BC19_range(Deltas, Delta_index, ax, color=colors[i], mf=SLN, evolved=True, exponent_PL_lower=exponent_PL_lower, with_bkg_subtr=with_bkg_subtr)             

        elif plot_CC3:
            plotter_BC19_range(Deltas, Delta_index, ax, color=colors[i], mf=CC3, evolved=True, exponent_PL_lower=exponent_PL_lower, with_bkg_subtr=with_bkg_subtr)             
    
    ax.set_ylabel("$f_\mathrm{PBH}$")
    ax.set_xlabel("$m_\mathrm{p}~[\mathrm{g}]$")
    ax.legend(title="$\Delta$", fontsize="x-small", loc="lower right")
    ax.set_xscale("log")
    ax.set_yscale("log")  
    ax.set_xlim(1e14, 5e18)
    ax.set_ylim(1e-10, 1)

    x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 10)
    ax.xaxis.set_major_locator(x_major)
    x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax.xaxis.set_minor_locator(x_minor)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.tick_params("x", pad=7)
    
    if plot_LN:
        fig.suptitle("Boudaud \& Cirelli 2019 constraints (LN)", fontsize="small")
    elif plot_SLN:
        fig.suptitle("Boudaud \& Cirelli 2019 constraints (SLN)", fontsize="small")
    elif plot_CC3:
        fig.suptitle("Boudaud \& Cirelli 2019 constraints (CC3)", fontsize="small")
    fig.tight_layout()
 

#%% Plot constraints for different Delta on the same plot

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
            
    exponent_PL_lower = 2
    data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower)
            
    #fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, tight_layout = {'pad': 0}, figsize=(10,15))
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, tight_layout = {'pad': 0}, figsize=(15,5))
       
    # Linestyles for different constraints
    linestyles = ["dashdot", "solid", "dashed", "dotted"]

    colors_LN = ["k", "tab:red", "pink"]
    colors_SLN = ["k", "tab:blue", "deepskyblue"]
    colors_CC3 = ["k", "tab:green", "lime"]
    
    # Opacities and line widths for different Delta:
    linewidth_values = [2, 2, 2]
    
    with_bkg_subtr = True
    
    for i, Delta_index in enumerate([0, 5, 6]):
        
        # Delta-function MF constraints
        plotter_Subaru_Croon20(Deltas, Delta_index, ax, color="tab:gray", linewidth=2, linestyle=linestyles[0], show_label=False)
        if with_bkg_subtr:
            plotter_KP23(Deltas, Delta_index, ax, color="tab:gray", linewidth=2, linestyle=linestyles[2])
        else:
            plotter_GC_Isatis(Deltas, Delta_index, ax, color="tab:gray", linewidth=2, linestyle=linestyles[1])

        plotter_BC19(Deltas, Delta_index, ax, color="tab:gray", prop_A=False, linewidth=2, with_bkg_subtr=with_bkg_subtr, linestyle=linestyles[3])
        
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
            data_filename_LN = data_folder + "/LN_GC_%s" % constraints_names_short[k] + "_Carr_Delta={:.1f}.txt".format(Deltas[Delta_index])
            data_filename_SLN = data_folder + "/SLN_GC_%s" % constraints_names_short[k]  + "_Carr_Delta={:.1f}.txt".format(Deltas[Delta_index])
            data_filename_CC3 = data_folder + "/CC3_GC_%s" % constraints_names_short[k]  + "_Carr_Delta={:.1f}.txt".format(Deltas[Delta_index])
                            
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
        if with_bkg_subtr:
            plotter_KP23(Deltas, Delta_index, ax0, color=colors_LN[i], mf=LN, linestyle=linestyles[2], linewidth=linewidth_values[i])
        else:
            plotter_GC_Isatis(Deltas, Delta_index, ax0, color=colors_LN[i], mf=LN, linestyle=linestyles[1], linewidth=linewidth_values[i], params=[sigmas_LN[Delta_index]])
        plotter_Subaru_Croon20(Deltas, Delta_index, ax0, color=colors_LN[i], mf=LN, linestyle=linestyles[0], linewidth=linewidth_values[i], show_label=False)
        plotter_BC19(Deltas, Delta_index, ax0, color=colors_LN[i], mf=LN, prop_A=False, with_bkg_subtr=with_bkg_subtr, linestyle=linestyles[3], linewidth=linewidth_values[i])
        ax0.set_title("LN") 
        ax0.plot(0, 0, color=colors_LN[i], label="{:.0f}".format(Deltas[Delta_index]))
        
        # Plot SLN MF results
        if with_bkg_subtr: 
            plotter_KP23(Deltas, Delta_index, ax1, color=colors_SLN[i], mf=SLN, linestyle=linestyles[2], linewidth=linewidth_values[i])
        else:
            plotter_GC_Isatis(Deltas, Delta_index, ax1, color=colors_SLN[i], mf=SLN, linestyle=linestyles[1], linewidth=linewidth_values[i], params=[sigmas_SLN[Delta_index], alphas_SLN[Delta_index]])
        plotter_Subaru_Croon20(Deltas, Delta_index, ax1, color=colors_SLN[i], mf=SLN, linestyle=linestyles[0], linewidth=linewidth_values[i], show_label=False)
        plotter_BC19(Deltas, Delta_index, ax1, color=colors_SLN[i], mf=SLN, prop_A=False, with_bkg_subtr=with_bkg_subtr, linestyle=linestyles[3], linewidth=linewidth_values[i])
        ax1.set_title("SLN") 
        ax1.plot(0, 0, color=colors_SLN[i], label="{:.0f}".format(Deltas[Delta_index]))
               
        # Plot CC3 MF results
        if with_bkg_subtr:
            plotter_KP23(Deltas, Delta_index, ax2, color=colors_CC3[i], mf=CC3, linestyle=linestyles[2], linewidth=linewidth_values[i])
        else:
            plotter_GC_Isatis(Deltas, Delta_index, ax2, color=colors_CC3[i], mf=CC3, linestyle=linestyles[1], linewidth=linewidth_values[i], params=[alphas_CC3[Delta_index], betas[Delta_index]])
        plotter_Subaru_Croon20(Deltas, Delta_index, ax2, color=colors_CC3[i], mf=CC3, linestyle=linestyles[0], linewidth=linewidth_values[i], show_label=False)
        plotter_BC19(Deltas, Delta_index, ax2, color=colors_CC3[i], mf=CC3, prop_A=False, with_bkg_subtr=with_bkg_subtr, linestyle=linestyles[3], linewidth=linewidth_values[i])
        ax2.set_title("CC3") 
        ax2.plot(0, 0, color=colors_CC3[i], label="{:.0f}".format(Deltas[Delta_index]))
               
    for ax in[ax0, ax1, ax2]:
        ax.set_xlabel("$m_\mathrm{p}~[\mathrm{g}]$")
        ax.set_ylabel("$f_\mathrm{PBH}$")
        ax.set_xscale("log")
        ax.set_yscale("log")  
        ax.set_xlim(1e15, 1e18)
        ax.set_ylim(1e-6, 1)
        ax.legend(title="$\Delta$", fontsize="xx-small", loc="lower right")
        ax.tick_params("x", pad=7)
        
        x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 10)
        ax.xaxis.set_major_locator(x_major)
        x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
        ax.xaxis.set_minor_locator(x_minor)
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.tick_params("x", pad=7)

    fig.tight_layout(pad=0.1)
    fig.subplots_adjust(wspace=0.3)
 
    
#%% Plot the most stringent GC photon constraint.

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    colors_evap = ["tab:orange", "tab:green", "tab:red", "tab:blue"]
    constraints_names_short = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
    
    # if True, use the constraints obtained for evolved MFs. Otherwise, use constraints obtained using the unevolved MFs.
    evolved = True
    
    # Parameters used for convergence tests in Galactic Centre constraints.
    cutoff = 1e-4
    delta_log_m = 1e-3
    E_number = 500    
    
    if E_number < 1e3:
        energies_string = "E{:.0f}".format(E_number)
    else:
        energies_string = "E{:.0f}".format(np.log10(E_number))
    
    plot_LN = False
    plot_SLN = False
    plot_CC3 = True
    
    approx = False
        
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
            if approx:
                if evolved:
                    data_filename_LN = data_folder + "/LN_GC_%s" % constraints_names_short[k] + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[i])
                    data_filename_SLN = data_folder + "/SLN_GC_%s" % constraints_names_short[k]  + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[i])
                    data_filename_CC3 = data_folder + "/CC3_GC_%s" % constraints_names_short[k]  + "_Carr_Delta={:.1f}_approx.txt".format(Deltas[i])
                else:
                    data_filename_LN = data_folder + "/LN_GC_%s" % constraints_names_short[k] + "_Carr_Delta={:.1f}_approx_unevolved.txt".format(Deltas[i])
                    data_filename_SLN = data_folder + "/SLN_GC_%s" % constraints_names_short[k]  + "_Carr_Delta={:.1f}_approx_unevolved.txt".format(Deltas[i])
                    data_filename_CC3 = data_folder + "/CC3_GC_%s" % constraints_names_short[k]  + "_Carr_Delta={:.1f}_approx_unevolved.txt".format(Deltas[i])
            else:
                if evolved:
                    data_filename_LN = data_folder + "/LN_GC_%s" % constraints_names_short[k] + "_Carr_Delta={:.1f}.txt".format(Deltas[i])
                    data_filename_SLN = data_folder + "/SLN_GC_%s" % constraints_names_short[k]  + "_Carr_Delta={:.1f}.txt".format(Deltas[i])
                    data_filename_CC3 = data_folder + "/CC3_GC_%s" % constraints_names_short[k]  + "_Carr_Delta={:.1f}.txt".format(Deltas[i])
                else:
                    data_filename_LN = data_folder + "/LN_GC_%s" % constraints_names_short[k] + "_Carr_Delta={:.1f}_unevolved.txt".format(Deltas[i])
                    data_filename_SLN = data_folder + "/SLN_GC_%s" % constraints_names_short[k]  + "_Carr_Delta={:.1f}_unevolved.txt".format(Deltas[i])
                    data_filename_CC3 = data_folder + "/CC3_GC_%s" % constraints_names_short[k]  + "_Carr_Delta={:.1f}_unevolved.txt".format(Deltas[i])

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
        ax.tick_params("x", pad=7)
        ax.set_xlabel("$m_\mathrm{p}~[\mathrm{g}]$")
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
    with_bkg_subtr = False
    
    # For Galactic Centre photon constraints
    constraints_names_short = ["COMPTEL_1107.0200", "EGRET_9811211", "Fermi-LAT_1101.1381", "INTEGRAL_1107.0200"]
    mc_values_evap = np.logspace(14, 20, 120)


    for Delta_index in range(len(Deltas)):
        
        fig, ax = plt.subplots(figsize=(8, 8))
                
        for colour_index, prop in enumerate([True, True]):
            
            prop_B = not prop_A
            
            for linestyle_index, with_bkg_subtr in enumerate([False, True]):
            
                label=""
                
                if prop_A:
                    prop_string = "prop_A"
                    label = "Prop A"
        
                elif prop_B:
                    prop_string = "prop_B"
                    label = "Prop B"
                    
                if not with_bkg_subtr:
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
                    
                with_bkg_subtr = not with_bkg_subtr
                   
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
        ax.tick_params("x", pad=7)
        ax.set_xlabel("$m_\mathrm{p}~[\mathrm{g}]$")
        ax.set_ylabel("$f_\mathrm{PBH}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize="x-small")
        fig.tight_layout()
        fig.suptitle("$\Delta={:.1f}$".format(Deltas[Delta_index]))
    
#%% Plot the constraints shown in Fig. 20 of Carr et al. (2021) [2002.12778].
from preliminaries import constraint_Carr, LN
    
m_min = 1e11
m_max = 1e20
epsilon = 0.4
m_star = 5.1e14

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
    
    m_pbh_values = 10**np.arange(np.log10(m_min), np.log10(m_max), 0.1)
    f_max_values = f_PBH_beta_prime(m_pbh_values, beta_prime_gamma_rays(m_pbh_values))
    mc_values = np.logspace(15, 22, 70)
    
    sigma = 2
    
    fig, ax = plt.subplots(figsize=(8,7))
    ax1 = ax.secondary_xaxis('top', functions=(g_to_Solmass, Solmass_to_g)) 

    f_PBH_values = constraint_Carr(mc_values, m_pbh_values, f_max_values, LN, [sigma], evolved=False)
    f_PBH_values_evolved = constraint_Carr(mc_values, m_pbh_values, f_max_values, LN, [sigma], evolved=True)

    m_delta_values_loaded, f_max_loaded = load_data("./2002.12778/Carr+21_mono_RH.csv")
    mc_LN_values_loaded, f_PBH_loaded = load_data("./2002.12778/Carr+21_Gamma_ray_LN_RH.csv")
    
    ax.plot(m_delta_values_loaded * 1.989e33, f_max_loaded, color="tab:grey", label="Delta func.")
    ax.plot(m_pbh_values, f_max_values, color="k", label="Delta func. [repr.]", linestyle="dashed")
    ax.plot(mc_LN_values_loaded * 1.989e33, f_PBH_loaded, color="lime", label="LN ($\sigma={:.1f}$)".format(sigma))
    ax.plot(mc_values, f_PBH_values, color="tab:green", label="LN ($\sigma={:.1f}$) [repr.]".format(sigma), linestyle="dashed")
    ax.plot(mc_values, f_PBH_values_evolved, color="tab:green", label="LN ($\sigma={:.1f}$) \n [repr., evolved]".format(sigma), linestyle="dashed", alpha=0.5)

    ax.set_ylabel("$f_\mathrm{PBH}$")
    ax.set_xlabel("$m_c~[\mathrm{g}]$")
    ax1.set_xlabel("$m_c~[M_\odot]$", labelpad=14)
    ax.legend(title="$\sigma$", fontsize="x-small")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-18*1.989e33, 5*1.989e33)
    ax.set_ylim(1e-4, 1)
    ax.tick_params(pad=7)
    fig.tight_layout()
    
        
#%% Plot extended MF constraints shown in Fig. 20 of 2002.12778 on the same axes as in that figure.

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    exponent_PL_lower = 2
                
    data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower)
            
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    plot_BC19 = False
    plot_KP23 = True
    
    if plot_BC19:
        comp_color = "r"
        
        # Plot delta-function MF constraint from Boudaud & Cirelli (2019)
        prop_A = False
        with_bkg_subtr = True
        prop_B_lower = True
        
        # Load delta-function MF constraint from Boudaud & Cirelli (2019)
        m_delta_BC19_loaded, f_max_BC19_loaded = load_data_Voyager_BC19(Deltas=Deltas, Delta_index=0, prop_A=False, with_bkg_subtr=True, prop_B_lower=True, mf=None)
        m_delta_BC19_extrapolated = 10**np.arange(11, np.log10(min(m_delta_BC19_loaded))+0.01, 0.1)
        f_max_BC19_extrapolated = min(f_max_BC19_loaded) * np.power(m_delta_BC19_extrapolated / min(m_delta_BC19_loaded), exponent_PL_lower)
        f_max_comparison = np.concatenate((f_max_BC19_extrapolated, f_max_BC19_loaded))
        m_delta_comparison = np.concatenate((m_delta_BC19_extrapolated, m_delta_BC19_loaded))
    
    elif plot_KP23:
        comp_color = "orange"
        
        # Power-law exponent to use between 1e15g and 1e16g.
        exponent_PL_upper = 2.0
        # Power-law exponent to use between 1e11g and 1e15g.
        exponent_PL_lower = 2.0
        
        # Load delta-function MF constraint from Boudaud & Cirelli (2019)
        m_delta_values_comparison, f_max_comparison = load_data_KP23(Deltas=Deltas, Delta_index=0, mf=None, exponent_PL_lower=exponent_PL_lower)

    # Range of characteristic masses for obtaining constraints
    mc_Carr21 = np.logspace(14, 22, 1000)
    mc_Carr21 = 10**np.arange(14, 22.5, 0.1)
    # Calculate delta-function MF constraints from Carr et al. (2021), using Eqs. 32-33 of 2002.12778 for beta_prime, and Eq. 57 to convert to f_PBH
    m_min = 1e11
    m_max = 1e20
    epsilon = 0.4
    m_star = 5.1e14
    
    # Calculate delta-function MF constraint from Carr et al. (2021), using the fitting formula for beta prime
    m_delta_Carr21_fit = 10**np.arange(np.log10(m_min), np.log10(m_max), 0.1)
    f_max_Carr21_fit = f_PBH_beta_prime(m_delta_Carr21_fit, beta_prime_gamma_rays(m_delta_Carr21_fit))
    
    for i, ax in enumerate(axes.flatten()):
        
        if plot_BC19:
            plotter_BC19(Deltas, 0, ax, color="r", prop_A=prop_A, with_bkg_subtr=with_bkg_subtr, prop_B_lower=prop_B_lower)
        elif plot_KP23:
            plotter_KP23(Deltas, 0, ax, color="orange")

        ax.plot(m_delta_Carr21_fit, f_max_Carr21_fit, color="cyan", linestyle="solid")
        
        if i in (0, 2):
            sigma = 2
        
        else:
            sigma = sigmas_LN[-1]
            
        f_PBH_comparison_evolved = constraint_Carr(mc_Carr21, m_delta_comparison, f_max_comparison, LN, [sigma], evolved=True)
        f_PBH_comparison_unevolved = constraint_Carr(mc_Carr21, m_delta_comparison, f_max_comparison, LN, [sigma], evolved=False)
        
        f_PBH_Carr21_fit_evolved = constraint_Carr(mc_Carr21, m_delta_Carr21_fit, f_max_Carr21_fit, LN, params=[sigma], evolved=True)
        f_PBH_Carr21_fit_unevolved = constraint_Carr(mc_Carr21, m_delta_Carr21_fit, f_max_Carr21_fit, LN, params=[sigma], evolved=False)

        mp_Carr21 = mc_Carr21 * np.exp(-sigma**2)
        
        ax.text(5e19, 5e-3, "$\sigma={:.1f}$".format(sigma))
        ax.set_ylabel(r"$f_{\rm PBH}$")
        ax.set_xlim(1e16, 1e21)
        ax.set_ylim(1e-3, 1)
        ax.set_xscale("log")
        ax.set_yscale("log")

        x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 10)
        ax.xaxis.set_major_locator(x_major)
        x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
        ax.xaxis.set_minor_locator(x_minor)
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.tick_params("x", pad=7)
       
        ax1 = ax.secondary_xaxis('top', functions=(g_to_Solmass, Solmass_to_g))
        ax1.tick_params("x")
        
        x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 10)
        ax1.xaxis.set_major_locator(x_major)
        x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
        ax1.xaxis.set_minor_locator(x_minor)
        ax1.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax1.tick_params("x", pad=7)
 
        # Plot against m_c in the top two panels
        if i <= 1:
            ax.set_xlabel(r"$m_c~[{\rm g}]$")
            ax.plot(mc_Carr21, f_PBH_comparison_evolved, color=comp_color, linestyle="dotted")
            ax.plot(mc_Carr21, f_PBH_comparison_unevolved, color=comp_color, linestyle="dotted", alpha=0.5)
            ax.plot(mc_Carr21, f_PBH_Carr21_fit_evolved, color="tab:cyan", linestyle="dotted")
            ax.plot(mc_Carr21, f_PBH_Carr21_fit_unevolved, color="tab:cyan", linestyle="dotted", alpha=0.5)
            ax1.set_xlabel("$m_c~[M_\odot]$", labelpad=14)
           
        else:
            ax.set_xlabel(r"$m_{\rm p}~[{\rm g}]$")
            ax.plot(mp_Carr21, f_PBH_comparison_evolved, color=comp_color, linestyle="dotted")
            ax.plot(mp_Carr21, f_PBH_comparison_unevolved, color=comp_color, linestyle="dotted", alpha=0.5)
            ax.plot(mp_Carr21, f_PBH_Carr21_fit_evolved, color="tab:cyan", linestyle="dotted")
            ax.plot(mp_Carr21, f_PBH_Carr21_fit_unevolved, color="tab:cyan", linestyle="dotted", alpha=0.5)
            ax1.set_xlabel(r"$m_{\rm p}~[M_\odot]$", labelpad=14)
    
    fig.subplots_adjust(hspace=0.1)
    fig.tight_layout(pad=0.3)

#%% Version with different values of sigma shown in each figure, with m_c and m_p used as the x-axis.

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    exponent_PL_lower = 2
                
    data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower)
            
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    
    # Plot delta-function MF constraint from Boudaud & Cirelli (2019)
    prop_A = False
    with_bkg_subtr = True
    prop_B_lower = True
    
    # Load delta-function MF constraint from Boudaud & Cirelli (2019)
    m_delta_BC19_loaded, f_max_BC19_loaded = load_data_Voyager_BC19(Deltas=Deltas, Delta_index=0, prop_A=False, with_bkg_subtr=True, prop_B_lower=True, mf=None)
    m_delta_BC19_extrapolated = 10**np.arange(11, np.log10(min(m_delta_BC19_loaded))+0.01, 0.1)
    f_max_BC19_extrapolated = min(f_max_BC19_loaded) * np.power(m_delta_BC19_extrapolated / min(m_delta_BC19_loaded), exponent_PL_lower)
    f_max_BC19 = np.concatenate((f_max_BC19_extrapolated, f_max_BC19_loaded))
    m_delta_BC19 = np.concatenate((m_delta_BC19_extrapolated, m_delta_BC19_loaded))
        
    # Range of characteristic masses for obtaining constraints
    mc_Carr21 = 10**np.arange(14, 22.5, 0.1)
    # Calculate delta-function MF constraints from Carr et al. (2021), using Eqs. 32-33 of 2002.12778 for beta_prime, and Eq. 57 to convert to f_PBH
    m_min = 1e11
    m_max = 1e20
    epsilon = 0.4
    m_star = 5.1e14

    for i, ax in enumerate(axes.flatten()):
                       
        if i == 0:
            sigma = sigmas_LN[-1]
        
        else:
            sigma = 2
        
        ax.plot(m_delta_BC19, f_max_BC19, color="r")
        ax.plot(m_delta_BC19 * np.exp(-sigma**2), f_max_BC19, color="b")
       
        f_PBH_BC19_evolved = constraint_Carr(mc_Carr21, m_delta_BC19, f_max_BC19, LN, [sigma], evolved=True)

        mp_Carr21 = mc_Carr21 * np.exp(-sigma**2)
        
        ax.text(7e17, 1.5e-3, "$\sigma={:.1f}$".format(sigma), fontsize="small")
        ax.set_ylabel(r"$f_{\rm PBH}$")
        ax.set_xlim(1e14, 1e19)
        ax.set_ylim(1e-3, 1)
        ax.set_xscale("log")
        ax.set_yscale("log")

        x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 10)
        ax.xaxis.set_major_locator(x_major)
        x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
        ax.xaxis.set_minor_locator(x_minor)
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

        ax.tick_params("x", pad=7)
        if i == 0:
            ax1 = ax.secondary_xaxis('top', functions=(lambda x: x * np.exp(sigmas_LN[-1]**2), lambda x: x * np.exp(-sigmas_LN[-1]**2)), color="b")
        else:
            ax1 = ax.secondary_xaxis('top', functions=(lambda x: x * np.exp(2**2), lambda x: x * np.exp(-2**2)), color="b")
        x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 100)
        ax1.xaxis.set_major_locator(x_major)
        x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
        ax1.xaxis.set_minor_locator(x_minor)
        ax1.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax1.tick_params("x", pad=7, which="both", color="b")
 
        # Plot against m_c in the top two panels
        ax1.set_xlabel(r"$m_c~[{\rm g}]$", labelpad=14, color="b")
        ax.plot(mp_Carr21, f_PBH_BC19_evolved, color="r", linestyle="dotted")
        
        ax.set_xlabel(r"$m_{\rm p}~[{\rm g}]$")        
        ax.spines["top"].set_color("b")
            
    fig.subplots_adjust(hspace=0.1)
    fig.tight_layout(pad=0.3)
    
#%% Version with different values of sigma shown in each panel, with m_c and m_p used as the x-axis.

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    exponent_PL_lower = 2
                
    data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower)
    
    original = False
    generic_mass = True
    text_on_plot = False

    if generic_mass:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig, ax = plt.subplots(figsize=(7, 5))

    plot_BC19 = True
    plot_KP23 = False
        
    if plot_BC19:
        
        comp_color = "r"
        
        # Plot delta-function MF constraint from Boudaud & Cirelli (2019)
        prop_A = False
        with_bkg_subtr = True
        prop_B_lower = True
        
        # Load delta-function MF constraint from Boudaud & Cirelli (2019)
        m_delta_loaded, f_max_loaded = load_data_Voyager_BC19(Deltas=Deltas, Delta_index=0, prop_A=False, with_bkg_subtr=True, prop_B_lower=True, mf=None)
    
        m_delta_extrapolated = 10**np.arange(11, np.log10(min(m_delta_loaded))+0.01, 0.1)
        f_max_extrapolated = min(f_max_loaded) * np.power(m_delta_extrapolated / min(m_delta_loaded), exponent_PL_lower)
        f_max = np.concatenate((f_max_extrapolated, f_max_loaded))
        m_delta = np.concatenate((m_delta_extrapolated, m_delta_loaded))
    
    elif plot_KP23:
        
        comp_color = "orange"
        
        # Power-law exponent to use between 1e15g and 1e16g.
        exponent_PL_upper = 2.0
        # Power-law exponent to use between 1e11g and 1e15g.
        exponent_PL_lower = 2.0
        
        # Load delta-function MF constraint from Boudaud & Cirelli (2019)
        m_delta, f_max = load_data_KP23(Deltas=Deltas, Delta_index=0, mf=None, exponent_PL_lower=exponent_PL_lower)

    # Range of characteristic masses for obtaining constraints
    mc_Carr21 = 10**np.arange(14, 22.5, 0.1)
        
    # Values of sigma (parameter in log-normal distribution)
    sigmas = [sigmas_LN[-1], 2]
                            
    f_PBH_evolved_sigma0 = constraint_Carr(mc_Carr21, m_delta, f_max, LN, [sigmas[0]], evolved=True)
    f_PBH_evolved_sigma1 = constraint_Carr(mc_Carr21, m_delta, f_max, LN, [sigmas[1]], evolved=True)
    if original:
        ax.plot(m_delta, f_max, color="k", label="Delta func.")
        ax.plot(mc_Carr21 * np.exp(-sigmas[0]**2), f_PBH_evolved_sigma0, color=comp_color, linestyle="solid", label="LN ($\sigma={:.1f}$), ".format(sigmas[0]) + r"$m_{\rm p}$")
        ax.plot(mc_Carr21 * np.exp(-sigmas[1]**2), f_PBH_evolved_sigma1, color=comp_color, linestyle="dotted", label="LN ($\sigma={:.0f}$), ".format(sigmas[1]) + r"$m_{\rm p}$")
        ax.plot(mc_Carr21, f_PBH_evolved_sigma0, color="b", linestyle="solid", label="LN ($\sigma={:.1f}$), ".format(sigmas[0]) + r"$m_c$")
        ax.plot(mc_Carr21, f_PBH_evolved_sigma1, color="b", linestyle="dotted", label="LN ($\sigma={:.0f}$), ".format(sigmas[1]) + r"$m_c$")
    elif generic_mass:
        ax.plot(m_delta, f_max, color="k", label=r"Delta func., $m_{\rm x}=m$")
        ax.plot(mc_Carr21 * np.exp(-sigmas[0]**2), f_PBH_evolved_sigma0, color=comp_color, linestyle="solid", label="LN ($\sigma={:.1f}$), ".format(sigmas[0]) + r"$m_{\rm x} = m_{\rm p}$")
        ax.plot(mc_Carr21 * np.exp(-sigmas[1]**2), f_PBH_evolved_sigma1, color=comp_color, linestyle="dotted", label="LN ($\sigma={:.0f}$), ".format(sigmas[1]) + r"$m_{\rm x} = m_{\rm p}$")
        ax.plot(mc_Carr21, f_PBH_evolved_sigma0, color="b", linestyle="solid", label="LN ($\sigma={:.1f}$), ".format(sigmas[0]) + r"$m_{\rm x} = m_c$")
        ax.plot(mc_Carr21, f_PBH_evolved_sigma1, color="b", linestyle="dotted", label="LN ($\sigma={:.0f}$), ".format(sigmas[1]) + r"$m_{\rm x} = m_c$")
    elif text_on_plot:
        ax.plot(m_delta, f_max, color="k", label="Delta func.")
        ax.plot(mc_Carr21 * np.exp(-sigmas[0]**2), f_PBH_evolved_sigma0, color=comp_color, linestyle="solid")
        ax.plot(mc_Carr21 * np.exp(-sigmas[1]**2), f_PBH_evolved_sigma1, color=comp_color, linestyle="dotted")
        ax.plot(mc_Carr21, f_PBH_evolved_sigma0, color="b", linestyle="solid")
        ax.plot(mc_Carr21, f_PBH_evolved_sigma1, color="b", linestyle="dotted")
        
        ax.text(2.5e16, 1e-2, "Delta-func.", color="k", rotation=80, fontsize="xx-small")
        ax.text(2.5e17, 1.5e-3, "LN ($\sigma={:.1f}$), ".format(sigmas[0]) + r"against $m_{\rm p}$", color="r", rotation=70, fontsize="xx-small")
        ax.text(2e18, 3e-2, "LN ($\sigma={:.1f}$), ".format(sigmas[1]) + r"against $m_{\rm p}$", color="r", rotation=70, fontsize="xx-small")
        ax.text(1.3e19, 8e-3, "LN ($\sigma={:.1f}$), ".format(sigmas[0]) + r"against $m_{c}$", color="b", rotation=70, fontsize="xx-small")
        ax.text(5e19, 5e-3, "LN ($\sigma={:.1f}$), ".format(sigmas[1]) + r"against $m_{c}$", color="b", rotation=70, fontsize="xx-small")

    ax.set_ylabel(r"$f_{\rm PBH}$")
    ax.set_ylim(1e-3, 1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e16, 5e20)    
   
    x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 10)
    ax.xaxis.set_major_locator(x_major)
    x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax.xaxis.set_minor_locator(x_minor)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.tick_params("x", pad=7, which="both")
    ax.set_ylabel(r"$f_{\rm PBH}$")

    if not text_on_plot:
        ax.legend(fontsize="xx-small")
    if not generic_mass:
        ax.set_xlabel(r"$m~[{\rm g}]$")
    elif generic_mass:
        ax.set_xlabel(r"$m_{\rm x}~[{\rm g}]$")
    fig.tight_layout(pad=0.3)


#%% Version with different values of sigma shown in each figure, with m_c and m_p used as the x-axis.

def mc_to_mp(m_c):
    return m_c * np.exp(-sigma**2)

def mp_to_mc(m_p):
    return m_p * np.exp(sigma**2)

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    exponent_PL_lower = 2
                
    data_folder = "./Data-tests/PL_exp_{:.0f}".format(exponent_PL_lower)
            
    # Plot psi_N, f_max and the integrand in the same figure window
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot delta-function MF constraint from Boudaud & Cirelli (2019)
    prop_A = False
    with_bkg_subtr = True
    prop_B_lower = True
    
    # Load delta-function MF constraint from Boudaud & Cirelli (2019)
    m_delta_BC19_loaded, f_max_BC19_loaded = load_data_Voyager_BC19(Deltas=Deltas, Delta_index=0, prop_A=False, with_bkg_subtr=True, prop_B_lower=True, mf=None)
    m_delta_BC19_extrapolated = 10**np.arange(11, np.log10(min(m_delta_BC19_loaded))+0.01, 0.1)
    f_max_BC19_extrapolated = min(f_max_BC19_loaded) * np.power(m_delta_BC19_extrapolated / min(m_delta_BC19_loaded), exponent_PL_lower)
    f_max_BC19 = np.concatenate((f_max_BC19_extrapolated, f_max_BC19_loaded))
    m_delta_BC19 = np.concatenate((m_delta_BC19_extrapolated, m_delta_BC19_loaded))
    
    # Range of characteristic masses for obtaining constraints
    mc_Carr21 = np.logspace(14, 22, 1000)
    mc_Carr21 = 10**np.arange(14, 22.5, 0.1)
    
    # Calculate delta-function MF constraint from Carr et al. (2021), using the fitting formula for beta prime
    ax.plot(m_delta_BC19, f_max_BC19, color="k")
    ax.plot(m_delta_BC19 * np.exp(-sigma**2), f_max_BC19, color="r")
    
    for i, ax in enumerate(axes.flatten()):
                       
        plotter_BC19(Deltas, 0, ax, color="r", prop_A=prop_A, with_bkg_subtr=with_bkg_subtr, prop_B_lower=prop_B_lower)
        
        ax1 = ax.secondary_xaxis('top', functions=(g_to_Solmass, Solmass_to_g))
        
        # Values of sigma (parameter in log-normal distribution)
        sigmas = [sigmas_LN[-1], 2]
        
        # Line styles to use for results obtained with different sigmas
        alphas = [1, 0.5]
        
        for k, sigma in enumerate(sigmas):
            f_PBH_BC19_evolved = constraint_Carr(mc_Carr21, m_delta_BC19, f_max_BC19, LN, [sigma], evolved=True)
            mp_Carr21 = mc_Carr21 * np.exp(-sigma**2)
            if i == 0:
                ax.plot(mp_Carr21, f_PBH_BC19_evolved, color="r", linestyle="dotted", alpha=alphas[k], label="$\sigma={:.1f}$".format(sigma))
                ax.set_xlabel(r"$m_{\rm p}~[{\rm g}]$")
                ax1.set_xlabel(r"$m_{\rm p}~[M_\odot]$", labelpad=14)
                ax.legend(fontsize="small")
            elif i == 1:
                ax.plot(mc_Carr21, f_PBH_BC19_evolved, color="r", linestyle="dotted", alpha=alphas[k], label="$\sigma={:.1f}$".format(sigma))
                ax.set_xlabel(r"$m_c~[{\rm g}]$")
                ax1.set_xlabel(r"$m_c~[M_\odot]$", labelpad=14)
       
        ax.set_ylabel(r"$f_{\rm PBH}$")
        ax.set_xlim(1e16, 5e20)
        ax.set_ylim(1e-3, 1)
        ax.set_xscale("log")
        ax.set_yscale("log")

        x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 10)
        ax.xaxis.set_major_locator(x_major)
        x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
        ax.xaxis.set_minor_locator(x_minor)
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.tick_params("x", pad=7)
        ax1.tick_params("x")
        
        x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 10)
        ax1.xaxis.set_major_locator(x_major)
        x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
        ax1.xaxis.set_minor_locator(x_minor)
        ax1.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax1.tick_params("x", pad=7)
     
    fig.subplots_adjust(hspace=0.1)
    fig.tight_layout(pad=0.3)


    #%%
if "__main__" == __name__:
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
            ax.tick_params("x", pad=7)
            ax.set_xlim(1e14, 1e17)
            ax.set_xscale("log")
            ax.set_yscale("log")
         
        # Plot the integrand on a linear scale
        fig2, ax4 = plt.subplots(figsize=(6,5))
        ax4.plot(m_delta_KP23, LN(m_delta_KP23, m_c=m_c, sigma=sigma_Carr21)/f_max_KP23, color=(0.5294, 0.3546, 0.7020), label="KP '23")
        ax4.plot(m_delta_Carr21, LN(m_delta_Carr21, m_c=m_c, sigma=sigma_Carr21)/f_max_Carr21, color="k", label="Carr+ '21")
        ax4.legend(fontsize="x-small")
        ax4.set_ylabel("$\psi_\mathrm{N}/f_\mathrm{max}~[\mathrm{g}^{-1}]$")
        ax4.tick_params("x", pad=7)
        ax4.set_xlabel("$m~[\mathrm{g}]$")
        ax4.set_xlim(0, 1e15)
        
        for fig in [fig1, fig2]:
            fig.suptitle("$m_p = {:.1e}".format(m_p) + "~\mathrm{g}$" + ", $\sigma={:.2f}$".format(sigma_Carr21), fontsize="small")
            fig.tight_layout()
        
#%% Plot psi_N, f_max and the integrand in the same figure window (sigma=2 case), using constraints from Auffinger (2022) [2201.01265] Fig. 3 RH panel
if "__main__" == __name__:

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