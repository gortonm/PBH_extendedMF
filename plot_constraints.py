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
from preliminaries import load_data, m_max_SLN, LN, SLN, CC3, constraint_Carr

# Specify the plot style
mpl.rcParams.update({'font.size': 24, 'font.family': 'serif'})
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

#%% Functions


def load_data_KP23(Deltas, Delta_index, mf=None, evolved=True, extrap_lower=True, exponent_PL=2):
    """
    Load extended MF constraints from the delta-function MF constraints 
    obtained using soft gamma-rays from Korwar & Profumo (2023) [2302.04408].

    Parameters
    ----------
    Deltas : Array-like
        Array of power-spectrum widths, which correspond to a given fitting 
        function.
    Delta_index : Integer
        Index of the array Delta corresponding to the desired value of Delta.
    mf : Function, optional
        Fitting function to use. The default is None (delta-function).
    evolved : Boolean, optional
        If True, use the evolved form of the fitting function. The default is 
        True.
    extrap_lower : Boolean, optional
        If True, for a delta-function MF, extrapolate the constraint to masses
        m < 1e16 using a power-law with exponent exponent_PL. The default is
        True.
    exponent_PL : Float, optional
        Denotes the exponent of the power-law used to extrapolate the delta-
        function MF constraint. The default is 2.

    Returns
    -------
    mp_KP23 : Array-like
        Peak masses.
    f_PBH_KP23 : Array-like
        Constraint on f_PBH.

    """

    # Path to extended MF constraints
    if evolved:
        data_folder = "./Data/PL_exp_{:.0f}".format(exponent_PL)
    else:
        data_folder = "./Data/unevolved/PL_exp_{:.0f}".format(exponent_PL)

    # Load data for the appropriate extended mass function (or delta-function MF):
    if mf == None:
        
        m_delta_values_loaded, f_max_loaded = load_data("./2302.04408/2302.04408_MW_diffuse_SPI.csv")

        if extrap_lower:
            m_delta_extrapolated = np.logspace(11, 16, 51)

            f_max_extrapolated = min(f_max_loaded) * np.power(m_delta_extrapolated / min(m_delta_values_loaded), exponent_PL)

            f_PBH_KP23 = np.concatenate((f_max_extrapolated, f_max_loaded))
            mp_KP23 = np.concatenate((m_delta_extrapolated, m_delta_values_loaded))
        else:
            f_PBH_KP23 = f_max_loaded
            mp_KP23 = m_delta_values_loaded

    elif mf == LN:
        data_filename = data_folder + "/LN_2302.04408_Delta={:.1f}_extrap_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL)
        mc_KP23, f_PBH_KP23 = np.genfromtxt(data_filename, delimiter="\t")
        mp_KP23 = mc_KP23 * np.exp(-sigmas_LN[Delta_index]**2)

    elif mf == SLN:
        data_filename = data_folder + "/SLN_2302.04408_Delta={:.1f}_extrap_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL)
        mc_KP23, f_PBH_KP23 = np.genfromtxt(data_filename, delimiter="\t")
        mp_KP23 = [m_max_SLN(m_c, sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index]) for m_c in mc_KP23]

    elif mf == CC3:
        data_filename = data_folder + "/CC3_2302.04408_Delta={:.1f}_extrap_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL)
        mp_KP23, f_PBH_KP23 = np.genfromtxt(data_filename, delimiter="\t")

    return np.array(mp_KP23), np.array(f_PBH_KP23)


def load_data_Voyager_BC19(Deltas, Delta_index, prop_A, with_bkg_subtr, mf=None, evolved=True, exponent_PL=2, prop_B_lower=False):
    """
    Load extended MF constraints from the Voyager 1 delta-function MF 
    constraints obtained by Boudaud & Cirelli (2019) [1807.03075].

    Parameters
    ----------
    Deltas : Array-like
        Array of power-spectrum widths, which correspond to a given fitting 
        function.
    Delta_index : Integer
        Index of the array Delta corresponding to the desired value of Delta.
    prop_A : Boolean
        If True, load constraints obtained using propagation model prop A. If 
        False, load constraints obtained using propagation model prop B.
    with_bkg_subtr : Boolean
        If True, load constraints obtained using background subtraction. If 
        False, load constraints obtained without background subtraction.   
    mf : Function, optional
        Fitting function to use. The default is None (delta-function).
    evolved : Boolean, optional
        If True, use the evolved form of the fitting function. The default is 
        True.
    exponent_PL : Float, optional
        Denotes the exponent of the power-law used to extrapolate the delta-
        function MF constraint. The default is 2.
        
    Returns
    -------
    mp_BC19 : Array-like
        Peak masses.
    f_PBH_BC19 : Array-like
        Constraint on f_PBH.

    """

    if evolved:
        data_folder = "./Data/PL_exp_{:.0f}".format(exponent_PL)
    else:
        data_folder = "./Data/unevolved/PL_exp_{:.0f}".format(exponent_PL)

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

    if not prop_A:
        if prop_B_lower:
            prop_string += "_lower"
        else:
            prop_string += "_upper"

    if mf == None:
        mp_BC19, f_PBH_BC19 = load_data("1807.03075/1807.03075_" + prop_string + ".csv")

    elif mf == LN:
        data_filename = data_folder + "/LN_1807.03075_" + prop_string + "_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL)
        mc_BC19, f_PBH_BC19 = np.genfromtxt(data_filename, delimiter="\t")
        mp_BC19 = mc_BC19 * np.exp(-sigmas_LN[Delta_index]**2)

    elif mf == SLN:
        data_filename = data_folder + "/SLN_1807.03075_" + prop_string + "_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL)
        mc_BC19, f_PBH_BC19 = np.genfromtxt(data_filename, delimiter="\t")
        mp_BC19 = [m_max_SLN(m_c, sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index]) for m_c in mc_BC19]

    elif mf == CC3:
        data_filename = data_folder + "/CC3_1807.03075_" + prop_string + "_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[Delta_index], exponent_PL)
        mp_BC19, f_PBH_BC19 = np.genfromtxt(data_filename, delimiter="\t")

    return np.array(mp_BC19), np.array(f_PBH_BC19)


def load_data_Subaru_Croon20(Deltas, Delta_index, mf=None, evolved=False):
    """
    Load extended MF constraints from the Subaru-HSC delta-function MF 
    constraints obtained by Croon et al. (2020) [2007.12697].

    Parameters
    ----------
    Deltas : Array-like
        Array of power-spectrum widths, which correspond to a given fitting 
        function.
    Delta_index : Integer
        Index of the array Delta corresponding to the desired value of Delta.
    mf : Function, optional
        Fitting function to use. The default is None (delta-function).
    evolved : Boolean, optional
        If True, use the evolved form of the fitting function. The default is 
        False.

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
        if evolved:
            mc_Subaru, f_PBH_Subaru = np.genfromtxt("./Data/LN_HSC_Delta={:.1f}_evolved.txt".format(Deltas[Delta_index]), delimiter="\t")
        else:
            mc_Subaru, f_PBH_Subaru = np.genfromtxt("./Data/LN_HSC_Delta={:.1f}.txt".format(Deltas[Delta_index]), delimiter="\t")

        mp_Subaru = mc_Subaru * np.exp(-sigmas_LN[Delta_index]**2)

    elif mf == SLN:
        if evolved:
            mc_Subaru, f_PBH_Subaru = np.genfromtxt("./Data/SLN_HSC_Delta={:.1f}_evolved.txt".format(Deltas[Delta_index]), delimiter="\t")
        else:
            mc_Subaru, f_PBH_Subaru = np.genfromtxt("./Data/SLN_HSC_Delta={:.1f}.txt".format(Deltas[Delta_index]), delimiter="\t")
        mp_Subaru = [m_max_SLN(m_c, sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index]) for m_c in mc_Subaru]

    elif mf == CC3:
        if evolved:
            mp_Subaru, f_PBH_Subaru = np.genfromtxt("./Data/CC3_HSC_Delta={:.1f}_evolved.txt".format(Deltas[Delta_index]), delimiter="\t")
        else:
            mp_Subaru, f_PBH_Subaru = np.genfromtxt("./Data/CC3_HSC_Delta={:.1f}.txt".format(Deltas[Delta_index]), delimiter="\t")

    return np.array(mp_Subaru), np.array(f_PBH_Subaru)


def load_data_GECCO(Deltas, Delta_index, mf=None, exponent_PL=2, evolved=True, NFW=True):
    """
    Load extended MF constraints from the prospective GECCO delta-function MF 
    constraints from Coogan et al. (2023) [2101.10370].

    Parameters
    ----------
    Deltas : Array-like
        Array of power-spectrum widths, which correspond to a given fitting 
        function.
    Delta_index : Integer
        Index of the array Delta corresponding to the desired value of Delta.
    mf : Function, optional
        Fitting function to use. The default is None (delta-function).
    evolved : Boolean, optional
        If True, use the evolved form of the fitting function. The default is 
        True.
    exponent_PL : Float, optional
        Denotes the exponent of the power-law used to extrapolate the delta-
        function MF constraint. The default is 2.
    NFW : Boolean, optional
        If True, load constraints obtained using an NFW profile. If False, load 
        constraints obtained using an Einasto profile.   

    Returns
    -------
    mp_GECCO : Array-like
        Peak masses.
    f_PBH_GECCO : Array-like
        Constraint on f_PBH.

    """
    if evolved:
        data_folder = "./Data/PL_exp_{:.0f}".format(exponent_PL)
    else:
        data_folder = "./Data/unevolved/PL_exp_{:.0f}".format(
            exponent_PL)

    if NFW:
        density_string = "NFW"
    else:
        density_string = "Einasto"

    if mf == None:
        mp_GECCO, f_PBH_GECCO = load_data(
            "2101.01370/2101.01370_Fig9_GC_%s.csv" % density_string)

    elif mf == LN:
        mc_GECCO, f_PBH_GECCO = np.genfromtxt(data_folder + "/LN_2101.01370_Delta={:.1f}_".format(
            Deltas[Delta_index]) + "%s" % density_string + "_extrapolated_exp{:.0f}.txt".format(exponent_PL))
        mp_GECCO = mc_GECCO * np.exp(-sigmas_LN[Delta_index]**2)

    elif mf == SLN:
        mc_GECCO, f_PBH_GECCO = np.genfromtxt(data_folder + "/SLN_2101.01370_Delta={:.1f}_".format(
            Deltas[Delta_index]) + "%s" % density_string + "_extrapolated_exp{:.0f}.txt".format(exponent_PL))
        mp_GECCO = [m_max_SLN(m_c, sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index],
                              log_m_factor=3, n_steps=1000) for m_c in mc_GECCO]

    elif mf == CC3:
        mp_GECCO, f_PBH_GECCO = np.genfromtxt(data_folder + "/CC3_2101.01370_Delta={:.1f}_".format(
            Deltas[Delta_index]) + "%s" % density_string + "_extrapolated_exp{:.0f}.txt".format(exponent_PL))

    return np.array(mp_GECCO), np.array(f_PBH_GECCO)


def load_data_Sugiyama(Deltas, Delta_index, mf=None):
    """
    Load extended MF constraints from the prospective white dwarf microlensing 
    delta-function MF constraints from Sugiyama et al. (2020) [1905.06066].

    Parameters
    ----------
    Deltas : Array-like
        Array of power-spectrum widths, which correspond to a given fitting 
        function.
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
        mc_Sugiyama, f_PBH_Sugiyama = np.genfromtxt("./Data/LN_Sugiyama20_Delta={:.1f}.txt".format(Deltas[Delta_index]), delimiter="\t")
        mp_Sugiyama = mc_Sugiyama * np.exp(-sigmas_LN[Delta_index]**2)

    elif mf == SLN:
        mc_Sugiyama, f_PBH_Sugiyama = np.genfromtxt("./Data/SLN_Sugiyama20_Delta={:.1f}.txt".format(Deltas[Delta_index]), delimiter="\t")
        mp_Sugiyama = [m_max_SLN(m_c, sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index]) for m_c in mc_Sugiyama]

    elif mf == CC3:
        mp_Sugiyama, f_PBH_Sugiyama = np.genfromtxt("./Data/CC3_Sugiyama20_Delta={:.1f}.txt".format(Deltas[Delta_index]), delimiter="\t")

    return np.array(mp_Sugiyama), np.array(f_PBH_Sugiyama)


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

    x_major = mpl.ticker.LogLocator(base=10.0, numticks=5)
    ax.xaxis.set_major_locator(x_major)
    x_minor = mpl.ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=5)
    ax.xaxis.set_minor_locator(x_minor)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    y_major = mpl.ticker.LogLocator(base=10.0, numticks=10)
    ax.yaxis.set_major_locator(y_major)
    y_minor = mpl.ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)
    ax.yaxis.set_minor_locator(y_minor)
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    ax.grid()


def plotter_BC19(Deltas, Delta_index, ax, color, prop_A, with_bkg_subtr, mf=None, exponent_PL=2, evolved=True, prop_B_lower=True, linestyle="solid", linewidth=1, marker=None, alpha=1):
    """
    Plot extended MF constraints from from the Voyager 1 delta-function MF 
    constraints obtained by Boudaud & Cirelli (2019) [1807.03075].    

    Parameters
    ----------
    Deltas : Array-like
        Array of power-spectrum widths, which correspond to a given fitting 
        function.
    Delta_index : Integer
        Index of the array Delta corresponding to the desired value of Delta.
    ax : Matplotlib Axes object
        Axis to add ticks and grid to.
    color : String
        Color to use for plotting.
    prop_A : Boolean
        If True, load constraints obtained using propagation model prop A. If 
        False, load constraints obtained using propagation model prop B.
    with_bkg_subtr : Boolean
        If True, load constraints obtained using background subtraction. If 
        False, load constraints obtained without background subtraction.   
    mf : Function, optional
        Fitting function to use. The default is None (delta-function).
    exponent_PL : Float, optional
        Denotes the exponent of the power-law used to extrapolate the delta-
        function MF. The default is 2.
    evolved : Boolean, optional
        If True, use the evolved form of the fitting function. The default is 
        True.
    linestyle : String, optional
        Linestyle to use for plotting. The default is "solid".
    linewidth : Float, optional
        Line width to use for plotting. The default is 1.
    marker : String, optional
        Marker to use for plotting. The default is None.
    alpha : Float, optional
        Transparency of the line. The default is 1 (zero transparency).

    Returns
    -------
    None.

    """
    mp, f_PBH = load_data_Voyager_BC19(Deltas, Delta_index, prop_A, with_bkg_subtr, mf, evolved, exponent_PL, prop_B_lower)
    ax.plot(mp, f_PBH, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, marker=marker)


def plotter_KP23(Deltas, Delta_index, ax, color, mf=None, extrap_lower=True, exponent_PL=2, evolved=True, linestyle="solid", linewidth=1, marker=None, alpha=1):
    """
    Plot extended MF constraints from the delta-function MF constraints 
    obtained by Korwar & Profumo (2023) [2302.04408].    

    Parameters
    ----------
    Deltas : Array-like
        Array of power-spectrum widths, which correspond to a given fitting 
        function.
    Delta_index : Integer
        Index of the array Delta corresponding to the desired value of Delta.
    ax : Matplotlib Axes object
        Axis to add ticks and grid to.
    color : String
        Color to use for plotting.
    mf : Function, optional
        Fitting function to use. The default is None (delta-function).
    extrap_lower : Boolean, optional
        If True, for a delta-function MF, extrapolate the constraint to masses
        m < 1e16 using a power-law with exponent exponent_PL. The default is
        True.
    exponent_PL : Float, optional
        Denotes the exponent of the power-law used to extrapolate the delta-
        function MF. The default is 2.
    evolved : Boolean, optional
        If True, use the evolved form of the fitting function. The default is 
        True.
    linestyle : String, optional
        Linestyle to use for plotting. The default is "solid".
    linewidth : Float, optional
        Line width to use for plotting. The default is 1.
    marker : String, optional
        Marker to use for plotting. The default is None.
    alpha : Float, optional
        Transparency of the line. The default is 1 (zero transparency).

    Returns
    -------
    None.

    """

    mp, f_PBH = load_data_KP23(Deltas, Delta_index, mf, evolved, extrap_lower, exponent_PL)
    ax.plot(mp, f_PBH, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, marker=marker)


def plotter_Subaru_Croon20(Deltas, Delta_index, ax, color, mf=None, evolved=False, normalised=True, n=1, show_label=True, linestyle="solid", linewidth=1, marker=None, alpha=1):
    """
    Plot extended MF constraints from the Subaru-HSC delta-function MF constraints obtained by Croon et al. (2020) [2007.12697].    

    Parameters
    ----------
    Deltas : Array-like
        Array of power-spectrum widths, which correspond to a given fitting 
        function.
    Delta_index : Integer
        Index of the array Delta corresponding to the desired value of Delta.
    ax : Matplotlib Axes object
        Axis to add ticks and grid to.
    color : String
        Color to use for plotting.
    mf : Function, optional
        Fitting function to use. The default is None (delta-function).
    evolved : Boolean, optional
        If True, use the evolved form of the fitting function. The default is 
        False.
    linestyle : String, optional
        Linestyle to use for plotting. The default is "solid".
    linewidth : Float, optional
        Line width to use for plotting. The default is 1.
    marker : String, optional
        Marker to use for plotting. The default is None.
    alpha : Float, optional
        Transparency of the line. The default is 1 (zero transparency).

    Returns
    -------
    None.

    """
    
    mp_Subaru, f_PBH_Subaru = load_data_Subaru_Croon20(Deltas, Delta_index, mf, evolved)
    ax.plot(mp_Subaru, f_PBH_Subaru, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha, marker=marker)


def plotter_GECCO(Deltas, Delta_index, ax, color, mf=None, exponent_PL=2, evolved=True, NFW=True, show_label=False, linestyle="solid", linewidth=1, marker=None, alpha=1):
    """
    Plot extended MF constraints from the prospective GECCO delta-function MF 
    constraints from Coogan et al. (2023) [2101.10370].    

    Parameters
    ----------
    Deltas : Array-like
        Array of power-spectrum widths, which correspond to a given fitting 
        function.
    Delta_index : Integer
        Index of the array Delta corresponding to the desired value of Delta.
    ax : Matplotlib Axes object
        Axis to add ticks and grid to.
    color : String
        Color to use for plotting.
    mf : Function, optional
        Fitting function to use. The default is None (delta-function).
    exponent_PL : Float, optional
        Denotes the exponent of the power-law used to extrapolate the delta-
        function MF. The default is 2.
    evolved : Boolean, optional
        If True, use the evolved form of the fitting function. The default is 
        True.
    NFW : Boolean, optional
        If True, load constraints obtained using an NFW profile. If False, load 
        constraints obtained using an Einasto profile.   
    linestyle : String, optional
        Linestyle to use for plotting. The default is "solid".
    linewidth : Float, optional
        Line width to use for plotting. The default is 1.
    marker : String, optional
        Marker to use for plotting. The default is None.
    alpha : Float, optional
        Transparency of the line. The default is 1 (zero transparency).

    Returns
    -------
    None.

    """

    mp_GECCO, f_PBH_GECCO = load_data_GECCO(Deltas, Delta_index, mf, exponent_PL=exponent_PL, evolved=evolved, NFW=NFW)
    ax.plot(mp_GECCO, f_PBH_GECCO, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha, marker=marker)


def plotter_Sugiyama(Deltas, Delta_index, ax, color, mf=None, show_label=True, linestyle="solid", linewidth=1, marker=None, alpha=1):
    """
    Plot extended MF constraints from the prospective white dwarf microlensing 
    delta-function MF constraints from Sugiyama et al. (2020) [1905.06066].

    Parameters
    ----------
    Deltas : Array-like
        Array of power-spectrum widths, which correspond to a given fitting 
        function.
    Delta_index : Integer
        Index of the array Delta corresponding to the desired value of Delta.
    ax : Matplotlib Axes object
        Axis to add ticks and grid to.
    color : String
        Color to use for plotting.
    mf : Function, optional
        Fitting function to use. The default is None (delta-function).
    linestyle : String, optional
        Linestyle to use for plotting. The default is "solid".
    linewidth : Float, optional
        Line width to use for plotting. The default is 1.
    marker : String, optional
        Marker to use for plotting. The default is None.
    alpha : Float, optional
        Transparency of the line. The default is 1 (zero transparency).

    Returns
    -------
    None.

    """
    
    mp_Sugiyama, f_PBH_Sugiyama = load_data_Sugiyama(Deltas, Delta_index, mf)
    ax.plot(mp_Sugiyama, f_PBH_Sugiyama, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha, marker=marker)
    
    
def Solmass_to_g(m):
    """Convert a mass m (in solar masses) to grams."""
    return 1.989e33 * m


def g_to_Solmass(m):
    """Convert a mass m (in grams) to solar masses."""
    return m / 1.989e33


# %% Plot all delta-function MF constraints (see Fig. 1 of paper / Fig. 3.1 of thesis)

if "__main__" == __name__:
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3,
        betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)

    plot_existing = True    # If True, plot existing constraints
    plot_prospective = True    # If True, plot prospective future constraints
    paper_only = True    # If True, plot only the constraints used in the paper

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    Delta_index = 0

    if plot_existing:

        mp_propA, f_PBH_propA = load_data_Voyager_BC19(Deltas, Delta_index, prop_A=True, with_bkg_subtr=True, mf=None)
        ax.plot(mp_propA, f_PBH_propA, color="r")
        #mp_propB, f_PBH_propB = load_data_Voyager_BC19(Deltas, Delta_index, prop_A=False, with_bkg_subtr=True, prop_B_lower=True, mf=None)
        #ax.plot(mp_propB, f_PBH_propB, color="r", linestyle="dashdot")
        
        plotter_BC19(Deltas[Delta_index], Delta_index, ax, color="r", prop_A=True, with_bkg_subtr=True, mf=None)
        plotter_BC19(Deltas[Delta_index], Delta_index, ax, color="r", prop_A=True, with_bkg_subtr=True, mf=None, linestyle="dashdot")

        ax.text(1.5e15, 0.1, "Electrons/\npositrons \n (Voyager 1)", fontsize="xx-small", color="r")
        #ax.text(7e15, 0.1, "Electrons/\npositrons \n (Voyager 1)", fontsize="xx-small", color="r")
        
        plotter_KP23(Deltas, Delta_index, ax, color="orange", extrap_lower=False)
        ax.text(7e16, 0.0001, "MeV gamma rays \n (INTEGRAL/SPI)", fontsize="xx-small", color="orange")

        plotter_Subaru_Croon20(Deltas, Delta_index, ax, color="tab:grey")
        ax.text(2.5e22, 0.4, "Subaru-HSC",fontsize="xx-small", color="tab:grey")
            
    if plot_prospective:
        plotter_GECCO(Deltas, Delta_index, ax,color="#5F9ED1", linestyle="dotted", NFW=True)
        ax.text(3e17, 0.005, "MeV gamma rays \n (future)",fontsize="xx-small", color="#5F9ED1")

        plotter_Sugiyama(Deltas, Delta_index, ax, color="k", linestyle="dotted")
        ax.text(1e21, 0.002, "WD microlensing", fontsize="xx-small", color="k")
                
    ax.tick_params("x", pad=7)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylabel("$f_\mathrm{PBH}$")
    #ax.set_xlabel("$m~[\mathrm{g}]$")
    ax.set_xlabel(r"$M_{\rm PBH}~[\mathrm{g}]$")  # for thesis
    ax.set_ylim(1e-5, 1)
    ax.set_xlim(1e15, 1e24)

    ax1 = ax.secondary_xaxis('top', functions=(g_to_Solmass, Solmass_to_g))
    #ax1.set_xlabel("$m~[M_\odot]$", labelpad=14)
    ax1.set_xlabel(r"$M_{\rm PBH}~[M_\odot]$", labelpad=14)  # for thesis
    ax1.tick_params("x")

    ax2 = ax.secondary_yaxis('right')
    ax2.set_yticklabels([])

    fig.tight_layout(pad=0.1)
    plt.savefig("all_delta_func_labelled_modelA_NFW.pdf")
    
# %% Existing constraints (see Fig. 2 of paper / Fig. 3.2 of thesis)

if "__main__" == __name__:

    # If True, plot the "prop A" constraint. Otherwise, plot the "prop B" constraint
    prop_A = True
    # If True, plot the more stringent "prop B" constraint
    prop_B_lower = False
    
    with_bkg_subtr = True

    colors = ['r', 'b', 'orange', 'tab:grey']
    linestyles = ['solid', 'dotted', 'dashdot', 'dashed']

    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)

    Delta_indices = [0, 5, 6]

    linewidth = 1

    if len(Delta_indices) == 3:   # for thesis
        plt.figure(figsize=(10, 12))   # for thesis
        ax = plt.subplot(2, 2, 1)   # for thesis

    elif len(Delta_indices) == 3:
        plt.figure(figsize=(14, 5.5))
        ax = plt.subplot(1, 3, 1)
        
    for axis_index, Delta_index in enumerate(Delta_indices):
        
        if len(Delta_indices) == 3:
            ax = plt.subplot(2, 2, axis_index + 1, sharex=ax)
        elif len(Delta_indices) == 3:
            ax = plt.subplot(1, 3, axis_index + 1, sharex=ax)

        plotter_KP23(Deltas, Delta_index, ax, color=colors[2], linestyle=linestyles[0], linewidth=linewidth)
        plotter_KP23(Deltas, Delta_index, ax, color=colors[2], mf=LN, linestyle=linestyles[1], linewidth=linewidth)
        plotter_KP23(Deltas, Delta_index, ax, color=colors[2], mf=SLN, linestyle=linestyles[2], linewidth=linewidth)
        plotter_KP23(Deltas, Delta_index, ax, color=colors[2], mf=CC3, linestyle=linestyles[3], linewidth=linewidth)

        plotter_BC19(Deltas, Delta_index, ax, color=colors[0], mf=None, prop_A=prop_A, with_bkg_subtr=with_bkg_subtr, prop_B_lower=prop_B_lower)
        plotter_BC19(Deltas, Delta_index, ax, color=colors[0], mf=LN, prop_A=prop_A, with_bkg_subtr=with_bkg_subtr, prop_B_lower=prop_B_lower, linestyle=linestyles[1], linewidth=linewidth)
        plotter_BC19(Deltas, Delta_index, ax, color=colors[0], mf=SLN, prop_A=prop_A, with_bkg_subtr=with_bkg_subtr, prop_B_lower=prop_B_lower, linestyle=linestyles[2], linewidth=linewidth)
        plotter_BC19(Deltas, Delta_index, ax, color=colors[0], mf=CC3, prop_A=prop_A, with_bkg_subtr=with_bkg_subtr, prop_B_lower=prop_B_lower, linestyle=linestyles[3], linewidth=linewidth)
                
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
        ax.plot(0, 0, color="k", linestyle=linestyles[3], label="GCC")

        ax.tick_params("x", pad=7)
        #ax.set_xlabel("$m_\mathrm{p}~[\mathrm{g}]$")
        ax.set_xlabel("$M_\mathrm{p}~[\mathrm{g}]$")   # for thesis
        ax.set_ylabel("$f_\mathrm{PBH}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        ax1 = ax.secondary_xaxis('top', functions=(g_to_Solmass, Solmass_to_g))
        #ax1.set_xlabel("$m_\mathrm{p}~[M_\odot]$", labelpad=14)
        ax1.set_xlabel("$M_\mathrm{p}~[M_\odot]$", labelpad=14)   # for thesis
        ax1.tick_params("x")

        ax2 = ax.secondary_yaxis('right')
        ax2.set_yticklabels([])

        if Deltas[Delta_index] == 5:
            ax.legend(loc=[1.4, 0.3])   # for thesis
        
        ax.set_title("$\Delta={:.0f}$".format(Deltas[Delta_index]), pad=25)

    #plt.tight_layout(pad=0.1)
    #plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(wspace=0.35, hspace=0.9)   # for thesis
    plt.savefig("constraints_existing.pdf")
  
# %% Existing constraints (talk version)
# Version showing constraints obtained using different observations in different colours, and using line style to distinguish between fitting functions.

if "__main__" == __name__:

    # If True, plot the "prop A" constraint. Otherwise, plot the "prop B" constraint
    prop_A = True
    # If True, plot the more stringent "prop B" constraint
    prop_B_lower = False
    
    # If True, plot the constraints for a lognormal MF
    plot_LN = True
    
    with_bkg_subtr = True

    colors = ['r', 'b', 'orange', 'tab:grey']
    linestyles = ['solid', 'dotted', 'dashdot', 'dashed']

    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)

    Delta_index = 6

    linewidth = 1
    
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    plotter_KP23(Deltas, Delta_index, ax, color=colors[2], linestyle=linestyles[0], linewidth=linewidth)
    if plot_LN:
        plotter_KP23(Deltas, Delta_index, ax, color=colors[2], mf=LN, linestyle=linestyles[1], linewidth=linewidth)
    plotter_KP23(Deltas, Delta_index, ax, color=colors[2], mf=SLN, linestyle=linestyles[2], linewidth=linewidth)
    #plotter_KP23(Deltas, Delta_index, ax, color=colors[2], mf=CC3, linestyle=linestyles[3], linewidth=linewidth)

    plotter_BC19(Deltas, Delta_index, ax, color=colors[0], mf=None, prop_A=prop_A, with_bkg_subtr=with_bkg_subtr, prop_B_lower=prop_B_lower)
    if plot_LN:
        plotter_BC19(Deltas, Delta_index, ax, color=colors[0], mf=LN, prop_A=prop_A, with_bkg_subtr=with_bkg_subtr, prop_B_lower=prop_B_lower, linestyle=linestyles[1], linewidth=linewidth)
    plotter_BC19(Deltas, Delta_index, ax, color=colors[0], mf=SLN, prop_A=prop_A, with_bkg_subtr=with_bkg_subtr, prop_B_lower=prop_B_lower, linestyle=linestyles[2], linewidth=linewidth)
    #plotter_BC19(Deltas, Delta_index, ax, color=colors[0], mf=CC3, prop_A=prop_A, with_bkg_subtr=with_bkg_subtr, prop_B_lower=prop_B_lower, linestyle=linestyles[3], linewidth=linewidth)
            
    xmin, xmax = 1e16, 5e23
    ymin, ymax = 1e-3, 1

    show_label_Subaru = False

    # Plot Subaru-HSC constraints
    plotter_Subaru_Croon20(Deltas, Delta_index, ax, color=colors[3], linestyle=linestyles[0], linewidth=linewidth, show_label=show_label_Subaru)
    if plot_LN:
        plotter_Subaru_Croon20(Deltas, Delta_index, ax, color=colors[3], mf=LN,  linestyle=linestyles[1], linewidth=linewidth, show_label=show_label_Subaru)
    plotter_Subaru_Croon20(Deltas, Delta_index, ax, color=colors[3], mf=SLN, linestyle=linestyles[2], linewidth=linewidth, show_label=show_label_Subaru)
    #plotter_Subaru_Croon20(Deltas, Delta_index, ax, color=colors[3], mf=CC3, linestyle=linestyles[3], linewidth=linewidth, show_label=show_label_Subaru)

    ax.plot(0, 0, color="k", linestyle=linestyles[0], label="Delta func.")
    if plot_LN:
        ax.plot(0, 0, color="k", linestyle=linestyles[1], label="LN")
    ax.plot(0, 0, color="k", linestyle=linestyles[2], label="SLN")
    #ax.plot(0, 0, color="k", linestyle=linestyles[3], label="GCC")

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

    ax.legend(fontsize="xx-small", loc="lower center")
    
    plt.tight_layout(pad=0.1)


# %% Prospective constraints (see Fig. 3 of paper / Fig. 3.3 of thesis)

if "__main__" == __name__:

    colors = ['tab:blue', 'k']
    linestyles = ['solid', 'dotted', 'dashdot', 'dashed']

    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3,
        betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)

    Delta_indices = [0, 5, 6]

    if len(Delta_indices) == 3:   # for thesis
        plt.figure(figsize=(10, 12))   # for thesis
        ax = plt.subplot(2, 2, 1)   # for thesis

    elif len(Delta_indices) == 3:
        plt.figure(figsize=(14, 5.5))
        ax = plt.subplot(1, 3, 1)

    for axis_index, Delta_index in enumerate(Delta_indices):

        if len(Delta_indices) == 3:   # for thesis
            ax = plt.subplot(2, 2, axis_index + 1, sharex=ax)
        elif len(Delta_indices) == 3:
            ax = plt.subplot(1, 3, axis_index + 1, sharex=ax)

        # Plot prospective extended MF constraints from the white dwarf microlensing survey proposed in Sugiyama et al. (2020) [1905.06066].
        NFW = True
        show_label = False
        prop_A = True
        with_bkg_subtr=True
        prop_B_lower=False

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

        plotter_BC19(Deltas, Delta_index, ax, color="r", mf=None, prop_A=prop_A, with_bkg_subtr=with_bkg_subtr, prop_B_lower=prop_B_lower)
        plotter_BC19(Deltas, Delta_index, ax, color="r", mf=LN, prop_A=prop_A, with_bkg_subtr=with_bkg_subtr, prop_B_lower=prop_B_lower, linestyle=linestyles[1])
        plotter_BC19(Deltas, Delta_index, ax, color="r", mf=SLN, prop_A=prop_A, with_bkg_subtr=with_bkg_subtr, prop_B_lower=prop_B_lower, linestyle=linestyles[2])
        plotter_BC19(Deltas, Delta_index, ax, color="r", mf=CC3, prop_A=prop_A, with_bkg_subtr=with_bkg_subtr, prop_B_lower=prop_B_lower, linestyle=linestyles[3])

        ax.plot(0, 0, color="k", linestyle=linestyles[0], label="Delta func.")
        ax.plot(0, 0, color="k", linestyle=linestyles[1], label="LN")
        ax.plot(0, 0, color="k", linestyle=linestyles[2], label="SLN")
        ax.plot(0, 0, color="k", linestyle=linestyles[3], label="GCC")

        ax.tick_params("x", pad=7)
        #ax.set_xlabel("$m_\mathrm{p}~[\mathrm{g}]$")
        ax.set_xlabel("$M_\mathrm{p}~[\mathrm{g}]$", labelpad=14)   # thesis version
        ax.set_ylabel("$f_\mathrm{PBH}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        ax1 = ax.secondary_xaxis('top', functions=(g_to_Solmass, Solmass_to_g))
        #ax1.set_xlabel("$m_\mathrm{p}~[M_\odot]$", labelpad=14)
        ax1.set_xlabel("$M_\mathrm{p}~[M_\odot]$", labelpad=14)   # thesis version
        ax1.tick_params("x")

        ax2 = ax.secondary_yaxis('right')
        ax2.set_yticklabels([])
        """
        if Deltas[Delta_index] in (1, 2):
            ax.legend(fontsize="xx-small", loc=[0.21, 0.05])
        """
        if Deltas[Delta_index] == 5:
            ax.legend(loc=[1.4, 0.3])   # for thesis
        
        ax.set_title("$\Delta={:.0f}$".format(Deltas[Delta_index]), pad=25)

    #plt.tight_layout(pad=0.1)
    #plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(wspace=0.35, hspace=0.9)   # for thesis
    plt.savefig("constraints_prospective.pdf")


# %% Prospective constraints (talk version)
# Version showing constraints obtained using different observations in different colours, and using line style to distinguish between fitting functions.

if "__main__" == __name__:
    
    # If True, plot the constraints for a lognormal MF
    plot_LN = False

    colors = ['tab:blue', 'k']
    linestyles = ['solid', 'dotted', 'dashdot', 'dashed']

    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3,
        betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)

    Delta_index = 6
    
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # Plot prospective extended MF constraints from the white dwarf microlensing survey proposed in Sugiyama et al. (2020) [1905.06066].
    NFW = True
    show_label = False
    prop_A = True
    with_bkg_subtr=True
    prop_B_lower=False

    # Set axis limits
    xmin, xmax = 1e16, 5e23
    ymin, ymax = 1e-3, 1

    # plot Einasto profile results
    plotter_GECCO(Deltas, Delta_index, ax, color=colors[0], NFW=NFW, linestyle=linestyles[0])
    if plot_LN:
        plotter_GECCO(Deltas, Delta_index, ax, color=colors[0], NFW=NFW, mf=LN, linestyle=linestyles[1])
    #plotter_GECCO(Deltas, Delta_index, ax, color=colors[0], NFW=NFW, mf=CC3, linestyle=linestyles[3])
    plotter_GECCO(Deltas, Delta_index, ax, color=colors[0], NFW=NFW, mf=SLN, linestyle=linestyles[2])

    plotter_Sugiyama(Deltas, Delta_index, ax, color=colors[1], linestyle=linestyles[0], show_label=show_label)
    plotter_Sugiyama(Deltas, Delta_index, ax, color=colors[1], mf=SLN, linestyle=linestyles[2], show_label=show_label)
    if plot_LN:
        plotter_Sugiyama(Deltas, Delta_index, ax, color=colors[1], mf=LN, linestyle=linestyles[1], show_label=show_label)
    #plotter_Sugiyama(Deltas, Delta_index, ax, color=colors[1], mf=CC3, linestyle=linestyles[3], show_label=show_label)

    plotter_BC19(Deltas, Delta_index, ax, color="r", mf=None, prop_A=prop_A, with_bkg_subtr=with_bkg_subtr, prop_B_lower=prop_B_lower)
    plotter_BC19(Deltas, Delta_index, ax, color="r", mf=SLN, prop_A=prop_A, with_bkg_subtr=with_bkg_subtr, prop_B_lower=prop_B_lower, linestyle=linestyles[2])
    if plot_LN:
        plotter_BC19(Deltas, Delta_index, ax, color="r", mf=LN, prop_A=prop_A, with_bkg_subtr=with_bkg_subtr, prop_B_lower=prop_B_lower, linestyle=linestyles[1])
    #plotter_BC19(Deltas, Delta_index, ax, color="r", mf=CC3, prop_A=prop_A, with_bkg_subtr=with_bkg_subtr, prop_B_lower=prop_B_lower, linestyle=linestyles[3])

    ax.plot(0, 0, color="k", linestyle=linestyles[0], label="Delta func.")
    if plot_LN:
        ax.plot(0, 0, color="k", linestyle=linestyles[1], label="LN")
    ax.plot(0, 0, color="k", linestyle=linestyles[2], label="SLN")
    #ax.plot(0, 0, color="k", linestyle=linestyles[3], label="GCC")

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

    ax.legend(fontsize="xx-small", loc=[0.25, 0.05])
    plt.tight_layout(pad=0.1)
    
# %% Plot the constraints shown in Fig. 20 of Carr et al. (2021) [2002.12778] (Fig. 3.4 of thesis)

if "__main__" == __name__:

    fig, ax = plt.subplots(figsize=(5.5, 5.7))
    ax1 = ax.secondary_xaxis('top', functions=(g_to_Solmass, Solmass_to_g))
    
    m_delta_values_loaded, f_max_loaded = load_data("./2002.12778/Carr+21_mono_RH.csv")
    mc_LN_values_loaded, f_PBH_loaded = load_data("./2002.12778/Carr+21_Gamma_ray_LN_RH.csv")
    
    ax.plot(0, 0, color="k", label="Delta func.")
    ax.plot(0, 0, color="k", linestyle="dotted", label=r"LN ($\sigma=2$), $M_{\rm c}$")
    
    ax.plot(m_delta_values_loaded * 1.989e33, f_max_loaded,color="purple")
    ax.plot(mc_LN_values_loaded * 1.989e33, f_PBH_loaded, color="purple", linestyle="dotted")

    # Microlensing constraints from Smyth & Profumo (2019)
    
    m_delta_values_Fig20, f_max_Fig20 = load_data("2002.12778/Subaru-HSC_2002.12778_mono.csv")
    mc_values_Fig20, f_PBH_Fig20 = load_data("2002.12778/Subaru-HSC_2002.12778_LN.csv")    
    ax.plot(m_delta_values_Fig20 * 1.989e33, f_max_Fig20, color="b")
    ax.plot(mc_values_Fig20 * 1.989e33, f_PBH_Fig20, color="b", linestyle="dotted")
   
    ax.set_ylabel("$f_\mathrm{PBH}$")
    ax.set_xlabel(r"$M \, [\mathrm{g}]$", labelpad=14)
    ax1.set_xlabel(r"$M \, [M_\odot]$", labelpad=14)
    ax.legend(fontsize="xx-small")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e16, 2e25)
    ax.set_ylim(1e-3, 1)
    ax.tick_params(pad=7, which="both")
    fig.tight_layout(pad=0.3)
    plt.savefig("Carr+_Fig20_reproduction.pdf")


# %% Constraints for a lognormal against peak mass and M_c (Fig. 4 of paper / Fig. 3.5 of thesis)

if "__main__" == __name__:

    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3,
        betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)

    exponent_PL = 2

    data_folder = "./Data/PL_exp_{:.0f}".format(exponent_PL)

    original = True

    fig, ax = plt.subplots(figsize=(7, 5))

    plot_BC19 = True
    plot_KP23 = False

    if plot_BC19:

        comp_color = "r"

        # Plot delta-function MF constraint from Boudaud & Cirelli (2019)
        prop_A = True
        with_bkg_subtr = True
        prop_B_lower = False

        # Load delta-function MF constraint from Boudaud & Cirelli (2019)
        m_delta_loaded, f_max_loaded = load_data_Voyager_BC19(Deltas=Deltas, Delta_index=0, prop_A=prop_A, with_bkg_subtr=with_bkg_subtr, prop_B_lower=prop_B_lower, mf=None)

    elif plot_KP23:

        comp_color = "orange"

        # Power-law exponent to use between 1e11g and 1e16g.
        exponent_PL = 2.0

        # Load delta-function MF constraint from Boudaud & Cirelli (2019)
        m_delta_loaded, f_max_loaded = load_data_KP23(Deltas=Deltas, Delta_index=0, mf=None, exponent_PL=exponent_PL)

    m_delta_extrapolated = 10**np.arange(11, np.log10(min(m_delta_loaded))+0.01, 0.1)
    f_max_extrapolated = min(f_max_loaded) * np.power(m_delta_extrapolated / min(m_delta_loaded), exponent_PL)
    f_max = np.concatenate((f_max_extrapolated, f_max_loaded))
    m_delta = np.concatenate((m_delta_extrapolated, m_delta_loaded))

    # Range of characteristic masses for obtaining constraints
    mc_Carr21 = 10**np.arange(14, 22.5, 0.1)

    # Values of sigma (parameter in log-normal distribution)
    sigmas = [sigmas_LN[-1], 2]

    f_PBH_evolved_sigma0 = constraint_Carr(mc_Carr21, m_delta, f_max, LN, [sigmas[0]], evolved=True)
    f_PBH_evolved_sigma1 = constraint_Carr(mc_Carr21, m_delta, f_max, LN, [sigmas[1]], evolved=True)
    
    ax.plot(m_delta, f_max, color="k", label="Delta func.")
    ax.plot(mc_Carr21 * np.exp(-sigmas[0]**2), f_PBH_evolved_sigma0, color=comp_color, linestyle="solid", label="LN ($\sigma={:.1f}$), ".format(sigmas[0]) + r"$M_{\rm p}$")   # thesis version
    ax.plot(mc_Carr21 * np.exp(-sigmas[1]**2), f_PBH_evolved_sigma1, color=comp_color, linestyle="dotted", label="LN ($\sigma={:.0f}$), ".format(sigmas[1]) + r"$M_{\rm p}$")       # thesis version   
    ax.plot(mc_Carr21, f_PBH_evolved_sigma0, color="b", linestyle="solid", label="LN ($\sigma={:.1f}$), ".format(sigmas[0]) + r"$M_{\rm c}$")   # thesis version
    ax.plot(mc_Carr21, f_PBH_evolved_sigma1, color="b", linestyle="dotted", label="LN ($\sigma={:.0f}$), ".format(sigmas[1]) + r"$M_{\rm c}$")   # thesis version

    ax.set_ylabel(r"$f_{\rm PBH}$")
    ax.set_ylim(1e-3, 1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e16, 5e20)

    x_major = mpl.ticker.LogLocator(base=10.0, numticks=10)
    ax.xaxis.set_major_locator(x_major)
    x_minor = mpl.ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)
    ax.xaxis.set_minor_locator(x_minor)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.tick_params("x", pad=7, which="both")
    ax.set_ylabel(r"$f_{\rm PBH}$")

    ax.legend(fontsize="xx-small", loc="upper left")
    #ax.set_xlabel(r"$m~[{\rm g}]$")
    ax.set_xlabel(r"$M~[{\rm g}]$")    # thesis version
    fig.tight_layout(pad=0.3)
    plt.savefig("fPBH_Voyager1_LN_mc_mp.pdf")