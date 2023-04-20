#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 14:45:28 2023

@author: ppxmg2
"""
# Script for calculating Galactic Centre constraints for the numerical mass
# function calculated in 2008.03289, using the method from 1705.05567.

import numpy as np
from preliminaries import mf_numeric, load_data, SLN, CC3
from extended_MF_checks import constraint_Carr, load_results_Isatis, envelope
import matplotlib.pyplot as plt
import matplotlib as mpl

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


#%% Constraints from COMPTEL, INTEGRAL, EGRET and Fermi-LAT.

if "__main__" == __name__:

    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    mc_values = np.logspace(14, 19, 100)
    m_mono_values = np.logspace(11, 21, 1000)

    # Load monochromatic MF constraints calculated using Isatis, to use the method from 1705.05567.
    # Using the envelope of constraints for each instrument for the monochromatic MF constraint.
    constraints_names, f_max = load_results_Isatis(modified=True, mf_string="GC_mono")


    for j in range(len(Deltas)):
        for j in [0, 1, 4, 5, 6]:
            params_CC3 = [alphas_CC3[j], betas[j]]
            params_numerical = [Deltas[j], params_CC3]
            f_pbh_numeric = []
                    
            for i in range(len(constraints_names)):            
                # Calculate constraint using method from 1705.05567, and plot.
                f_pbh_numeric.append(constraint_Carr(mc_values, m_mono_values, f_max[i], mf_numeric, params_numerical))
            
            f_pbh_numeric_envelope = envelope(f_pbh_numeric) 
    
            data_filename_numeric = "./Data/numeric_GC_Carr_Delta={:.1f}.txt".format(Deltas[j])
            #np.savetxt(data_filename_numeric, [mc_values, f_pbh_numeric_envelope], delimiter="\t")


#%% Constraints from 2302.04408 (MW diffuse SPI with NFW template)

if "__main__" == __name__:
    
    plot_extrapolate = False
    include_extrapolated = True

    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    mc_values = np.logspace(14, 19, 100)

    # Load monochromatic MF constraints calculated using Isatis, to use the method from 1705.05567.
    # Using the envelope of constraints for each instrument for the monochromatic MF constraint.
    m_mono_values, f_max = load_data("2302.04408/2302.04408_MW_diffuse_SPI.csv")

    # Plot extrapolated monochromatic MF constraints down to 5e14g

    if plot_extrapolate:
        
        # Estimate slope of constraints, using the data from 1e16g < m < a few times 3e16g
        
        # Maximum PBH mass for which the constraint shown in Figs. 1-2 of 2302.04408 is well-approximated by a power law.
        m_mono_max = 2e16
        m_mono_truncated = m_mono_values[m_mono_values < m_mono_max]
        f_max_truncated = f_max[m_mono_values < m_mono_max]
        
        # Estimated power-law slope
        slope_PL = (np.log10(max(f_max_truncated)) - np.log10(min(f_max_truncated))) / (np.log10(max(m_mono_truncated)) - np.log10(min(m_mono_truncated)))
        print("Maximum delta-function PBH mass used = {:.2e}".format(m_mono_max))
        print("Approximate power law slope = {:.2f}".format(slope_PL))
        print("Length of truncated arrays = {:.0f}".format(len(m_mono_truncated)))
        
        m_mono_extrapolated = np.logspace(np.log10(5e14), 16, 100)
        f_max_extrapolated = min(f_max) * np.power(m_mono_extrapolated / min(m_mono_values), slope_PL)
        
        f_max_total = np.concatenate((f_max_extrapolated, f_max))
        m_mono_total = np.concatenate((m_mono_extrapolated, m_mono_values))
        
        fig, ax = plt.subplots()
        ax.plot(m_mono_total, f_max_total, color=(0.5294, 0.3546, 0.7020), linestyle="dashed")
        ax.set_xlabel("$m$ [g]")
        ax.set_ylabel("$f_\mathrm{PBH}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(min(m_mono_total), 1e18)
        ax.set_ylim(1e-8, 1)
        fig.tight_layout()

    
    for j in range(len(Deltas)):
                
        if include_extrapolated:
            # Estimate f_max at PBH masses below 1e16g.
            slope_PL = 2.0
            m_mono_extrapolated = np.logspace(np.log10(5e14), 16, 100)
            f_max_extrapolated = min(f_max) * np.power(m_mono_extrapolated / min(m_mono_values), slope_PL)
            f_max_total = np.concatenate((f_max_extrapolated, f_max))
            m_mono_total = np.concatenate((m_mono_extrapolated, m_mono_values))
            
            data_filename_SLN = "./Data/SLN_2302.04408_Carr_Delta={:.1f}_extrapolated.txt".format(Deltas[j])
            data_filename_CC3 = "./Data/CC3_2302.04408_Carr_Delta={:.1f}_extrapolated.txt".format(Deltas[j])
            #data_filename_numeric = "./Data/numeric_2302.04408_Carr_Delta={:.1f}_extrapolated.txt".format(Deltas[j])
                      
        else:
            f_max_total = f_max
            m_mono_total = m_mono_values
            
            data_filename_SLN = "./Data/SLN_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[j])
            data_filename_CC3 = "./Data/CC3_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[j])
            #data_filename_numeric = "./Data/numeric_2302.04408_Carr_Delta={:.1f}.txt".format(Deltas[j])
            
        params_SLN = [sigmas_SLN[j], alphas_SLN[j]]
        params_CC3 = [alphas_CC3[j], betas[j]]
        params_numerical = [Deltas[j], params_CC3]
        
        f_pbh_SLN = constraint_Carr(mc_values, m_mono_total, f_max_total, SLN, params_SLN)
        f_pbh_CC3 = constraint_Carr(mc_values, m_mono_total, f_max_total, CC3, params_CC3)
        #f_pbh_numeric = constraint_Carr(mc_values, m_mono_total, f_max_total, mf_numeric, params_numerical)
                          
        np.savetxt(data_filename_SLN, [mc_values, f_pbh_SLN], delimiter="\t")
        np.savetxt(data_filename_CC3, [mc_values, f_pbh_CC3], delimiter="\t")
        #np.savetxt(data_filename_numeric, [mc_values, f_pbh_numeric], delimiter="\t")