#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 8 14:45:28 2023

@author: ppxmg2
"""
# Script for calculating constraints for the fitting functions from 2009.03204
# using the method from 1705.05567.

import numpy as np
from preliminaries import load_data, LN, SLN, CC3, constraint_Carr
import os

t_0 = 13.8e9 * 365.25 * 86400    # Age of Universe, in seconds
# Load mass function parameters.
[Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)

#%% Constraints from 2302.04408 (MW diffuse SPI with NFW template).

if "__main__" == __name__:
    # If True, use extrapolate delta-function MF constraints down to 1e11g (using a power law fit) to calculate extended MF constraint.
    include_extrap = True
    # If True, use evolved mass function.
    evolved = True
    # If True, evaluate the evolved mass function at t=0.
    t_initial = False
    if t_initial:
        evolved = True
        
    t = t_0
        
    mc_values = np.logspace(14, 20, 120)
    
    # Load delta function MF constraints calculated using Isatis, to use the method from 1705.05567.
    m_delta_values, f_max = load_data("2302.04408/2302.04408_MW_diffuse_SPI.csv")
    
    # Power-law exponent to use between 1e11g and 1e16g
    for exponent_PL in [0, 2, 3, 4]:
        
        if not evolved:
            data_folder = "./Data/unevolved"
        elif t_initial:
            data_folder = "./Data/t_initial"
            t = 0
        else:
            data_folder = "./Data"

        if include_extrap:
                    
            m_delta_extrap = np.logspace(11, 16, 51)
            f_max_extrap = min(f_max) * np.power(m_delta_extrap / min(m_delta_values), exponent_PL)
        
            f_max_total = np.concatenate((f_max_extrap, f_max))
            m_delta_total = np.concatenate((m_delta_extrap, m_delta_values))
        
            data_folder += "/PL_exp_{:.0f}".format(exponent_PL)
        
        else:
            f_max_total = f_max
            m_delta_total = m_delta_values
           
        # Produce directory to save files if it does not exist
        os.makedirs(data_folder, exist_ok=True)
        
        for j in range(len(Deltas)):
            
            if include_extrap:
                data_filename_LN = data_folder + "/LN_2302.04408_Delta={:.1f}_extrap_exp{:.0f}.txt".format(Deltas[j], exponent_PL)
                data_filename_SLN = data_folder + "/SLN_2302.04408_Delta={:.1f}_extrap_exp{:.0f}.txt".format(Deltas[j], exponent_PL)
                data_filename_CC3 = data_folder + "/CC3_2302.04408_Delta={:.1f}_extrap_exp{:.0f}.txt".format(Deltas[j], exponent_PL)
        
            else:          
                data_filename_LN = data_folder + "/LN_2302.04408_Delta={:.1f}.txt".format(Deltas[j])
                data_filename_SLN = data_folder + "/SLN_2302.04408_Delta={:.1f}.txt".format(Deltas[j])
                data_filename_CC3 = data_folder + "/CC3_2302.04408_Delta={:.1f}.txt".format(Deltas[j])
                
            params_LN = [sigmas_LN[j]]
            params_SLN = [sigmas_SLN[j], alphas_SLN[j]]
            params_CC3 = [alphas_CC3[j], betas[j]]
                           
            f_pbh_LN = constraint_Carr(mc_values, m_delta_total, f_max_total, LN, params_LN, evolved, t)
            f_pbh_SLN = constraint_Carr(mc_values, m_delta_total, f_max_total, SLN, params_SLN, evolved, t)
            f_pbh_CC3 = constraint_Carr(mc_values, m_delta_total, f_max_total, CC3, params_CC3, evolved, t)
                        
            np.savetxt(data_filename_LN, [mc_values, f_pbh_LN], delimiter="\t")                          
            np.savetxt(data_filename_SLN, [mc_values, f_pbh_SLN], delimiter="\t")
            np.savetxt(data_filename_CC3, [mc_values, f_pbh_CC3], delimiter="\t")
            

#%% Constraints from 1807.03075 (Voyager-1 electron / positron detections).

if "__main__" == __name__:
    
    # If True, use extrapolated delta-function MF constraints down to 1e11g (using a power law fit) to calculate extended MF constraint.
    include_extrapolated = True
    # If True, use evolved mass function.
    evolved = True
    # If True, evaluate the evolved mass function at t=0.
    t_initial = False
    if t_initial:
        evolved = True
                
    # If True, load the more stringent or less stringent "prop B" data
    prop_B_lower = False
            
    t = t_0
    
    mc_values = np.logspace(14, 20, 120)
    
    extrap_numeric_lower = False
    normalised = True
    
    if not evolved:
        data_folder_base = "./Data/unevolved"
    elif t_initial:
        data_folder_base = "./Data/t_initial"
        t = 0
    else:
        data_folder_base = "./Data"

    # Boolean determines which propagation model to load delta-function MF constraint from
    for prop_A in [True]:

        prop_B = not prop_A
        
        # If True, load constraint obtained with a background or without background subtraction.
        for with_bkg_subtr in [True]:
            
            # Load delta function MF constraints calculated using Isatis, to use the method from 1705.05567.
            # Load the most stringent delta-function MF constraints.
            if prop_A:
                prop_string = "prop_A"
                if with_bkg_subtr:
                    m_delta_values, f_max = load_data("1807.03075/1807.03075_prop_A_bkg.csv")
                else:
                    m_delta_values, f_max = load_data("1807.03075/1807.03075_prop_A_nobkg.csv")
        
            elif prop_B:
                prop_string = "prop_B"
                if with_bkg_subtr:
                    if not prop_B_lower:
                        m_delta_values, f_max = load_data("1807.03075/1807.03075_prop_B_bkg_upper.csv")
                    else:
                        m_delta_values, f_max = load_data("1807.03075/1807.03075_prop_B_bkg_lower.csv")                        
                else:
                    if not prop_B_lower:
                        m_delta_values, f_max = load_data("1807.03075/1807.03075_prop_B_nobkg_upper.csv")
                    else:
                        m_delta_values, f_max = load_data("1807.03075/1807.03075_prop_B_nobkg_lower.csv")                        
                
            if not with_bkg_subtr:
                prop_string += "_nobkg"
            if prop_B and prop_B_lower:
                prop_string += "_lower"
            elif prop_B:
                prop_string += "_upper"

            # Power-law exponent to use between 1e11g and the smallest mass the delta-function MF constraint is calculated for.            
            for exponent_PL_lower in [0, 2, 3, 4]:
            
                if include_extrapolated:
            
                    m_delta_extrapolated = 10**np.arange(11, np.log10(min(m_delta_values))+0.01, 0.1)
                    f_max_extrapolated = min(f_max) * np.power(m_delta_extrapolated / min(m_delta_values), exponent_PL_lower)
                
                    f_max_total = np.concatenate((f_max_extrapolated, f_max))
                    m_delta_total = np.concatenate((m_delta_extrapolated, m_delta_values))
                
                    data_folder = data_folder_base + "/PL_exp_{:.0f}".format(exponent_PL_lower)
            
                else:
                    f_max_total = f_max
                    m_delta_total = m_delta_values
                    data_folder = data_folder_base
                    
                # Produce directory to save files if it does not exist
                os.makedirs(data_folder, exist_ok=True)
                                        
                for j in range(len(Deltas)):                
                    if include_extrapolated:                     
                        data_filename_LN = data_folder + "/LN_1807.03075_" + prop_string + "_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
                        data_filename_SLN = data_folder + "/SLN_1807.03075_" + prop_string + "_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
                        data_filename_CC3 = data_folder + "/CC3_1807.03075_" + prop_string + "_Delta={:.1f}_extrapolated_exp{:.0f}.txt".format(Deltas[j], exponent_PL_lower)
           
                    else:          
                        data_filename_LN = data_folder + "/LN_1807.03075_" + prop_string + "_Delta={:.1f}.txt".format(Deltas[j])
                        data_filename_SLN = data_folder + "/SLN_1807.03075_" + prop_string + "_Delta={:.1f}.txt".format(Deltas[j])
                        data_filename_CC3 = data_folder + "/CC3_1807.03075_" + prop_string + "_Delta={:.1f}.txt".format(Deltas[j])
                       
                    params_LN = [sigmas_LN[j]]
                    params_SLN = [sigmas_SLN[j], alphas_SLN[j]]
                    params_CC3 = [alphas_CC3[j], betas[j]]
                    
                    f_pbh_LN = constraint_Carr(mc_values, m_delta_total, f_max_total, LN, params_LN, evolved, t)
                    f_pbh_SLN = constraint_Carr(mc_values, m_delta_total, f_max_total, SLN, params_SLN, evolved, t)
                    f_pbh_CC3 = constraint_Carr(mc_values, m_delta_total, f_max_total, CC3, params_CC3, evolved, t)
              
                    # Produce directory to save files if it does not exist
                    os.makedirs(os.path.dirname(data_folder), exist_ok=True)
    
                    np.savetxt(data_filename_LN, [mc_values, f_pbh_LN], delimiter="\t")                          
                    np.savetxt(data_filename_SLN, [mc_values, f_pbh_SLN], delimiter="\t")
                    np.savetxt(data_filename_CC3, [mc_values, f_pbh_CC3], delimiter="\t")
                    
        
#%% Constraints from 2101.01370 (Prospective gamma-ray constraints from GECCO).
if "__main__" == __name__:
    # If True, use extrapolated delta-function MF constraints down to 1e11g (using a power law fit) to calculate extended MF constraint
    include_extrapolated = True
    # If True, plot extrapolated delta-function MF constraints down to 1e11g
    evolved = True
    # If True, evaluate the evolved mass function at t=0.
    t_initial = False
    if t_initial:
        evolved = True
    # If True, plot the projected constraints for an NFW profile. If False, plot the projected constraints for an Einasto profile
    NFW = True
    
    t = t_0
        
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    mc_values = np.logspace(14, 21, 130)
    
    for NFW in [False, True]:
    
        for exponent_PL_lower in [0, 2, 3, 4]:
            
            if not evolved:
                data_folder = "./Data/unevolved"
            elif t_initial:
                data_folder = "./Data/t_initial"
                t = 0
            else:
                data_folder = "./Data"
        
            # Load delta function MF constraints calculated using Isatis, to use the method from 1705.05567.
            if NFW:
                m_delta_values, f_max = load_data("2101.01370/2101.01370_Fig9_GC_NFW.csv")
                profile_string = "NFW"
                
            else:
                m_delta_values, f_max = load_data("2101.01370/2101.01370_Fig9_GC_Einasto.csv")
                profile_string = "Einasto"
                    
            if include_extrapolated:
        
                # Power-law exponent to use between 1e11g and 1e15g.
                
                m_delta_extrapolated = np.logspace(11, 15, 41)
                f_max_extrapolated = min(f_max) * np.power(m_delta_extrapolated / min(m_delta_values), exponent_PL_lower)
            
                f_max_total = np.concatenate((f_max_extrapolated, f_max))
                m_delta_total = np.concatenate((m_delta_extrapolated, m_delta_values))
                data_folder += "/PL_exp_{:.0f}".format(exponent_PL_lower)
            
            else:
                f_max_total = f_max
                m_delta_total = m_delta_values
                
            # Produce directory to save files if it does not exist
            os.makedirs(data_folder, exist_ok=True)

            for j in range(len(Deltas)):                
                if include_extrapolated:                     
                    data_filename_LN = data_folder + "/LN_2101.01370_Delta={:.1f}_".format(Deltas[j]) + "%s_" % profile_string + "extrapolated_exp{:.0f}.txt".format(exponent_PL_lower)
                    data_filename_SLN = data_folder + "/SLN_2101.01370_Delta={:.1f}_".format(Deltas[j]) + "%s_" % profile_string + "extrapolated_exp{:.0f}.txt".format( exponent_PL_lower)
                    data_filename_CC3 = data_folder + "/CC3_2101.01370_Delta={:.1f}_".format(Deltas[j]) + "%s_" % profile_string + "extrapolated_exp{:.0f}.txt".format(exponent_PL_lower)
                else:           
                    data_filename_LN = data_folder + "/LN_2101.01370_Delta={:.1f}_".format(Deltas[j]) + "%s_" % profile_string + ".txt"
                    data_filename_SLN = data_folder + "/SLN_2101.01370_Delta={:.1f}_".format(Deltas[j]) + "%s_" % profile_string + ".txt"
                    data_filename_CC3 = data_folder + "/CC3_2101.01370_Delta={:.1f}_".format(Deltas[j]) + "%s_" % profile_string + ".txt"
                   
                params_LN = [sigmas_LN[j]]
                params_SLN = [sigmas_SLN[j], alphas_SLN[j]]
                params_CC3 = [alphas_CC3[j], betas[j]]
                    
                f_pbh_LN = constraint_Carr(mc_values, m_delta_total, f_max_total, LN, params_LN, evolved, t)
                f_pbh_SLN = constraint_Carr(mc_values, m_delta_total, f_max_total, SLN, params_SLN, evolved, t)
                f_pbh_CC3 = constraint_Carr(mc_values, m_delta_total, f_max_total, CC3, params_CC3, evolved, t)
                                
                np.savetxt(data_filename_LN, [mc_values, f_pbh_LN], delimiter="\t")                          
                np.savetxt(data_filename_SLN, [mc_values, f_pbh_SLN], delimiter="\t")
                np.savetxt(data_filename_CC3, [mc_values, f_pbh_CC3], delimiter="\t")

#%% Constraints from 2007.12697 Fig. 4 (Subaru-HSC point-like constraint).
if "__main__" == __name__:

    mc_subaru = 10**np.linspace(17, 30, 1000)
    
    # Constraints for a delta-function MF.
    m_subaru_delta, f_max_subaru_delta = load_data("./2007.12697/Subaru-HSC_2007.12697_dx=5.csv")
    
    # Mass function parameter values, from 2009.03204.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SL, sigmas_SLN, alphas_SL, mp_CC, alphas_CC, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    evolved = False
        
    for i in range(len(Deltas)):
    
        # Calculate constraints for extended MF from microlensing.
        params_SLN = [sigmas_SLN[i], alphas_SL[i]]
        params_CC3 = [alphas_CC[i], betas[i]]
        params_LN = [sigmas_LN[i]]
            
        f_pbh_LN = constraint_Carr(mc_subaru, m_subaru_delta, f_max_subaru_delta, LN, params_LN, evolved)
        f_pbh_SLN = constraint_Carr(mc_subaru, m_subaru_delta, f_max_subaru_delta, SLN, params_SLN, evolved)
        f_pbh_CC3 = constraint_Carr(mc_subaru, m_subaru_delta, f_max_subaru_delta, CC3, params_CC3, evolved)
        
        if evolved:
            data_filename_SLN = "./Data/SLN_HSC_Delta={:.1f}_evolved.txt".format(Deltas[i])
            data_filename_CC3 = "./Data/CC3_HSC_Delta={:.1f}_evolved.txt".format(Deltas[i])
            data_filename_LN = "./Data/LN_HSC_Delta={:.1f}_evolved.txt".format(Deltas[i])
        else:
            data_filename_SLN = "./Data/SLN_HSC_Delta={:.1f}.txt".format(Deltas[i])
            data_filename_CC3 = "./Data/CC3_HSC_Delta={:.1f}.txt".format(Deltas[i])
            data_filename_LN = "./Data/LN_HSC_Delta={:.1f}.txt".format(Deltas[i])
            
        np.savetxt(data_filename_SLN, [mc_subaru, f_pbh_SLN], delimiter="\t")
        np.savetxt(data_filename_CC3, [mc_subaru, f_pbh_CC3], delimiter="\t")
        np.savetxt(data_filename_LN, [mc_subaru, f_pbh_LN], delimiter="\t")


#%% Constraints from 1905.06066 Fig. 8 (prospective white dwarf microlensing).
if "__main__" == __name__:
    
    mc_values = 10**np.linspace(17, 29, 1000)
        
    # Constraints for delta-function MF.
    m_delta, f_max_delta = load_data("./1905.06066/1905.06066_Fig8_finite+wave.csv")
    
    # Mass function parameter values, from 2009.03204.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SL, sigmas_SLN, alphas_SL, mp_CC, alphas_CC, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    evolved = False
                        
    # Mass function parameter values, from 2009.03204.
    [Deltas, sigmas_LN, ln_mc_SL, mp_SL, sigmas_SLN, alphas_SL, mp_CC, alphas_CC, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    for i in range(len(Deltas)):
    
        # Calculate constraints for extended MF from microlensing.
        params_SLN = [sigmas_SLN[i], alphas_SL[i]]
        params_CC3 = [alphas_CC[i], betas[i]]
        params_LN = [sigmas_LN[i]]
            
        f_pbh_LN = constraint_Carr(mc_values, m_delta, f_max_delta, LN, params_LN, evolved)
        f_pbh_SLN = constraint_Carr(mc_values, m_delta, f_max_delta, SLN, params_SLN, evolved)
        f_pbh_CC3 = constraint_Carr(mc_values, m_delta, f_max_delta, CC3, params_CC3, evolved)

        if evolved:
            data_filename_SLN = "./Data/SLN_Sugiyama20_Delta={:.1f}_evolved.txt".format(Deltas[i])
            data_filename_CC3 = "./Data/CC3_Sugiyama20_Delta={:.1f}_evolved.txt".format(Deltas[i])
            data_filename_LN = "./Data/LN_Sugiyama20_Delta={:.1f}_evolved.txt".format(Deltas[i])
        else:
            data_filename_SLN = "./Data/SLN_Sugiyama20_Delta={:.1f}.txt".format(Deltas[i])
            data_filename_CC3 = "./Data/CC3_Sugiyama20_Delta={:.1f}.txt".format(Deltas[i])
            data_filename_LN = "./Data/LN_Sugiyama20_Delta={:.1f}.txt".format(Deltas[i])
                       
        np.savetxt(data_filename_SLN, [mc_values, f_pbh_SLN], delimiter="\t")
        np.savetxt(data_filename_CC3, [mc_values, f_pbh_CC3], delimiter="\t")
        np.savetxt(data_filename_LN, [mc_values, f_pbh_LN], delimiter="\t")
    
    
