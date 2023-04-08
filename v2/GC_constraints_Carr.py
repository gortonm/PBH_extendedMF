#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 14:45:28 2023

@author: ppxmg2
"""
# Script for calculating Galactic Centre constraints for the numerical mass
# function calculated in 2008.03289, using the method from 1705.05567.

import numpy as np
from preliminaries import mf_numeric
from extended_MF_checks import constraint_Carr, load_results_Isatis, envelope

if "__main__" == __name__:

    # Mass function parameter values, from 2009.03204.
    Deltas = np.array([0., 0.1, 0.3, 0.5, 1.0, 2.0, 5.0])
    
    mc_values = np.logspace(14, 19, 100)
    m_mono_values = np.logspace(11, 21, 1000)

    # Load monochromatic MF constraints calculated using Isatis, to use the method from 1705.05567.
    # Using the envelope of constraints for each instrument for the monochromatic MF constraint.
    constraints_names, f_max = load_results_Isatis(modified=True, mf_string="GC_mono")


    for j in range(len(Deltas)):
        for j in [0, 1, 4, 5, 6]:
            params_numerical = [Deltas[j]]
            f_pbh_numeric = []
                    
            for i in range(len(constraints_names)):            
                # Calculate constraint using method from 1705.05567, and plot.
                f_pbh_numeric.append(constraint_Carr(mc_values, m_mono_values, f_max[i], mf_numeric, params_numerical))
            
            f_pbh_numeric_envelope = envelope(f_pbh_numeric) 
    
            data_filename_numeric = "./Data/numeric_GC_Carr_Delta={:.1f}.txt".format(Deltas[j])
            np.savetxt(data_filename_numeric, [mc_values, f_pbh_numeric_envelope], delimiter="\t")
