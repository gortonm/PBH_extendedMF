#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 19:19:55 2023

@author: ppxmg2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from preliminaries import constraint_Carr, LN, CC3

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

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)

    # Peak PBH masses
    mp_values = np.logspace(14, 17, 30)

    # Functional form of f_max
    PL_exp = 2
    m_pbh_values = np.logspace(11, 20, 1000)
    f_max = (1 / max(mp_values**PL_exp)) * m_pbh_values**PL_exp
        
    for i in range(len(Deltas)):
        
        fig, ax = plt.subplots(figsize=(7,7))
        ax.plot(m_pbh_values, f_max, color="tab:grey", label="Delta function", linewidth=2)
        
        # Log-normal MF constraint
        params_LN = [sigmas_LN[i]]
        mc_values_LN = mp_values * np.exp(sigmas_LN[i]**2)
        f_PBH_values_LN = constraint_Carr(mc_values_LN, m_delta=m_pbh_values, f_max=f_max, psi_initial=LN, params=params_LN, evolved=False)
    
        # CC3 MF
        params_CC3 = [alphas_CC3[i], betas[i]]
        f_PBH_values_CC3 = constraint_Carr(mp_values, m_delta=m_pbh_values, f_max=f_max, psi_initial=CC3, params=params_CC3, evolved=False)

        ax.plot(mp_values, f_PBH_values_LN, color="tab:red", linestyle="dashed", label="LN")
        ax.plot(mp_values, f_PBH_values_CC3, color="tab:green", linestyle="dashed", label="CC3")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(min(mp_values), max(mp_values))
        ax.set_ylim(min(f_PBH_values_CC3), max(f_PBH_values_CC3))
        ax.set_xlabel("$m_p~[\mathrm{g}]$")
        ax.set_ylabel("$f_\mathrm{PBH}$")
        ax.legend(fontsize="small")
        fig.suptitle("$\Delta={:.1f}$".format(Deltas[i]) + ", $f_\mathrm{max}(m) " + "\propto m^{:.0f}$".format(PL_exp))
        fig.tight_layout()