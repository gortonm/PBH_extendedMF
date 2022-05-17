#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 10:08:29 2022

@author: ppxmg2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cProfile
import pstats

# Specify the plot style
mpl.rcParams.update({'font.size': 24,'font.family':'serif'})
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


filepath = './Extracted_files/'


def load_data(filename):
    return np.genfromtxt(filepath+filename, delimiter=',', unpack=True)

if "__main__" == __name__:    

    # Subaru-HSC constraints
    m_subaru_Carr, f_subaru_Carr = load_data('Subaru-HSC_mono.csv')
    
    m_subaru_Croon_R90_0, f_subaru_Croon_R90_0 = load_data('R_90_0_BS.csv')
    m_subaru_Croon_NFW_R90_1, f_subaru_Croon_NFW_R90_1 = load_data('R_90_0.1R_odot_NFW.csv')
    m_subaru_Croon_NFW_R90_0, f_subaru_Croon_NFW_R90_0 = load_data('R_90_R_odot_NFW.csv')
    m_subaru_Croon_BS_R90_1, f_subaru_Croon_BS_R90_1 = load_data('R_90_0.1R_odot_BS.csv')
    
    m_subaru_Smyth_g, f_subaru_Smyth = load_data('Smyth+20.csv')
    m_subaru_Smyth = np.array(m_subaru_Smyth_g) / 1.989e33
    
    plt.figure(figsize=(12,9))
    plt.plot(m_subaru_Carr, f_subaru_Carr, linewidth = 3, label='Carr+ 21 (Fig. 20 LH panel)')
    plt.plot(m_subaru_Croon_R90_0, f_subaru_Croon_R90_0, linewidth = 3, label='Croon+ 20 BS $(R_{90} = 0)$')
    plt.plot(m_subaru_Croon_BS_R90_1, f_subaru_Croon_BS_R90_1, linewidth = 3, label='Croon+ 20 BS $(R_{90} = 0.1 R_\odot)$')
    plt.plot(m_subaru_Croon_NFW_R90_1, f_subaru_Croon_NFW_R90_1, linewidth = 3, label='Croon+ 20 NFW $(R_{90} = 0.1 R_\odot)$')
    plt.plot(m_subaru_Croon_NFW_R90_0, f_subaru_Croon_NFW_R90_0, linewidth = 3, label='Croon+ 20 NFW $(R_{90} = R_\odot)$')
    
    plt.plot(m_subaru_Smyth, f_subaru_Smyth, linewidth = 3, label='Smyth+ 20', linestyle='dashed')
    
    
    plt.xlabel('$M_\mathrm{PBH}~[M_\odot]$')
    plt.ylabel('$f_\mathrm{PBH}$')
    plt.title('Subaru-HSC (monochromatic)')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-3, 1)
    plt.xlim(1e-12, 1e-5)
    plt.legend(fontsize='small')
    plt.savefig('./Figures/comparison_Subaru.png')
    
    
    
    # Evaporation constraints
    m_evap_Carr21, f_evap_Carr21 = load_data('Gamma-ray_mono.csv')
    
    m_evap_Carr10_g, beta_prime_evap_Carr10 = load_data('Carr+10_beta_prime.csv')
    m_evap_Carr10 = np.array(m_evap_Carr10_g) / 1.989e33
    f_evap_Carr10 = 3.81e8 * np.array(beta_prime_evap_Carr10) * np.array(m_evap_Carr10)**(-1/2)
    
    plt.figure(figsize=(12,9))
    plt.plot(m_evap_Carr21, f_evap_Carr21, linewidth = 3, label='Carr+ 21 (Fig. 20 LH panel)')
    plt.plot(m_evap_Carr10, f_evap_Carr10, linewidth = 3, label='Carr+ 10 (Fig. 5)')
    
    plt.xlabel('$M_\mathrm{PBH}~[M_\odot]$')
    plt.ylabel('$f_\mathrm{PBH}$')
    plt.title('Subaru-HSC (monochromatic)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(fontsize='small')
    plt.savefig('./Figures/comparison_evap.png')
