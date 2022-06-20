#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 10:08:29 2022

@author: ppxmg2
"""

# Compare constraints extracted from figures of other papers to those
# shown in the LH panel of Fig. 20 of Carr+ '21

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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

m_BC19_propA_nobkg, f_BC19_propA_nobkg = load_data('Prop_A_wo_bkg.csv')
m_BC19_propA_bkg, f_BC19_propA_bkg = load_data('Prop_A_with_bkg.csv')
m_BC19_propB_nobkg, f_BC19_propB_nobkg = load_data('Prop_B_wo_bkg_upper.csv')
m_BC19_propB_bkg, f_BC19_propB_bkg = load_data('Prop_B_with_bkg.csv')

m_L19_NFW_3000pc, f_L19_NFW_3000pc = load_data('Laha19_NFW_3kpc.csv')
m_L19_Iso_1500pc, f_L19_Iso_1500pc = load_data('Laha19_Iso_1.5kpc.csv')

m_DG19_all, f_DG19_all = load_data('DG19_Fig1_all.csv')
m_DG19_less1MeV, f_DG19_less1MeV = load_data('DG19_Fig1_less1MeV.csv')

m_B22_SK, f_B22_SK = load_data('Bernal+21_SK.csv')

if "__main__" == __name__:        
    
    
    plt.figure(figsize=(14,9))
    BC19_1, = plt.plot(m_BC19_propA_bkg, f_BC19_propA_bkg, linewidth = 3, color='b', linestyle='dotted', label='Prop A, w/ background')
    BC19_2, = plt.plot(m_BC19_propA_nobkg, f_BC19_propA_nobkg, linewidth = 3, color='b', label='Prop A, w/o background')
    BC19_3, = plt.plot(m_BC19_propB_bkg, f_BC19_propB_bkg, linewidth = 3, color='purple', linestyle='dotted', label='Prop B, w/ background')
    BC19_4, = plt.plot(m_BC19_propB_nobkg, f_BC19_propB_nobkg, linewidth = 3, color='purple', label='Prop B, w/o background')

    #legend_1 = plt.legend(handles=[BC19_1, BC19_2], title='Voyager 1 (BC19)', fontsize='small', loc='right')
    legend_1 = plt.legend(handles=[BC19_1, BC19_2, BC19_3, BC19_4], title='Voyager 1 (BC19)', fontsize='small', loc='right')
    

    DG19_1, = plt.plot(m_DG19_all, f_DG19_all, linewidth = 3, color='orange', label="DG19 (all positron injection energies)")
    DG19_2, = plt.plot(m_DG19_less1MeV, f_DG19_less1MeV, linewidth = 3, color='orange', linestyle='dashed', label="DG19 (positron injection energy $<$ 1 MeV)")

    L19_1, = plt.plot(m_L19_NFW_3000pc, f_L19_NFW_3000pc, linewidth = 3, color='r', label="L19 (NFW, 3 kpc)")
    L19_2, = plt.plot(m_L19_Iso_1500pc, f_L19_Iso_1500pc, linewidth = 3, color='r', linestyle='dashed', label="L19 (Iso, 1.5 kpc)")
    
    legend_2 = plt.legend(handles=[DG19_1, DG19_2, L19_1, L19_2], title='511 keV', fontsize='small', loc='lower right')
    #legend_2 = plt.legend(handles=[L19_1, L19_2], title='511 keV', fontsize='small', loc='lower right')

    #plt.plot(m_B22_SK, f_B22_SK, linewidth = 3, color='y', label="Super-Kamiokande")
    
    plt.xlabel('$M_\mathrm{PBH}~[M_\odot]$')
    plt.ylabel('$f_\mathrm{PBH}$')
    plt.title('Evaporation (monochromatic)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e15, 10**17.5)
    plt.ylim(1e-7, 1)
    plt.gca().add_artist(legend_1)
    plt.gca().add_artist(legend_2)
    plt.savefig('./Figures/constraints_comparison_evap_bkg_no_bkg_noDG19.pdf')
