#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 11:51:13 2022
@author: ppxmg2
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from J_factor_A22 import j_avg

# Script to compare the fluxes used in the calculations of Auffinger '22
# Fig. 3 (2201.01265), to the constraints from Coogan, Morrison & Profumo '21
# (2010.04797)

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

file_path_extracted = './Extracted_files/'

def load_data(filename):
    return np.genfromtxt(file_path_extracted+filename, delimiter=',', unpack=True)

def read_col(fname, first_row=0, col=1, convert=int, sep=None, skip_lines=1):
    """Read text files with columns separated by `sep`.
    fname - file name
    col - index of column to read
    convert - function to convert column entry with
    sep - column separator
    If sep is not specified or is None, any
    whitespace string is a separator and empty strings are
    removed from the result.
    """
    
    data = []
    
    with open(fname) as fobj:
        i=0
        for line in fobj:
            i += 1
            #print(line)
            
            if i >= first_row:
                #print(line.split(sep=sep)[col])
                data.append(line.split(sep=sep)[col])
    return data    
     
def read_blackhawk_spectra(fname, col=1):
    """Read spectra files for a particular particle species, calculated
    from BlackHawk.
    fname - file name
    col - index of column to read (i.e. which particle type to use)
        - photons: col = 1 (primary and secondary spectra)
        - 7 for electrons (primary spectra)
    """
    energies_data = read_col(fname, first_row=2, col=0, convert=float)
    spectrum_data = read_col(fname, first_row=2, col=col, convert=float)
    
    energies = []
    for i in range(2, len(energies_data)):
        energies.append(float(energies_data[i]))

    spectrum = []
    for i in range(2, len(spectrum_data)):
        spectrum.append(float(spectrum_data[i]))
        
    return np.array(energies), np.array(spectrum)

# returns a number as a string in standard form
def string_scientific(val):
    exponent = np.floor(np.log10(val))
    coefficient = val / 10**exponent
    return r'${:.0f} \times 10^{:d}$'.format(coefficient, int(exponent))

if "__main__" == __name__:
    
    file_path_extracted = './Extracted_files/A22_COMPTEL/'

    # COMPTEL measured flux from Auffinger '22 Fig. 2 (top left panel)
    E_Auffinger_mean, spec_Auffinger_mean = load_data('means.csv')
    E_Auffinger_bin_lower, a = load_data('lower_bin.csv')
    E_Auffinger_bin_upper, a = load_data('upper_bin.csv')
    
    # Use data values given in Isatis for COMPTEL measured values
    spec_Auffinger_mean = np.array([2.38601e+00, 3.44189e-01, 3.40997e-02])
    energies_minus = np.array([7.21020e-04, 2.50612e-03, 7.20580e-03])
    energies_plus = np.array([1.23204e-03, 4.47746e-03, 1.26645e-02])
    
    E_Auffinger_mean = np.array([1.73836e-03, 5.51171e-03, 1.73730e-02])
    E_Auffinger_bin_lower = E_Auffinger_mean - energies_minus
    E_Auffinger_bin_upper = E_Auffinger_mean + energies_plus
    
    # Unit conversions
    g_to_solar_mass = 1 / 1.989e33    # g to solar masses
    pc_to_cm = 3.085678e18    # pc to cm
    
    # Calculate J-factor (units: solar mass * pc^{-2} * sr^{-1})
    b_max_Auffinger, l_max_Auffinger = np.radians(15), np.radians(30)
    J_A22 = j_avg(b_max_Auffinger, l_max_Auffinger)
    #print('rho_0 [g cm^{-3}]', rho_0 * (g_to_solar_mass)**(-1) * (pc_to_cm)**(-3))
    print('J [g cm^{-2}]', J_A22 * (g_to_solar_mass)**(-1) * (pc_to_cm)**(-2))
    
    f_PBH_A22 = []
    
    #m_pbh_values = np.array([0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.5, 2, 3, 4, 6, 8]) * 10**16
    m_pbh_values = 10**(np.linspace(14, 18, 25))
    
    for i, m_pbh in enumerate(m_pbh_values):
            
        exponent = np.floor(np.log10(m_pbh))
        coefficient = m_pbh / 10**exponent
        
        #file_path_data = "../blackhawk_v2.0/results/A22_Fig3_" + "{:.1f}e{:.0f}g/".format(coefficient, exponent)
        #print("{:.1f}e{:.0f}g".format(coefficient, exponent))    # check that the correct PBH mass is being loaded
 
        file_path_data = "./../Downloads/version_finale/results/runs_COMPTEL_0510_{:.0f}/".format(i+1)
        
        # Load photon spectra from BlackHawk outputs
        energies, spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_secondary_spectra.txt", col=1)
        
        if m_pbh == 1e15:
            print(spectrum)
        
        # Calculate flux from a theoretical PBH population, and compare to that
        # observed by COMPTEL.
        flux_theoretical = []
        flux_measured = []
        integrals = []
        flux_ratio = []
        
        for i in range(0, 3):
            
            E_min = E_Auffinger_bin_lower[i]
            E_max = E_Auffinger_bin_upper[i]     
            
            energies_interp = 10**np.linspace(np.log10(E_min), np.log10(E_max), 100)
            spectrum_interp = 10**np.interp(np.log10(energies_interp), np.log10(energies), np.log10(spectrum))
            integral = np.trapz(spectrum_interp, energies_interp)
            
            flux_measured.append(spec_Auffinger_mean[i] * (E_max - E_min))
            flux_theoretical.append(integral * J_A22[0] / (4 * np.pi * m_pbh))
                    
            flux_ratio.append(pc_to_cm**2 * g_to_solar_mass * flux_measured[i] / flux_theoretical[i])
            integrals.append(integral)
            
        print('f_PBH values : ', flux_ratio)
        print('f_PBH_2 / f_PBH_3 : ', flux_theoretical[1] / flux_theoretical[2] )
        print('I_2 / I_3 : ', integrals[1] / integrals[2])
        f_PBH_A22.append(min(flux_ratio))
        
        print('M_{PBH} [g] : ' + ' {0:.1e}'.format(m_pbh))
        print('M_{PBH} loaded [g] : ' + ' {:.1f}e+{:.0f}'.format(coefficient, exponent))    # check that the correct PBH mass is being loaded
        print("Bin with minimum f_{PBH, i} [Auffinger] : ", np.argmin(flux_ratio))
        #print("Integral : ", integral)
    
    # Load result extracted from Fig. 3 of Auffinger '22
    file_path_extracted = './Extracted_files/'
    m_pbh_A22_extracted, f_PBH_A22_extracted = load_data("A22_Fig3.csv")
    
    
    # Load result from Isatis
    Isatis_path = './../Downloads/version_finale/scripts/Isatis/'
    results_name = "test_COMPTEL" 

    constraints_file = np.genfromtxt("%sresults_photons_%s.txt"%(Isatis_path,results_name),dtype = "str")
    constraints_names_bis = constraints_file[0,1:]
    constraints = np.zeros([len(constraints_file)-1,len(constraints_file[0])-1])
    for i in range(len(constraints)):
        for j in range(len(constraints[0])):
            constraints[i,j] = float(constraints_file[i+1,j+1])

    constraint_COMPTEL = []
    for i in range(len(constraints)):
        constraint_COMPTEL.append(constraints[i][1])

 
    results_name_mod = "test_COMPTEL_modified" 

    constraints_file = np.genfromtxt("%sresults_photons_%s.txt"%(Isatis_path,results_name_mod),dtype = "str")
    constraints_names_bis = constraints_file[0,1:]
    constraints = np.zeros([len(constraints_file)-1,len(constraints_file[0])-1])
    for i in range(len(constraints)):
        for j in range(len(constraints[0])):
            constraints[i,j] = float(constraints_file[i+1,j+1])

    constraint_COMPTEL_modified = []
    for i in range(len(constraints)):
        constraint_COMPTEL_modified.append(constraints[i][1])

        
    Mmin = 1e14
    Mmax = 1e18
    masses_Isatis = np.logspace(np.log10(Mmin),np.log10(Mmax),len(constraint_COMPTEL))

    
    
    
    plt.figure(figsize=(7,7))
    plt.plot(m_pbh_A22_extracted, f_PBH_A22_extracted, label="Extracted from Auffinger '22 ")
    plt.plot(masses_Isatis, constraint_COMPTEL_modified, 'x', label="Isatis (modified loop condition)")
    plt.plot(masses_Isatis, constraint_COMPTEL, 'x', label="Isatis (unmodified)")
    plt.plot(m_pbh_values, f_PBH_A22, 'x', label="Reproduction (BlackHawk spectra)")
    
    plt.xlabel('$M_\mathrm{PBH}$ [g]')
    plt.ylabel('$f_\mathrm{PBH}$')
    plt.tight_layout()
    plt.legend(fontsize='small')
    plt.xscale('log')
    plt.yscale('log')
    #plt.title('Excluding highest-energy bin')
    plt.xlim(1e14, 1e18)
    plt.ylim(1e-10, 1)
    plt.tight_layout()
    #%% Investigate which bins are causing the constraint, and why
    
    
    #m_pbh_values = np.array([0.4, 0.6, 0.8, 1, 1.2, 1.5, 4]) * 10**16
    
    #fig_flux, ax_flux = plt.subplots()
    fig_integral, ax_integral = plt.subplots()
    fig_integral2, ax_integral2 = plt.subplots()
    fig_ratio, ax_ratio = plt.subplots()
    
    for i in range(0, 3):
        
        E_min = E_Auffinger_bin_lower[i]
        E_max = E_Auffinger_bin_upper[i]       
    
        flux = []
        integral = []
        ratio = []
        
        for m_pbh in m_pbh_values:
                
            exponent = np.floor(np.log10(m_pbh))
            coefficient = m_pbh / 10**exponent
            
            file_path_data = "../blackhawk_v2.0/results/A22_Fig3_" + "{:.1f}e{:.0f}g/".format(coefficient, exponent)
            
            # Load photon spectra from BlackHawk outputs
            energies_primary, spectrum_primary = read_blackhawk_spectra(file_path_data + "instantaneous_primary_spectra.txt", col=1)
            energies_total, spectrum_total = read_blackhawk_spectra(file_path_data + "instantaneous_secondary_spectra.txt", col=1)
                                
            # Load photon primary spectrum
            energies_total_interp = 10**np.linspace(np.log10(E_min), np.log10(E_max), 100)
            spectrum_total_interp = 10**np.interp(np.log10(energies_total_interp), np.log10(energies_total), np.log10(spectrum_total))
            integral_primary = np.trapz(spectrum_total_interp, energies_total_interp)
            
            flux.append(spec_Auffinger_mean[i] * (E_max - E_min))
            integral.append(integral_primary) 
            ratio.append(spec_Auffinger_mean[i] * (E_max - E_min) / integral_primary)
            
            
            if 1e15 < m_pbh <= 4e15:
                # compare original photon spectra to interpolated versions
                if i == 2:
                    plt.figure(figsize=(8, 6))
                    plt.plot(energies_total, spectrum_total, label='Original (primary and secondary)')
                    plt.plot(energies_total_interp, spectrum_total_interp, 'x', label='Interpolated')
                    plt.xlim(1e-4, 0.03)
                    plt.ylim(1e15, 5e21)
                    plt.xlabel('Energy E [GeV]')
                    plt.ylabel('$\mathrm{d}^2 n_\gamma / (\mathrm{d}t\mathrm{d}E)$ [cm$^{-3}$ s$^{-1}$ GeV$^{-1}$]')
                    plt.title('$M_\mathrm{PBH}$ = ' + "{:.1f}e{:.0f}".format(coefficient, exponent) + 'g')
                    
                    plt.yscale('log')
                    plt.xscale('log')
                    plt.tight_layout()
                    plt.legend()
                    
                    plt.axvline(E_Auffinger_bin_lower[0], ymin=0, ymax=1, linestyle='dotted', color='grey')
                    plt.axvline(E_Auffinger_bin_lower[1], ymin=0, ymax=1, linestyle='dotted', color='grey')
                    plt.axvline(E_Auffinger_bin_lower[2], ymin=0, ymax=1, linestyle='dotted', color='grey')
                    plt.axvline(E_Auffinger_bin_upper[2], ymin=0, ymax=1, linestyle='dotted', color='grey')
                
                """
                if i == 0:
                    # Plot photon spectra for different PBH masses to illustrate differences
                    plt.figure()
                    plt.plot(energies_primary, spectrum_primary, label='Primary')
                    plt.plot(energies_total, spectrum_total, linestyle='dashed', label='Total')
                    plt.xlim(1e-4, 0.03)
                    plt.ylim(1e15, 5e21)
                    plt.xlabel('Energy E [GeV]')
                    plt.ylabel('$\mathrm{d}^2 n_\gamma / (\mathrm{d}t\mathrm{d}E)$ [cm$^{-3}$ s$^{-1}$ GeV$^{-1}$]')
                    plt.title('$M_\mathrm{PBH}$ = ' + "{:.1f}e{:.0f}".format(coefficient, exponent) + 'g')
                    
                    #plt.yscale('log')
                    #plt.xscale('log')
                    plt.tight_layout()
                    plt.legend()
                    
                    plt.axvline(E_Auffinger_bin_lower[0], ymin=0, ymax=1, linestyle='dotted', color='grey')
                    plt.axvline(E_Auffinger_bin_lower[1], ymin=0, ymax=1, linestyle='dotted', color='grey')
                    plt.axvline(E_Auffinger_bin_lower[2], ymin=0, ymax=1, linestyle='dotted', color='grey')
                    plt.axvline(E_Auffinger_bin_upper[2], ymin=0, ymax=1, linestyle='dotted', color='grey')
                """
            
        
        print(flux[0])
        #ax_flux.plot(m_pbh_values, flux, 'x', label=str(i+1))
        ax_integral.plot(m_pbh_values, integral, 'x', label=str(i+1))
        ax_integral2.plot(m_pbh_values, integral / (E_max - E_min), 'x', label=str(i+1))
        ax_ratio.plot(m_pbh_values, ratio, 'x', label=str(i+1))
        
    """
    ax_flux.set_yscale('log')
    ax_flux.set_xscale('log')
    ax_flux.set_ylabel('Measured flux $[\mathrm{cm}^{-2} \cdot \mathrm{s}^{-1} \cdot \mathrm{sr}^{-1}]$')
    ax_flux.set_xlabel('$M_\mathrm{PBH}$ [g]')
    ax_flux.legend()
    fig_flux.tight_layout()
    """
    ax_integral.set_yscale('log')
    ax_integral.set_xscale('log')
    ax_integral.set_ylabel('Integral $[\mathrm{s}^{-1}]$')
    ax_integral.set_xlabel('$M_\mathrm{PBH}$ [g]')
    ax_integral.legend()
    fig_integral.tight_layout()
    
    
    ax_integral2.set_yscale('log')
    ax_integral2.set_xscale('log')
    ax_integral2.set_ylabel('Integral / (bin width) $[\mathrm{GeV}^{-1}\mathrm{s}^{-1}]$')
    ax_integral2.set_xlabel('$M_\mathrm{PBH}$ [g]')
    ax_integral2.legend()
    fig_integral2.tight_layout()
    
    
    ax_ratio.set_yscale('log')
    ax_ratio.set_xscale('log')
    ax_ratio.set_ylabel('Flux ratio (observed/predicted)')
    ax_ratio.set_xlabel('$M_\mathrm{PBH}$ [g]')
    ax_ratio.legend(title='Bin', fontsize='small')
    fig_ratio.tight_layout()
    
    
    #%% Check that I can reproduce CMP '21 Figure 2
    m_pbh_values = [5.3e14, 3.5e15]
    
    
    for m_pbh in m_pbh_values:
        exponent = np.floor(np.log10(m_pbh))
        coefficient = m_pbh / 10**exponent
        file_path_data = "../blackhawk_v2.0/results/CMP21_Fig2_" + "{:.1f}e{:.0f}g/".format(coefficient, exponent)
        
        energies_primary, primary_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_primary_spectra.txt", col=1)
        energies_secondary, secondary_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_secondary_spectra.txt", col=1)
    
        plt.figure()
    
        E_CMP21_Fig2, spec_CMP21_Fig2 = load_data("CMP21_Fig2_{:.1f}e{:.0f}g.csv".format(coefficient, exponent))
        
        plt.plot(E_CMP21_Fig2, spec_CMP21_Fig2, label='Extracted', color='k', alpha=0.2  )
        plt.plot(1e3 * np.array(energies_primary), 1e-3 * np.array(primary_spectrum), linestyle='dashed', label='Primary')
        plt.plot(1e3 * np.array(energies_secondary), 1e-3 * np.array(secondary_spectrum), linestyle='dotted', linewidth=3, label='Total')
            
        plt.title('$M_\mathrm{PBH}$ = ' + "{:.1f}e{:.0f}".format(coefficient, exponent) + 'g')
        plt.xlabel('Energy [MeV]')
        plt.ylabel(r'$\frac{\mathrm{d}N_\gamma}{\mathrm{d}E_\gamma\mathrm{d}t}~(\mathrm{MeV}^{-1}\mathrm{s}^{-1})$')
        plt.ylim(1e16, 1e21)
        plt.xlim(1e-3, 1e3)
        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
        plt.tight_layout()
        
    m_pbh_values = [3.5e16, 1.8e17]
    
    for m_pbh in m_pbh_values:
        exponent = np.floor(np.log10(m_pbh))
        coefficient = m_pbh / 10**exponent
        file_path_data = "../blackhawk_v2.0/results/CMP21_Fig2_" + "{:.1f}e{:.0f}g/".format(coefficient, exponent)
        
        energies_primary, primary_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_primary_spectra.txt", col=1)
        energies_secondary, secondary_spectrum = read_blackhawk_spectra(file_path_data + "instantaneous_secondary_spectra.txt", col=1)
    
        plt.figure()
        E_CMP21_Fig2, spec_CMP21_Fig2 = load_data("CMP21_Fig2_{:.1f}e{:.0f}g.csv".format(coefficient, exponent))
        
        plt.plot(E_CMP21_Fig2, spec_CMP21_Fig2, label='Extracted', color='k', alpha=0.2  )
        plt.plot(1e3 * np.array(energies_primary), 1e-3 * np.array(primary_spectrum), linestyle='dashed', label='Primary')
        plt.plot(1e3 * np.array(energies_secondary), 1e-3 * np.array(secondary_spectrum), linestyle='dotted', linewidth=3, label='Total')
        plt.title('$M_\mathrm{PBH}$ = ' + "{:.1f}e{:.0f}".format(coefficient, exponent) + 'g')
        plt.xlabel('Energy [MeV]')
        plt.ylabel(r'$\frac{\mathrm{d}N_\gamma}{\mathrm{d}E_\gamma\mathrm{d}t}~(\mathrm{MeV}^{-1}\mathrm{s}^{-1})$')
        plt.ylim(1e15, 1e20)
        plt.xlim(1e-3, 1e3)
        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
        plt.tight_layout()