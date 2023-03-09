#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 15:25:33 2023
@author: ppxmg2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import erf, loggamma

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


#%% Create .txt file with fitting function parameters.
# Results for lognormal from Table II of 2008.03289.
# Results for SLN and CC3 from Table II of 2009.03204.

if "__main__" == __name__:
    Deltas = [0., 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    sigmas_LN = [0.374, 0.377, 0.395, 0.430, 0.553, 0.864, 0.]
    ln_mc_SLN = [4.13, 4.13, 4.15, 4.21, 4.40, 4.88, 5.41]
    mp_SLN = [40.9, 40.9, 40.9, 40.8, 40.8, 40.6, 32.9]
    sigmas_SLN = [0.55, 0.55, 0.57, 0.60, 0.71, 0.97, 2.77]
    alphas_SLN = [-2.27, -2.24, -2.07, -1.82, -1.31, -0.66, 1.39]
    mp_CC3 = [40.8, 40.8, 40.7, 40.7, 40.8, 40.6, 35.1]
    alphas_CC3 = [3.06, 3.09, 3.34, 3.82, 5.76, 18.9, 13.9]
    betas = [2.12, 2.08, 1.72, 1.27, 0.51, 0.0669, 0.0206]
    
    file_header = "Delta \t sigma (LN) \t ln_mc (SLN) \t m_p (SLN) \t sigma (SLN) \t alpha (SLN) \t alpha (CC3) \t beta (CC3)"
    params = [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas]
    
    np.savetxt("MF_params.txt", np.column_stack(params), delimiter="\t\t ", header=file_header, fmt="%s")
    
    # Check file loads correctly
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    
#%% Functions

def load_data(filename, directory="./Extracted_files/"):
    """
    Load data from a file located in the folder './Extracted_files/'.
    Parameters
    ----------
    filename : String
        Name of file to load data from.
    directory : String
        Directory in which the file is located. The default is "./../Extracted_files/".
    Returns
    -------
    Array-like.
        Contents of file.
    """
    return np.genfromtxt(directory+filename, delimiter=',', unpack=True)


def LN(m, m_c, sigma):
    """
    Log-normal mass function (with characteristic mass m_c and standard deviation sigma), evaluated at m.
    Parameters
    ----------
    m : Array-like
        PBH mass.
    m_c : Float
        Characteristic PBH mass.
    sigma : Float
        Standard deviation of the log-normal mass function.
    Returns
    -------
    Array-like
        Value of the log-normal mass function.
    """
    return np.exp(-np.log((m / m_c)**2) / (2 * sigma**2)) / (np.sqrt(2*np.pi) * sigma * m)


def SLN(m, m_c, sigma, alpha):
    """
    Skew-lognormal mass function (defined in 2009.03204 Eq. 8), with characteristic mass m_c and parameters sigma and alpha, evaluated at m.
    Parameters
    ----------
    m : Array-like
        PBH mass.
    m_c : Float
        Characteristic PBH mass.
    sigma : Float
        Parameter relates to width of skew-lognormal mass function.
    alpha : Float
        Parameter controls the skewness (alpha=0 reduces to a lognormal).
    Returns
    -------
    Array-like
        Value of the skew-lognormal mass function.
    """
    return np.exp(-np.log(m/m_c)**2 / (2*sigma**2)) * (1 + erf(alpha * np.log(m/m_c) / (np.sqrt(2) * sigma))) / (np.sqrt(2*np.pi) * sigma * m)


def CC3(m, m_p, alpha, beta):
    """
    Critical collapse 3 mass function (defined in 2009.03204 Eq. 9), with peak mass m_p and parameters alpha and beta, evaluated at m.
    Parameters
    ----------
    m : Array-like
        PBH mass.
    m_p : Float
        Peak mass.
    alpha : Float
        Controls location and shape of the low mass tail.
    beta : Float
        Controls location and shape of the hig mass tail.
    Returns
    -------
    Array-like
        Value of the critical collapse 3 mass function.
    """
    m_f = m_p * np.power(beta/alpha, 1/beta)
    log_psi = np.log(beta/m_f) - loggamma((alpha+1) / beta) + (alpha * np.log(m/m_f)) - np.power(m/m_f, beta)
    return np.exp(log_psi)


def m_max_LN(m_c, sigma):
    """
    Calculate the mass at which the log-normal mass function is maximised.
    Parameters
    ----------
    m_c : Float
        Characteristic PBH mass.
    sigma : Float
        Standard deviation of the log-normal mass function.
    Returns
    -------
    Float
        Peak mass of the log-normal mass function.
    """
    return m_c * np.exp(-sigma**2)   


def m_max_SLN(m_c, sigma, alpha, log_m_factor=5, n_steps=100000):
    """
    Estimate the mass at which the skew-lognormal mass function is maximised.
    Parameters
    ----------
    m_c : Float
        Characteristic PBH mass.
    sigma : Float
        Parameter relates to width of skew-lognormal mass function.
    alpha : Float
        Parameter controls the skewness (alpha=0 reduces to a lognormal).
    log_m_factor : Float, optional
        Number of multiples of sigma (in log-space) of masses around m_c to consider when estimating the maximum. The default is 5.
    n_steps : Integer, optional
        Number of masses to use for estimating the peak mass of the skew-lognormal mass function. The default is 100000.
    Returns
    -------
    Float
        Estimate for the peak mass of the skew-lognormal mass function..
    """
    log_m_min = np.log(m_c) - log_m_factor*sigma
    log_m_max = np.log(m_c) + log_m_factor*sigma
    m_pbh_values = np.logspace(log_m_min, log_m_max, n_steps)
    
    # Calculate mass function at each PBH mass.
    psi_values = SLN(m_pbh_values, m_c, sigma, alpha)
    
    return m_pbh_values[np.argmax(psi_values)]


def load_results_Isatis_mono(modified=True):
    
    # Choose path to Isatis.
    if modified:
        Isatis_path = "../../Downloads/version_finale/scripts/Isatis/"
    else:
        Isatis_path = "../../Downloads/version_finale_unmodified/scripts/Isatis/"

    # Load Isatis constraints data.
    
    constraints_file = np.genfromtxt("%sresults_photons_GC_mono.txt"%(Isatis_path), dtype = "str")
    constraints_names_loaded = constraints_file[0,1:]
    constraints_loaded = np.zeros([len(constraints_file)-1,len(constraints_file[0])-1])
    
    constraints_names = []
    f_PBH_Isatis = []
    
    # Create arrays of constraints, using results obtained from Isatis.
    # Based upon code appearing in plotting.py within Isatis.
    for i in range(len(constraints_loaded)):
        for j in range(len(constraints_loaded[0])):            
            constraints_loaded[i,j] = float(constraints_file[i+1,j+1])
        
        if not(all(constraints_loaded[i] < 0)):
            f_PBH_Isatis.append(constraints_loaded[i])
            constraints_names.append(constraints_names_loaded[i])
            
    return constraints_names, f_PBH_Isatis


#%% Code outside of method to use for loading Isatis data.
modified = False
# Choose path to Isatis.
if modified:
    Isatis_path = "../../Downloads/version_finale/scripts/Isatis/"
else:
    Isatis_path = "../../Downloads/version_finale_unmodified/scripts/Isatis/"

# Load Isatis constraints data.
constraints_file = np.genfromtxt("%sresults_photons_GC_mono.txt"%(Isatis_path), dtype = "str", unpack=True)[1:]

constraints_names = []
f_PBH_Isatis = []

for i in range(len(constraints_file)):
    #print(constraints_file[i])
    
    constraint = [float(constraints_file[i][j]) for j in range(1, len(constraints_file[i]))]
    
    if all(constraint) > 0:
        print(constraint)

        f_PBH_Isatis.append(constraint)
        
        # Create labels
        # Based upon code appearing in plotting.py within Isatis.
        temp = constraints_file[i][0].split("_")
        temp2 = ""
        for i in range(len(temp)-1):
            temp2 = "".join([temp2,temp[i],'\,\,'])
        temp2 = "".join([temp2,'\,\,[arXiv:',temp[-1],']'])

        constraints_names.append(temp2)

#%% Compare SLN and CC3 MF to Fig. 5 of 2009.03204.

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)

    # Indices in array of Delta values to reproduce Fig. 5 of 2009.03204.
    # Corresponds to Delta = 0.1, 5.0.
    Delta_indices_Fig3 = [1, 6]
    
    # Number of masses to plot
    n_masses = 1000
    
    # Approximate range of masses to include to reproduce Fig. 5 of 2009.03204.
    # Corresponds to Delta = 0.1, 5.0. 
    m_pbh_Fig3 = np.array([np.logspace(1, np.log10(80), n_masses), np.logspace(0, np.log10(2000), n_masses)])
    
    for i, Delta_index in enumerate(Delta_indices_Fig3):
        
        print(ln_mc_SLN[Delta_index])
        print(sigmas_SLN[Delta_index])
        print(alphas_SLN[Delta_index])
        print(mp_SLN[Delta_index])
        print(mp_CC3[Delta_index])
        print(alphas_CC3[Delta_index])
        print(betas[Delta_index])

        # Load data from Fig. 3 of 2009.03204.
        m_loaded_numeric, psi_scaled_numeric = load_data("Delta_{:.1f}_numeric.csv".format(Deltas[Delta_index]), directory="./Extracted_files/2009.03204/")
        m_loaded_SLN, psi_scaled_SLN = load_data("Delta_{:.1f}_SLN.csv".format(Deltas[Delta_index]), directory="./Extracted_files/2009.03204/")
        m_loaded_CC3, psi_scaled_CC3 = load_data("Delta_{:.1f}_CC3.csv".format(Deltas[Delta_index]), directory="./Extracted_files/2009.03204/")
        
        # Calculate the LN, SLN and CC3 mass functions.
        m_pbh_values = m_pbh_Fig3[i]
        if Delta_index == 1:
            psi_SLN = SLN(m_pbh_values, np.exp(ln_mc_SLN[Delta_index]), sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index])
        else:
            psi_SLN = SLN(m_pbh_values, np.exp(ln_mc_SLN[Delta_index]), sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index])            
        psi_CC3 = CC3(m_pbh_values, mp_CC3[Delta_index], alpha=alphas_CC3[Delta_index], beta=betas[Delta_index])
        
        m_max_SLN_val = m_max_SLN(np.exp(ln_mc_SLN[Delta_index]), sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index], n_steps=10000)
        print("M_max (SLN) = {:.1f} M_\odot".format(m_max_SLN_val))
        
        if Delta_index == 1:
            psi_SLN_max = max(SLN(m_pbh_values, np.exp(ln_mc_SLN[Delta_index]), sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index]))
            print(psi_SLN[10])
            print(psi_SLN_max)

        else:
            psi_SLN_max = max(SLN(m_pbh_values, np.exp(ln_mc_SLN[Delta_index]), sigma=sigmas_SLN[Delta_index], alpha=alphas_SLN[Delta_index]))

        psi_CC3_max = max(CC3(m_pbh_values, mp_CC3[Delta_index], alpha=alphas_CC3[Delta_index], beta=betas[Delta_index]))
        
        
        # Plot the mass function.
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.plot(m_loaded_numeric, psi_scaled_numeric, color="k", label="Numeric", linewidth=2)
        ax.plot(m_loaded_SLN, psi_scaled_SLN, color="b", linestyle="None", marker="x")
        ax.plot(m_loaded_CC3, psi_scaled_CC3, color="tab:green", linestyle="None", marker="x")
        ax.plot(m_pbh_values, psi_SLN / psi_SLN_max, color="b", label="SLN", linewidth=2)
        
        if Delta_index == 6:
            ax.plot(m_pbh_values, 0.98 * psi_CC3 / psi_CC3_max, color="tab:green", label="CC3", linewidth=2)
        else:
            ax.plot(m_pbh_values, psi_CC3 / psi_CC3_max, color="tab:green", label="CC3", linewidth=2)

        ax.plot(0, 0, color="grey", linestyle="None", marker="x", label="Extracted (Fig. 5 2009.03204)")
        ax.set_xlabel("$M_\mathrm{PBH}~[M_\odot]$")
        ax.set_ylabel("$\psi(M_\mathrm{PBH}) / \psi_\mathrm{max}$")
        ax.set_xlim(min(m_pbh_values), max(m_pbh_values))
        ax.set_ylim(0.1, 1.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize="small", title="$\Delta={:.1f}$".format(Deltas[Delta_index]))
        plt.tight_layout()
        plt.show()

#%% Compare peak mass of the skew lognormal with different mass ranges and numbers of masses tested, to the values from Table II of 2009.03204.

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    # Set required fractional precision
    precision = 1e-5
    print("Fractional precision =", precision)
    
    m_c = 1

    # Number of steps to use when estimating the peak mass.
    n_steps_range = 10**np.arange(2, 6.1, 1)
    # Number of sigmas to use for mass range when estimating the peak mass.   
    n_sigmas_range = np.arange(1, 10.1, 1)
    
    # Minimum number of steps to use when estimating the peak mass.
    n_steps_min = max(n_steps_range) * np.ones(len(Deltas))
    # Minimum number of sigmas to use for mass range when estimating the peak mass.
    n_sigma_min = max(n_sigmas_range) * np.ones(len(Deltas))
    
    # Cycle through range of number of masses to use in the estimate
    for i in range(len(Deltas)):
        
        alpha = alphas_SLN[i]
        sigma = sigmas_SLN[i]

        sigma_range_min = max(n_sigmas_range) * np.ones(len(Deltas))
        n_steps_range_min = max(n_steps_range) * np.ones(len(Deltas))
        
        # Calculate the most precise estimate for the peak mass
        mp_best_estimate = m_max_SLN(m_c, sigma, alpha, log_m_factor=max(n_sigmas_range), n_steps=int(max(n_steps_range)))
        
        stop_loop = False
        
        for n_steps in n_steps_range:
            
            for n_sigma in n_sigmas_range:
                
                # Estimated peak mass of the SLN mass function.
                m_max_SLN_est = m_max_SLN(m_c, sigma, alpha, log_m_factor=n_sigma, n_steps=int(n_steps))
                frac_diff = abs((m_max_SLN_est - mp_best_estimate) / mp_best_estimate)
                
                if frac_diff < precision:
                    
                    n_steps_min[i] = n_steps
                    n_sigma_min[i] = n_sigma
                    
                    # Break loop, to give the minimum number of steps required and minimum range for a given precision of the M_p calculation.
                    stop_loop = True
                    break
                
                if stop_loop:
                    break
                    
            if stop_loop:
                break

        print("Delta = {:.1f}".format(Deltas[i]))
        print("n_steps_min = {:.2e}".format(n_steps_min[i]))
        print("n_sigmas_min = {:.0f}".format(n_sigma_min[i]))
        
#%% Check m_max_SLN(), by comparing results to Table II of 2009.03204, accounting for the uncertainty in parameters due to the limited precision given.

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)
    
    # Peak mass of the skew lognormal mass function, from Table II of 2009.03204.
    mp_SL = [40.9, 40.9, 40.9, 40.8, 40.8, 40.6, 32.9]
        
    # Cycle through range of number of masses to use in the estimate
    for i in range(len(Deltas)):
        print("\nDelta = {:.1f}".format(Deltas[i]))
        
        # Account for uncertainty due to the limited precision of values given in Table II of 2009.03204.
        for ln_mc in [ln_mc_SLN[i]-0.005,  ln_mc_SLN[i], ln_mc_SLN[i]+0.005]:
            
            for alpha in [alphas_SLN[i]-0.005, alphas_SLN[i], alphas_SLN[i]+0.005]:
                
                for sigma in [sigmas_SLN[i]-0.005, sigmas_SLN[i], sigmas_SLN[i]+0.005]:
                    
                    # Estimated peak mass of the SLN mass function.
                    m_max_SLN_est = m_max_SLN(np.exp(ln_mc), sigma, alpha, log_m_factor=n_sigma_min[i], n_steps=int(n_steps_min[i]))
                    
                    # Compare to peak mass given in Table II of 2009.03204
                    if abs(m_max_SLN_est - mp_SL[i]) < 0.05:
                        
                        print("Success")
                        
                        # Calculate and print fractional difference
                        frac_diff = abs((m_max_SLN_est - mp_SL[i]) / mp_SL[i])
                        print("Fractional difference = {:.2e}".format(frac_diff))


#%% Estimate the range of masses for which the mass function is non-negligible.

if "__main__" == __name__:
    
    # Load mass function parameters.
    [Deltas, sigmas_LN, ln_mc_SLN, mp_SLN, sigmas_SLN, alphas_SLN, mp_CC3, alphas_CC3, betas] = np.genfromtxt("MF_params.txt", delimiter="\t\t ", skip_header=1, unpack=True)

    # Minimum value of the mass function (scaled to its peak value).
    cutoff = 0.01
    
    # Number of masses to use to estimate the peak mass.
    n_steps = 1000000
    
    # Arrays of minimum and maximum masses for which the mass function is non-negligible. 
    m_range_LN = []
    m_range_SLN = []
    m_range_CC3 = []

    for i in range(len(Deltas[:-1])):
        
        # Assign characteristic mass values.
        mc_LN = mp_SLN[i] * np.exp(sigmas_LN[i]**2)   # compute m_c for lognormal by relating it to the peak mass of the SLN MF
        m_c = np.exp(ln_mc_SLN[i])
        m_p = mp_CC3[i]
        
        # Assign lower and upper masses of the range.
        m_min, m_max = m_c / 1e3, m_c * 1e3
        
        # Range of values of the PBH mass to use to estimate for the cutoff.
        m_pbh_values = np.logspace(np.log10(m_min), np.log10(m_max), n_steps)

        print("\nDelta = {:.1f}".format(Deltas[i]))
        
        psi_LN_scaled = LN(m_pbh_values, m_c, sigma=sigmas_LN[i]) / LN(m_max_LN(m_c, sigma=sigmas_LN[i]), m_c, sigma=sigmas_LN[i])
        psi_SLN_scaled = SLN(m_pbh_values, m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i]) / max(SLN(m_pbh_values, m_c, sigma=sigmas_SLN[i], alpha=alphas_SLN[i]))
        psi_CC3_scaled = CC3(m_pbh_values, m_p, alpha=alphas_CC3[i], beta=betas[i]) / CC3(m_p, m_p, alpha=alphas_CC3[i], beta=betas[i])
                
        m_range_LN.append([min(m_pbh_values[psi_LN_scaled > cutoff]), max(m_pbh_values[psi_LN_scaled > cutoff])])
        m_range_SLN.append([min(m_pbh_values[psi_SLN_scaled > cutoff]), max(m_pbh_values[psi_SLN_scaled > cutoff])])
        m_range_CC3.append([min(m_pbh_values[psi_CC3_scaled > cutoff]), max(m_pbh_values[psi_CC3_scaled > cutoff])])
        
        print("Mass range where psi/psi_max > {:.1e}:".format(cutoff))
        print("LN: ", m_range_LN[i])
        print("SLN : ", m_range_SLN[i])
        print("CC3 : ", m_range_CC3[i])
        
        print("Scaled mass range where psi/psi_max > {:.1e}:".format(cutoff))
        print("LN: ", m_range_LN[i] / m_c)
        print("SLN : ", m_range_SLN[i] / m_c)
        print("CC3 : ", m_range_CC3[i] / m_p)