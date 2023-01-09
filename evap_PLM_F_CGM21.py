import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Specify the plot style
mpl.rcParams.update({'font.size': 20,'font.family':'serif'})
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

m_star = 5e14 / 1.989e33    # use value of M_* from CGM '21
gamma = 1
epsilon = 0.84

m2 = 1e17 / 1.989e33
m1 = 1e15 / 1.989e33

m_min = m1    # Minimum mass for which the power-law MF is non-zero
#m_min = m_star

def power_law_MF(m_values, m_max):
    MF_values = []
    for m in m_values:
        if m < m_min:
            MF_values.append(0)
        elif m > m_max:
            MF_values.append(0)
        else:
            MF_values.append(m**(gamma-1) * gamma / (m_max**gamma - m_min**gamma))
    return MF_values

def load_data(filename):
    return np.genfromtxt(filepath+filename, delimiter=',', unpack=True)

def integrand(m, m_max):
    return power_law_MF(m, m_max) / f_evap(m)    

def constraint_mono_analytic(m):
    return 4.32e-8 * np.array(m/m_star)**(3+epsilon)

def constraint_analytic(m_range, m_max):
    m2 = max(m_range)
    m1 = min(m_range)
    return 4.32e-8 * ((m_max**gamma - m_min**gamma) / gamma) * (gamma - (3+epsilon)) * (min(m2, m_max)**(gamma-(3+epsilon)) - max(m1, m_min)**(gamma-(3+epsilon)) )**(-1) / (m_star**(3+epsilon))

m_max_evaporation = 10**np.linspace(15, 17, 100) / 1.989e33
m_evaporation_mono, f_max_evaporation_mono = load_data('CGM_mono_combined.csv')

def f_evap(m):
    return np.interp(m, np.array(m_evaporation_mono)/1.989e33, f_max_evaporation_mono)


if "__main__" == __name__:

    # Plot constraints for monochromatic MF
    
    plt.figure(figsize=(12,8))
    plt.plot(m_evaporation_mono, f_max_evaporation_mono)
    plt.plot(m_evaporation_mono, constraint_mono_analytic(np.array(m_evaporation_mono) / 1.989e33), label='Analytic')
    plt.xlabel('$M$ [g]')
    plt.ylabel('$f_\mathrm{PBH}$')
    plt.xlim(1e15, 1e17)
    plt.ylim(1e-8, 1)
    plt.xscale('log')
    plt.yscale('log')

    # calculate constraints for extended MF from evaporation
    f_pbh_evap = []
    
    for m_max in m_max_evaporation:
        
        m_range = 10**np.linspace(np.log10(min(m_evaporation_mono)), np.log10(max(m_evaporation_mono)), 100000)
        f_pbh_evap.append(1/np.trapz(integrand(m=m_range, m_max=m_max), m_range))
        
        
    # Plot integrand 
    plt.figure(figsize=(12,8))
    m_range = 10**np.linspace(np.log10(m1), np.log10(m2), 10000)
    m_max = 10**(15.5) / 1.989e33
    
    m_values = 10**np.linspace(np.log10(m1), np.log10(m2), 100)
    integrand_values = integrand(m_values, m_max)   
    plt.plot(np.array(m_values)*1.989e33, integrand_values, linewidth=6)
    plt.xlabel('$M_\mathrm{PBH}~$[g]')
    plt.ylabel('$\psi(M, f_\mathrm{PBH}=1) / f_\mathrm{max}(M) $')
    plt.xscale('log')
    plt.yscale('log')
   
        
   # Plot constraints for a power-law MF
   
    plt.figure(figsize=(12,8))
    m_max_evaporation_PL, f_pbh_evaporation_PL = load_data('CGM_PL_gamma1_yellowdotted.csv')

    f_pbh_evap = []
    f_pbh_evap_analytic = []
    for m_max in m_max_evaporation:
                
        m_range = 10**np.linspace(np.log10(max(m1, m_min)), np.log10(min(m2, m_max)), 10000)
        
        print(m_range)
        
        f_pbh_evap.append(1/np.trapz(integrand(m=m_range, m_max=m_max), m_range))
        f_pbh_evap_analytic.append(constraint_analytic(m_range, m_max))
        
    plt.plot(np.array(m_max_evaporation)*1.989e33, np.array(f_pbh_evap), label='Calculated', linestyle = 'dotted', linewidth=6)
    #plt.plot(np.array(m_max_evaporation)*1.989e33, np.array(f_pbh_evap_analytic), label='Analytic', linestyle = 'dotted', linewidth=6)
    plt.plot(np.array(m_max_evaporation_PL), f_pbh_evaporation_PL, color='k', alpha=0.25, linewidth=4, label='Extracted (CGM 21)')

    plt.xlabel('$M_\mathrm{max}~$[g]')
    plt.ylabel('$f_\mathrm{PBH}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.ylim(10**(-7), 10**(-3.5))
    plt.title('Power-Law ($\gamma = {:.0f}$)'.format(gamma))
    plt.tight_layout()
    plt.savefig('Figures/CGM21_PL_gamma=1.pdf')