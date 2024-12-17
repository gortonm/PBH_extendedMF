# PBH_extendedMF


Code for calculating constraints for primordial black holes (PBHs) with more realistic extended mass functions, associated with the paper "*How open is the asteroid-mass primordial black hole window?*"  (see https://arxiv.org/abs/2403.03839).

# Motivation

Observational constraints on PBHs appear to exclude PBHs from making up all of the dark matter unless their mass, $M_{\rm PBH}$, lies in the range $10^{17} \, {\rm g} \lesssim M_{\rm PBH} \lesssim 10^{22} \, {\rm g}$, often known as the `asteroid-mass window'.

Constraints on PBHs are usually obtained assuming all PBHs have the same mass, though accounting for critical collapse shows they would have an extended mass function. A lognormal fit has been widely used to parameterise the PBH mass function, though recent work has shown that other functions provide a better fit (see Gow et al. (https://arxiv.org/abs/2009.03204)). 

This code recalculates both current and prospective future constraints on PBHs with these improved fitting functions, to assess to what extent the asteroid-mass window remains open. 


# Contents
The code, plots etc. are arranged as follows

## Folders
* `Data` Contains extended mass function constraints, calculated using `calc_extended_MF_constraints.py`. The sub-folders, labelled `PL_exp_{X}` include constraints calculated by extrapolating the delta-function mass function to masses smaller than those for which data is publicly available, assuming a power-law with exponent `{X}`,
* `Extracted_files` Includes relevant data files from other papers, i.e. the relevant delta-function mass function constraints and the $\alpha_{\rm eff}$ parameter calculated in Mosbech & Picker (https://arxiv.org/abs/2203.05743).

## .py files

* ` preliminaries.py ` Includes methods for calculating extended mass function constraints and code to analyse data.
* `calc_extended_MF_constraints` Calculates constraints for the fitting functions from Gow et al. ([https://arxiv.org/abs/2009.03204](https://arxiv.org/abs/2009.03204))  using the method from Carr et al. (https://arxiv.org/abs/1705.05567).
* ` plot_constraints.py` Plots constraints on PBHs with extended mass functions.

## Other files
* `asteroid_mass_gap.yml` Conda environment file.
* `MF_params.txt` Includes the parameters for the best-fit fitting functions found by Gow et al. ([https://arxiv.org/abs/2009.03204). If it does not already exist, it is generated when running `calc_extended_MF_constraints.py`.
# Requirements
Python 3

