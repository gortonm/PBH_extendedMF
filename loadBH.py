#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 14:52:58 2022

@author: ppxmg2
"""
import numpy as np

file_path_extracted = './Extracted_files/'

def load_data(fname):
    """
    Load data from a file located in the folder './Extracted_files/'.

    Parameters
    ----------
    fname : String
        File name.

    Returns
    -------
    Numpy array of type Float.
        Contents of file.

    """
    return np.genfromtxt(file_path_extracted+fname, delimiter=',', unpack=True)


def read_col(fname, first_row=0, col=1, convert=int, sep=None):
    """
    Read text files with columns separated by 'sep'.

    Parameters
    ----------
    fname : String
        File name.
    first_row : TYPE, optional
        First row of file to load. The default is 0.
    col : Integer, optional
        Index of column to read. The default is 1.
    convert : optional
        Change results from
    sep : Character, optional
        Column separator. The default is None.

    Returns
    -------
    data : Numpy array of type Float
        Data from text file.

    """
    data = []

    with open(fname) as fobj:
        i = 0
        for line in fobj:

            i += 1

            if i >= first_row:
                data.append(convert(line.split(sep=sep)[col]))

    return np.array(data)


def read_blackhawk_spectra(fname, col=1):
    """
    Read spectrum for a particle species, calculated using BlackHawk.

    Parameters
    ----------
    fname : String
        File name.
    col : Integer, optional
        Index of column to read (i.e. which particle type to use).
        The default is 1 (photons).
        For electrons and positrons, use col=7 (primary spectrum
        only) or col=2 (total spectrum).

    Returns
    -------
    Numpy array of type Float
        Energies which the spectrum is evaluated at.

    Numpy array of type Float
        Spectrum, computed using BlackHawk.
    """
    energies_data = read_col(fname, first_row=3, col=0, convert=float)
    spectrum_data = read_col(fname, first_row=3, col=col, convert=float)

    energies = []
    for i in range(2, len(energies_data)):
        energies.append(float(energies_data[i]))

    spectrum = []
    for i in range(2, len(spectrum_data)):
        spectrum.append(float(spectrum_data[i]))

    return np.array(energies), np.array(spectrum)
