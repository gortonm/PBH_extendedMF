#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 15:28:15 2022

@author: ppxmg2
"""

import numpy as np
import matplotlib.pyplot as plt

filepath = './Extracted_files/'

def load_data(filename):
    return np.genfromtxt(filepath+filename, delimiter=',', unpack=True)

def trapezium(x, y):
    area = 0
    for i in range(len(x) - 1):
        area += (x[i+1] - x[i]) * (y[i] + 0.5*(x[i+1]-x[i])*(y[i+1]-y[i]))
    return area

R_source_Rsol, pdf_load = load_data('mean_R_pdf.csv')
# convert units of source radius from solar radii to parsecs
r_sol = 2.25461e-8  # solar radius, in pc
R_source_pc = R_source_Rsol * r_sol

# normalise PDF
normalisation_factor = 1 / (trapezium(R_source_Rsol, pdf_load))
pdf_normed = normalisation_factor * pdf_load

# Linearly interpolated version of the PDF of source radii
def pdf_source_radii(R_source):
    return np.interp(R_source, R_source_pc, pdf_normed, left=0, right=0)

r_source = np.linspace(0, 20 * r_sol, 100)

plt.plot(R_source_pc, pdf_normed, label='Extracted')
plt.plot(r_source, pdf_source_radii(r_source), label='Interpolated', linestyle='dotted')
plt.xlabel('$p(R_*)$ [pc]')
plt.ylabel('$p(R_*)$')
plt.legend()