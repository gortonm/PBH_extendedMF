#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 09:53:09 2022

@author: ppxmg2
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

#Specify the plot style
mpl.rcParams.update({'font.size': 16,'font.family':'serif'})
mpl.rcParams['xtick.major.size'] = 7
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.size'] = 3.5
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.major.size'] = 7
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.size'] = 3.5
mpl.rcParams['ytick.minor.width'] = 1
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rc('text', usetex=True)

mpl.rcParams['legend.edgecolor'] = 'inherit'

m_pbh_values = 10**np.linspace(np.log10(5e14), 17, 25)
ratio_A262 = np.array([2.05599975,  2.14428295,  2.17387688,  2.1591015,   2.12387427,  2.09923875,  2.07381181,  2.05564418,  2.04252848,  2.01069942,  2.00403147,  1.98297137,  1.94130077,  1.91468437,  1.90242808,  1.8619396,   1.82395022,  1.77354206, 1.70587374,  1.62688725,  1.51801674,  1.43275683,  3.57560236, 10.99378365, 42.25292111])
ratio_NGC5044 = np.array([1.96577299, 2.08576116, 2.07139375, 2.09065528, 2.1376123, 2.02382004, 2.11281174, 1.96910581, 1.94094256, 1.91722722, 1.8918004, 1.88164215, 1.86606619, 1.82650375, 1.8107449, 1.77490641, 1.72844987, 1.64791093, 1.59931172, 1.53920723, 3.17042457, 7.53207337, 19.78336299, 60.73663435, 233.18256366])

plt.plot(m_pbh_values, ratio_A262-2, 'x', label="A262")
plt.plot(m_pbh_values, ratio_NGC5044-2, 'x', label="NGC5044")
plt.xlabel("$M_\mathrm{PBH}$ [g]")
plt.ylabel("$f_\mathrm{PBH, calculated} / f_\mathrm{PBH, extracted}$ - 2")
plt.xscale('log')
plt.ylim(-0.5, 0.5)
plt.legend()
plt.tight_layout()