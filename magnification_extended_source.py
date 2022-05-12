#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 10:29:44 2022

@author: ppxmg2
"""

import numpy as np
from scipy.special import ellipe, ellipk
from sympy.functions.special.elliptic_integrals import elliptic_pi
import pytest
import matplotlib.pyplot as plt
import matplotlib as mpl

r = 1


def n(u):
    return 4 * u * r / (u + r) ** 2


def k(u):
    return np.sqrt(4 * n(u) / (4 + (u - r) ** 2))


def mu_pointsource(u):
    return (2 + u ** 2) / (u * np.sqrt(4 + u ** 2))


def mu(u):
    if r == 0:
        return mu_pointsource(u)

    a = (u + r) * np.sqrt(4 + (u - r) ** 2) / (2 * r ** 2)
    b = ((u - r) / r ** 2) * (4 + (u ** 2 - r ** 2) / 2) / np.sqrt(4 + (u - r) ** 2)
    c = (
        2
        * ((u - r) ** 2 / (r ** 2 * (u + r)))
        * (1 + r ** 2)
        / np.sqrt(4 + (u - r) ** 2)
    )
    m = k(u) ** 2
    return (1 / np.pi) * (
        ellipe(m) * a - ellipk(m) * b + elliptic_pi(n(u), np.pi / 2, m) * c
    )


def findroot(f, a, b, tolerance, n_max):
    n = 1
    while n <= n_max:
        c = (a + b) / 2
        if f(c) == 0 or abs((b - a) / 2) < tolerance:
            return c
            break
        n += 1

        # set new interval
        if np.sign(f(c)) == np.sign(f(a)):
            a = c
        else:
            b = c
    print("Method failed")


def findroot2(f, a, b, tolerance, n_max):
    n = 1
    while n <= n_max:
        c = 10 ** ((np.log10(a) + np.log10(b)) / 2)

        if f(c) == 0 or abs((b - a) / 2) < tolerance:
            return c
            break
        n += 1

        # set new interval
        if np.sign(f(c)) == np.sign(f(a)):
            a = c
        else:
            b = c
    print("Method failed")


def findroot3(f, a, b, tolerance, n_max):
    n = 1
    while n <= n_max:
        c = 10 ** ((np.log10(a) + np.log10(b)) / 2)

        if abs(f(c)) < tolerance:
            return c
            break
        n += 1

        # set new interval
        if np.sign(f(c)) == np.sign(f(a)):
            a = c
        else:
            b = c
    print("Method failed")


def findroot4(f, a, b, tolerance, n_max):
    n = 1
    while n <= n_max:
        c = (a + b) / 2

        print('a = ', a)
        print('b = ', b)
        print('c = ', c)
        print('f(a) = ', f(a))
        print('f(b) = ', f(b))

        if abs(f(c)) < tolerance:
            return c
            break
        n += 1

        # set new interval
        if np.sign(f(c)) == np.sign(f(a)):
            a = c
        else:
            b = c
    print("Method failed")


def root_func(u):
    return mu(u) - 1.34

filepath = './Data_files/'
def load_data(filename):
    return np.genfromtxt(filepath+filename, delimiter=',', unpack=True)

# Specify the plot style
mpl.rcParams.update({"font.size": 14, "font.family": "serif"})
mpl.rcParams["xtick.major.size"] = 7
mpl.rcParams["xtick.major.width"] = 1
mpl.rcParams["xtick.minor.size"] = 3
mpl.rcParams["xtick.minor.width"] = 1
mpl.rcParams["ytick.major.size"] = 7
mpl.rcParams["ytick.major.width"] = 1
mpl.rcParams["ytick.minor.size"] = 3
mpl.rcParams["ytick.minor.width"] = 1
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["lines.linewidth"] = 1.5
mpl.rcParams["xtick.top"] = False
mpl.rcParams["ytick.right"] = False
mpl.rcParams["font.family"] = "serif"
mpl.rc("text", usetex=True)
mpl.rcParams["legend.edgecolor"] = "lightgrey"



# Plot u_1.34 as a function of the source radius, for a point lens
r_values = np.linspace(0., 2.5, 1000)
u_134_values = []
for r in r_values:
    if r == 0:
        u_134 = 1.
    if r > 2.2405:
        u_134 = 0.
    else:
        u_134 = findroot(root_func, a=0, b=2, tolerance=1e-4, n_max=1000)
    u_134_values.append(u_134)

np.savetxt('Data_files/u_134.csv', np.column_stack((r_values, u_134_values)), delimiter=',')

plt.figure()
plt.plot(r_values, u_134_values, label='Directly calculated')
plt.xlabel("$r_\mathrm{S}$")
plt.ylabel("$u_{1.34}$")
plt.ylim(0, 2)

# Function for u_134 with r_S:
r_values_load, u_134_values_load = load_data('u_134.csv')
def u_134(r_S):
    return np.interp(r_S, r_values_load, u_134_values_load, left=1, right=0)

plt.plot(r_values, u_134(r_values), label='Interpolated', linestyle='dotted')
plt.legend()

# Find threshold amplification at one solar radius
print(u_134(r_S = 1))

# Reproduce Fig. 3 of Smyth+ '20
from reproduce_extended_MF import einstein_radius

r_sol = 2.25461e-8  # solar radius, in pc
d_s = 770e3  # LMC distance, in pc

dL_values_m = 10 ** np.linspace(17, 22, 200)  # lens distance, in m
dL_values = dL_values_m / (3.0857e16)  # lens distance, in pc
Rs_values = r_sol * np.array([1, 2.5])
colours = ["b", "r"]

plt.figure(figsize=(6, 5))
for i, Rs in enumerate(Rs_values):
    m_pbh = 1e-10
    u_134_values = []
    for dL in dL_values:
        x = dL / d_s
        r = x * Rs / einstein_radius(x, m_pbh=m_pbh)
        u_134 = findroot3(root_func, a=1e-15, b=2, tolerance=3e-5, n_max=10000)
        u_134_values.append(u_134)

    plt.plot(dL_values_m, u_134_values, color=colours[i])

    m_pbh = 1e-11
    u_134_values = []
    for dL in dL_values:
        x = dL / d_s
        r = x * Rs / einstein_radius(x=dL / d_s, m_pbh=m_pbh)
        u_134 = findroot3(root_func, a=1e-15, b=2, tolerance=3e-5, n_max=10000)
        u_134_values.append(u_134)

    plt.plot(dL_values_m, u_134_values, color=colours[i], linestyle="dashed")

plt.ylim(0, 2.5)
plt.xscale("log")
plt.xlim(min(dL_values_m), max(dL_values_m))
plt.xlabel("$D_L~[\mathrm{m}]$")
plt.ylabel("$u_{1.34}$")
plt.tight_layout()


# Reproduce RH panel of Fig. 1 of Witt & Mao '94

# Source radius
r = 0.1
v = 1
t_vals = np.linspace(-1, 1, 1000)

fig = plt.figure(figsize=(3.5, 7))
gs = fig.add_gridspec(3, ncols=1, hspace=0)
axes = gs.subplots(sharex=True)

y_max = [25, 15, 6]

for i, b in enumerate([0.05, 0.1, 0.2]):

    ax = axes[i]

    mu_vals = []
    mu_pointsource_vals = []
    for t in t_vals:

        u = np.sqrt(b ** 2 + (v * t) ** 2)

        mu_vals.append(mu(u))
        mu_pointsource_vals.append(mu_pointsource(u))

    ax.plot(t_vals, mu_pointsource_vals)
    ax.plot(t_vals, mu_vals, label="$b = {:.2f}$".format(b))
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, y_max[i])
    ax.set_xticks(np.linspace(-1, 1, 5))

    if i == 0:
        ax.set_yticks(np.linspace(0, y_max[i], 6))

    else:
        ax.set_yticks(np.linspace(0, y_max[i], 4))

    ax.set_xlabel("$t$")
    ax.set_ylabel("$\mu$")
    ax.legend(fontsize="small")

plt.tight_layout()

r = 1
assert pytest.approx(n(u=2)) == 8 / 9
assert pytest.approx(k(u=2)) == 2 * np.sqrt(8 / 45)
