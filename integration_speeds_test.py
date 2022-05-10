#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 16:27:37 2022

@author: ppxmg2
"""

import numpy as np
from reproduce_extended_MF import left_riemann_sum
import time

def trapezium(y, x):
    area = 0
    for i in range(len(x) - 1):
        area += (x[i+1] - x[i]) * (y[i] + 0.5*(x[i+1]-x[i])*(y[i+1]-y[i]))
    return area

def test_function(x):
    return 471 + 12*x


x = np.linspace(7, 10, 1000)
y = test_function(x)

tolerance = 1e-4
n_steps = 100



t1 = time.time()
left_riemann_sum(y, x)
t2 = time.time()
print('Left Riemann sum: time taken = ', t2-t1)

t1 = time.time()
trapezium(y, x)
t2 = time.time()
print('trapezium(): time taken = ', t2-t1)

t1 = time.time()
np.trapz(y, x)
t2 = time.time()
print('np.trapz(): time taken = ', t2-t1)


# Find time taken to calculate an integral to within a specified degree of accuracy
err = 1
t1 = time.time()
while err > tolerance:
    x = np.linspace(7, 10, n_steps)
    y = test_function(x)
    
    integral = left_riemann_sum(y, x)
    
    if abs(integral - 1719) < tolerance:
        break
    
    n_steps *= 2

t2 = time.time()
print('Left Riemann sum: time taken = ', t2-t1)


err = 1
t1 = time.time()
while err > tolerance:
    x = np.linspace(7, 10, n_steps)
    y = test_function(x)
    
    integral = trapezium(y, x)
    
    if abs(integral - 1719) < tolerance:
        break
    
    n_steps *= 2

t2 = time.time()
print('trapezium: time taken = ', t2-t1)


err = 1
t1 = time.time()
while err > tolerance:
    x = np.linspace(7, 10, n_steps)
    y = test_function(x)
    
    integral = np.trapz(y, x)
    
    if abs(integral - 1719) < tolerance:
        break
    
    n_steps *= 2

t2 = time.time()
print('np.trapz: time taken = ', t2-t1)
