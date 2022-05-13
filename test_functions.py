#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 11:49:23 2022

@author: ppxmg2
"""

import pytest
import numpy as np
from reproduce_extended_MF import rho_M31, rho_MW, einstein_radius, log_normal_MF, double_integral, triple_integral, pdf_source_radii, u_134
from dgamma_Niikura import dgamma_integrand_MW

class TestClass:
    
    def test_rho_M31(self):
        assert pytest.approx(rho_M31(x=0.5)) == 1.207926502e-6
        
    def test_rho_MW(self):
        assert pytest.approx(rho_MW(x=0.5)) == 7.336997e-7
    
    def test_einstein_radius(self):
        assert pytest.approx(einstein_radius(x=0.5, m_pbh=10)) == 6.070225671e-4
    
    def test_log_normal_MF(self):
        sigma = 2
        f_pbh = 0.5
        m_c = 1e-10
        m = 1e-11
        assert pytest.approx(log_normal_MF(f_pbh, m, m_c)) == 5.140755370e9 
                
    def test_pdf_source_radii(self):
        assert pdf_source_radii(1e-7) - 0.082293 < 1e-4
        
    def test_u134(self):
        assert pytest.approx(u_134(1)) == 1.205676
  
    def test_dgamma_integrand_MW(self):
        assert pytest.approx(dgamma_integrand_MW(x=0.5, u_min=0.5, m_pbh=10, t_hat=10)) == 1.643873973e-15   
  
    def integration_function_double(self, x, y, k):
        return k * (x**2 + 4*y)
    
    def integration_function_triple(self, x, y, z, k):
        return k * (z*x**2 + 4*y*z**3)

    n_steps = 10000
    
    # Compare output of numerical double integration to exact value
    def test_double_integral(self):
        assert abs(double_integral(self.integration_function_double, 11, 14, 7, 10, self.n_steps, 1) - 1719) < 1

    def test_triple_integral(self):
        assert abs(triple_integral(self.integration_function_triple, 11, 14, 7, 10, 1, 2, self.n_steps, 1) - 3267) < 1
        