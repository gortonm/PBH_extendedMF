#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 11:49:23 2022

@author: ppxmg2
"""

import pytest
import numpy as np
from reproduce_extended_MF import log_normal_MF, left_riemann_sum

class TestClass:
    def test_log_normal_MF(self):
        sigma = 2
        f_pbh = 0.5
        m_c = 1e-10
        m = 1e-11
        assert pytest.approx(log_normal_MF(f_pbh, m, m_c)) == 5.140755370e9 
        
    def test_left_riemann_sum(self):
        x = np.array([0, 1, 3, 5])
        y = np.array([1, 3, 10, 26])
        assert pytest.approx(left_riemann_sum(y, x)) == 27