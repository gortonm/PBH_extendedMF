#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 16:12:48 2023

@author: ppxmg2
"""
# Short script to run BlackHawk and Isatis for extagalctic gamma ray photon constraints calculated using PYTHIA or Hazma hadronisation.

import numpy as np
import os

Hazma = True

if Hazma:
    hadronisation = "Hazma"
else:
    hadronisation = "PYTHIA"

# Path to BlackHawk and Isatis
BlackHawk_path = os.path.expanduser('~') + "/Downloads/version_finale/"
Isatis_path = BlackHawk_path + "scripts/Isatis/"
os.chdir(BlackHawk_path)

filename_params_Isatis = "parameters_EXGB_%s.txt" % hadronisation
runs_filename = "runs_EXGB_%s.txt" % hadronisation
    

#for i in range(1, 31):
for i in range(1, 6):
   
    destination_folder = "EXGB_%s_{:.0f}".format(i) % hadronisation
    
    # Run BlackHawk
    os.chdir(BlackHawk_path)
    command = "./BlackHawk_tot.x " + "scripts/Isatis/BH_launcher/" + destination_folder + ".txt<input.txt"
    os.system(command)
      
# Run Isatis
os.chdir("./scripts/Isatis")
command = "./Isatis.x %s ./BH_launcher/%s" % (filename_params_Isatis, runs_filename)
os.system(command)
