#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 14:56:14 2023

@author: Mo Cohen
"""
import numpy as np

source = 1e-07 # kg/kg
radii = [500e-09] # m
densities = [1262] # kg/m3
## Source of 1e-09 gives a number density of 1e08 particles/m3
#densities = [1000, 1262, 1328]
#radii = [1e-09, 5e-09, 10e-09, 30e-09, 50e-09, 60e-09, 80e-09, 100e-09, 
#         500e-09, 1000e-09]

for radius in radii:
    svol = (4/3)*np.pi*radius**3
    for density in densities:
        mpart = svol*density
        numrho = source*(1/mpart)*0.115
        print('Source is ' + str(source))
        print('Radius is ' + str(radius))
        print('Density is ' + str(density))
        print('Number density is ' + str(numrho))
        