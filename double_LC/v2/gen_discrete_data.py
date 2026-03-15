#!/home/alex/miniconda3/envs/torch/bin/python

import numpy as np
import config

E12 = [1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2]

R_decades = np.array([10**i for i in range(3, 4)]) 
R_vals = np.array([e*r for r in R_decades for e in E12])

C_decades = np.array([10**i for i in range(-7,-5)])
C_vals = np.array([e*c for c in C_decades for e in E12])

L_decades = np.array([10**i for i in range(-5,-2)])
L_vals = np.array([e*l for l in L_decades for e in E12])

print(len(R_vals), len(L_vals), len(C_vals))