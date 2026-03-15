#!/home/alex/miniconda3/envs/torch/bin/python

import numpy as np
import matplotlib.pyplot as plt
import config

vals = np.linspace(0.1,10, 200)
#R_vals = np.array([i for i in vals if i >= 1])

def randVals():
    R = [np.random.choice(vals)*10**3]
    L = np.empty(2)
    C = np.empty(2)

    for i in range(2):
        L[i] = np.random.choice(vals)*10**np.random.randint(-9,1)
        C[i] = np.random.choice(vals)*10**np.random.randint(-12,1)
    
    return np.concatenate([R,L,C])

f = np.logspace(np.log10(10), np.log10(10E9), config.freq_length)
s = 1j*2*np.pi*f

data = np.empty((config.n_datapoints, 2, config.freq_length+5))

for i in range(config.n_datapoints):
    R, L1, L2, C1, C2 = randVals()

    hf = (C1*s)/((C1*L1*s**2 + 1)*(1/R + (C1*s)/(C1*L1*s**2 + 1) + (C2*s)/(C2*L2*s**2 + 1)))
    mag = np.abs(hf)
    mag = np.maximum(mag, 1E-12)
    phase = np.angle(hf)
    phase = np.unwrap(phase)

    data[i,0,:5] = np.array([R, L1, L2, C1, C2])
    data[i,0,5:] = mag
    data[i,1,5:] = phase

np.save('data.npy', data)