#!/home/alex/miniconda3/envs/torch/bin/python

'''
Next:
- Try diff values for Vt = [0.2V, 1V], Kp = [.01m, .2m]
'''

import PyLTSpice as lt
from PyLTSpice import SimRunner, LTspice
from PyLTSpice.log.ltsteps import LTSpiceLogReader
import numpy as np
import matplotlib.pyplot as plt
import os
from config import n_datapoints, asc_path, data_file
os.environ["WINEDEBUG"] = "-all"

# Fixed values: R2, C1, C2, NMOS width
# Input: RL, maxVo
# Output: VDD, RD, R1, Vto, Kp

os.system("rm ./circuit_sim/temp_files/*")
data = np.zeros((n_datapoints, 7)) # shape: (n, 7) for (X,y)
# X = [RL, maxVo], y = [VDD, RD, R1, Vto, Kp]

def processing_data(raw_file, log_file, i, RL, VDD, RD, R1, Vto, Kp):
    global data
    read_log = LTSpiceLogReader(log_file)
    peakvo = read_log.get_measure_value("peakvo")
    data[i] = [RL, peakvo, VDD, RD, R1, Vto, Kp]
    print(f"Data point {i}/{n_datapoints} collected.")


'''
Code Flow:
1. Generate range of values or random values for RL, VDD, RD, R1
2. Set params for each combination of values
3. Run simulations, extract values in callback function
4. Use threadlocking for data variable
5. Save data to np file
'''

net = lt.AscEditor(asc_path)
#net.add_instruction(".options plotwinsize=0")
net.add_instruction(".save v(vo)")
runner = SimRunner(output_folder="./circuit_sim/temp_files", simulator=LTspice, parallel_sims=10)

for i in range(n_datapoints):
    comps = {
        "R1": np.random.uniform(1e6, 5e6),
        "RD": np.random.uniform(10e3, 1e6),
        "RL": np.random.uniform(10e3, 1e6),
        "VDD": np.random.uniform(5.0, 20.0),
        "Vto": np.random.uniform(0.2, 1.0),
        "Kp": np.random.uniform(0.01e-3, 0.2e-3)
    }
    for key, value in comps.items():
        if key == "Vto" or key == "Kp":
            net.set_parameter(key, value)
        else:
            net.set_component_value(key, value)
    
    runner.run(net, callback=processing_data, callback_args=(i, comps["RL"], comps["VDD"], comps["RD"], comps["R1"], comps["Vto"], comps["Kp"]))
runner.wait_completion()
np.save(data_file, data)

# Make VDD input
# Add Gain, frequency, and CL as input
# Replace C2 and RL with CL connected to Vo and gnd
# CL range [1nF, 10nF]
# Frequency range [1-10 kHz]
# Gain range [2 to 15 V/V]

# Look at paper to see what X/y features were used