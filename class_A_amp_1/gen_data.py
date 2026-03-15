#!/home/alex/miniconda3/envs/torch/bin/python

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
# Output: VDD, RD, R1

os.system("rm ./circuit_sim/temp_files/*")
data = np.zeros((n_datapoints, 5)) # shape: (n, 5) for [RL, maxVo, VDD, RD, R1]
# X = [RL, maxVo], y = [VDD, RD, R1]

def processing_data(raw_file, log_file, i, RL, VDD, RD, R1):
    global data, index, lock
    read_log = LTSpiceLogReader(log_file)
    peakvo = read_log.get_measure_value("peakvo")
    data[i] = [RL, peakvo, VDD, RD, R1]
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
        "VDD": np.random.uniform(5, 20)
    }
    for key, value in comps.items():
        net.set_component_value(key, value)
    
    runner.run(net, callback=processing_data, callback_args=(i, comps["RL"], comps["VDD"], comps["RD"], comps["R1"]))
runner.wait_completion()
np.save(data_file, data)