#!/home/alex/miniconda3/envs/torch/bin/python

import PyLTSpice as lt
from PyLTSpice import SimRunner, LTspice
from PyLTSpice.log.ltsteps import LTSpiceLogReader
import numpy as np
import matplotlib.pyplot as plt
import os
from config import n_datapoints, asc_path, data_file
os.environ["WINEDEBUG"] = "-all"

# X = [VDD, CL, frequency, gain, Vin]
# y = [R1, RD, Vto, Kp]

os.system("rm ./circuit_sim/temp_files/*")
data = np.zeros((n_datapoints, 9)) 


def processing_data(raw_file, log_file, i, VDD, CL, f, Vin, R1, RD, Vto, Kp):
    global data
    read_log = LTSpiceLogReader(log_file)
    peakvo = read_log.get_measure_value("peakvo")
    # gain in V/V, not dB
    gain = peakvo/Vin
    data[i] = [VDD, CL, f, gain, Vin, R1, RD, Vto, Kp]
    print(f"Data point {i}/{n_datapoints} collected.")

net = lt.AscEditor(asc_path)
net.add_instruction(".save v(vo)")
runner = SimRunner(output_folder="./circuit_sim/temp_files", simulator=LTspice, parallel_sims=10)

for i in range(n_datapoints):
    params = {
        "VDD": np.random.uniform(5.0, 20.0),
        "CL": np.random.uniform(1e-9, 10e-9),
        "f": np.random.uniform(1e3, 10e3),
        "Vin": np.random.uniform(0.01, 1.0),
        "R1": np.random.uniform(1e6, 5e6),
        "RD": np.random.uniform(10e3, 1e6),
        "Vto": np.random.uniform(0.2, 1.0),
        "Kp": np.random.uniform(0.01e-3, 0.2e-3)
    }
    for key, value in params.items():
        if key == "Vto" or key == "Kp" or key == "f" or key == "Vin":
            net.set_parameter(key, value)
        else:
            net.set_component_value(key, value)
    
    runner.run(net, callback=processing_data, callback_args=(i, params["VDD"], params["CL"], params["f"], params["Vin"], params["R1"], params["RD"], params["Vto"], params["Kp"]))
runner.wait_completion()
np.save(data_file, data)

os.system("rm ./circuit_sim/temp_files/*")

# Add VDD, Gain, frequency, and CL as input
# Replace C2 and RL with CL connected to Vo and gnd X
# CL range [1nF, 10nF]
# Frequency range [1-10 kHz]
# Gain range [2 to 15 V/V]

# Look at paper to see what X/y features were used