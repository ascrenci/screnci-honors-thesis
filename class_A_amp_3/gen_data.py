#!/home/alex/miniconda3/envs/torch/bin/python

import PyLTSpice as lt
from PyLTSpice import SimRunner, LTspice
from PyLTSpice.log.ltsteps import LTSpiceLogReader
import numpy as np
import matplotlib.pyplot as plt
import os
from config import n_datapoints, asc_path, temp_path, data_file
from linecache import getline
os.environ["WINEDEBUG"] = "-all"

# X = [gain, bandwidth, VDD, Vto, THD=1]
# y = [R1, RD, Kp, CL]

os.system(f"rm {temp_path}*")
data = np.zeros((n_datapoints, 9))

net = lt.AscEditor(asc_path)
net.add_instruction(".save v(vo)")
runner = SimRunner(output_folder=temp_path, simulator=LTspice, parallel_sims=10)

def processing_data(raw_file, log_file, analysis_mode, i, VDD, R1, RD, CL, Vto, Kp):
    global data

    if analysis_mode == "TRAN":
        THD = float(getline(f'./{log_file}', 32)[29:-2])
        data[i, 2:] = [VDD, Vto, THD, R1, RD, Kp, CL]
    elif analysis_mode == "AC":
        read_log = LTSpiceLogReader(log_file)
        gain = 10**(np.abs(read_log.get_measure_value("a0"))/20)
        bandwidth = np.abs(read_log.get_measure_value("f_c"))
        data[i, :2] = [gain, bandwidth]



for i in range(n_datapoints):
    params = {
        "VDD": np.random.uniform(5.0, 20.0),
        "R1": np.random.uniform(100e3, 1e6),
        "RD": np.random.uniform(500e3, 1e6),
        "CL": np.random.uniform(1e-9, 10e-9),
        "Vto": np.random.uniform(0.2, 1.0),
        "Kp": np.random.uniform(0.01e-3, 0.2e-3)
    }

    for key, value in params.items():
        if key == "Vto" or key == "Kp":
            net.set_parameter(key, value)
        else:
            net.set_component_value(key, value)
        
    net.add_instruction('.tran 0 11 10')
    runner.run(net, callback=processing_data, callback_args=(
        "TRAN", i, params["VDD"], params["R1"], params["RD"], params["CL"], params["Vto"], params["Kp"]
    ))

    net.add_instruction('.ac dec 100 10 1G')
    runner.run(net, callback=processing_data, callback_args=(
        "AC", i, params["VDD"], params["R1"], params["RD"], params["CL"], params["Vto"], params["Kp"]
    ))
    print(f"Data point {i+1}/{n_datapoints} collected.")
    

runner.wait_completion()
np.save(data_file, data)

os.system(f"rm {temp_path}*")