#!/home/alex/miniconda3/envs/torch/bin/python

import PyLTSpice as lt
from PyLTSpice import SimRunner, LTspice
from PyLTSpice.log.ltsteps import LTSpiceLogReader
import numpy as np
import matplotlib.pyplot as plt
import os
from config import n_datapoints, asc_path, temp_path, data_file
from linecache import getline
import re
os.environ["WINEDEBUG"] = "-all"

"""

Vto = 1.4, Kp = 1m for now

To sim:
Input:   [c, f , Vin, VDD, R1, R2]
Measure: [maxVo, THD, Vd]

gain = (maxVo - Vd)/Vin

User inputs [C, f, maxVo, gain]

Vin = maxVo/gain
wC = 2*pi*f*c
#maxIL = maxVo*wC 

X = [c, f, maxVo, gain, Vin, THD]

y = [R1, RD, VDD]

"""

os.system(f"rm {temp_path}*")
data = []

net = lt.AscEditor(asc_path)
net.add_instruction(".save v(vo)")
runner = SimRunner(output_folder=temp_path, simulator=LTspice, parallel_sims=20)

print(net.get_all_parameter_names())

def processing_data(raw_file, log_file, params):
    global data
    read_log = LTSpiceLogReader(log_file)

    with open(log_file, 'r') as f:
        log = f.read()

    # DC component (your Vd)
    m = re.search(r'DC component:\s*([-\d.eE+]+)', log)
    if m is None:
        raise RuntimeError(f"DC component not found in {log_file}")
    Vd = float(m.group(1))

    # Total Harmonic Distortion (percent)
    m = re.search(r'Total Harmonic Distortion:\s*([-\d.eE+]+)\s*%', log)
    if m is None:
        raise RuntimeError(f"THD not found in {log_file}")
    THD = float(m.group(1))
    maxVo = np.abs(read_log.get_measure_value("maxvo"))

    i = params["i"]
    VDD = params["VDD"]
    Vin = params["Vin"]
    R1 = params["R1"]
    RD = params["RD"]
    CL = params["CL"]
    f = params["f"]
    

    gain = (maxVo - Vd)/Vin
    #wC = 2*np.pi*f*CL

    data.append([CL, f, maxVo, gain, Vin, THD, R1, RD, VDD])


for i in range(n_datapoints):
    params = {
        "VDD": np.random.uniform(9.0, 20.0),
        "Vin": np.random.uniform(10e-3, 200e-3),
        "R1":  np.random.uniform(400e3, 2e6),
        "RD":  np.random.uniform(2e3, 10e3),
        "CL":  np.random.uniform(1e-9, 10e-9),
        "f":   np.random.uniform(1e3, 10e3),
        "i":   i,
    }

    for key, value in params.items():
        net.set_parameter(key, value)
        
    runner.run(net, callback=processing_data, callback_args=(params,))

    print(f"Data point {i+1}/{n_datapoints} collected.")
    

runner.wait_completion()
np.save(data_file, data)

os.system(f"rm {temp_path}*")