#!/home/alex/miniconda3/envs/torch/bin/python

import PyLTSpice as lt
import numpy as np
from numpy.fft import fft, fftfreq
from config import asc_path, temp_path
from PyLTSpice.log.ltsteps import LTSpiceLogReader
from PyLTSpice import SimRunner, LTspice, RawRead
import matplotlib.pyplot as plt

'''
net = lt.AscEditor(asc_path)
net.set_parameter("Vto", 1)
net.save_netlist(asc_path)
'''

'''
log = LTSpiceLogReader("./master/circuit_sim/class_A_amp.log")
meas_names = log.get_measure_names()
print(meas_names)
'''

'''
def callback_function(raw_file, log_file):
    print(log_file)

net = lt.AscEditor(asc_path)
net.add_instruction('.tran 0 1 0.5')
#net.add_instruction('.ac dec 100 10 1G')
#net.add_instruction(".save v(vo)")

runner = SimRunner(output_folder='./testing/circuit_sim/temp_files/', simulator=LTspice)
runner.run(net, callback=callback_function)

net.add_instruction('.ac dec 100 10 1G')
runner.run(net, callback=callback_function)
'''

import linecache
linecontent = linecache.getline(f'{temp_path}class_A_amp_1.log', 32)
print(linecontent[29:-2])