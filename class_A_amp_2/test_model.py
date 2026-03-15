#!/home/alex/miniconda3/envs/torch/bin/python

import torch
import PyLTSpice as lt
from PyLTSpice import SimRunner, LTspice
from PyLTSpice.log.ltsteps import LTSpiceLogReader
import numpy as np
from model_classes import ClassAModel
from config import asc_path
from config import n_datapoints
import joblib
import os
os.environ["WINEDEBUG"] = "-all"

# Load dataset and MinMaxScalers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ClassAModel().to(device)
model.load_state_dict(torch.load(f"class_a_model_2_{n_datapoints}.pth"))
model.eval()

# Input array for testing: [RL, maxvo]
RL = 10e3
maxVo = 100e-3
x_scaler, y_scaler = joblib.load("x_scaler.pkl"), joblib.load("y_scaler.pkl")
X_test = x_scaler.transform(np.log10(np.array([[RL, maxVo]])))
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

with torch.no_grad():
    y_pred = model(X_test).cpu().numpy()

# Inverse transform output
y_pred_inv = y_scaler.inverse_transform(y_pred)
y_pred_real = 10**y_pred_inv[0] # VDD, RD, R1, Vto, Kp

print(f"VDD: {y_pred_real[0]:.2f} V | RD: {y_pred_real[1]/1e3:.2f} kOhm | R1: {y_pred_real[2]/1e3:.2f} kOhm | Vto: {y_pred_real[3]:.2f} V | Kp: {y_pred_real[4]*1e3:.4f} mA/V^2")

net = lt.AscEditor(asc_path)
net.set_component_value("VDD", y_pred_real[0])
net.set_component_value("RD", y_pred_real[1])
net.set_component_value("R1", y_pred_real[2])
net.set_component_value("RL", RL)
net.set_parameter("Vto", y_pred_real[3])
net.set_parameter("Kp", y_pred_real[4])
net.save_netlist(asc_path)

def processing_data(raw_file, log_file):
    global maxVo
    maxVo *= 1000
    read_log = LTSpiceLogReader(log_file)
    peakVo = read_log.get_measure_value("peakvo")*1000
    print(f"Simulated maxVo: {peakVo:.2f} mV")
    print(f"Desired maxVo: {maxVo:.2f} mV")
    abs_error = abs(maxVo - peakVo)
    rel_error = abs_error/abs(maxVo)*100
    print(f"Abs error: {abs_error:.2f} mV")
    print(f"Relative Error: {rel_error:.2f}%")

runner = SimRunner(output_folder="./circuit_sim/temp_files", simulator=LTspice)
runner.run(net, callback=processing_data)