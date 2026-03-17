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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset and MinMaxScalers
model = ClassAModel().to(device)
model.load_state_dict(torch.load(f"class_a_model_3_{n_datapoints}.pth"))
model.eval()

# input array for testing
# X = [VDD, CL, frequency, gain, Vin]

VDD = np.random.uniform(5.0, 20.0)
CL = np.random.uniform(1e-9, 10e-9)
f = np.random.uniform(1e3, 10e3)
gain = np.random.uniform(2.0, 15.0)
Vin = np.random.uniform(0.01, 1.0)

x_scaler, y_scaler = joblib.load("x_scaler.pkl"), joblib.load("y_scaler.pkl")
X_test = x_scaler.transform(np.log10(np.array([[VDD, CL, f, gain, Vin]])))
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

with torch.no_grad():
    y_pred = model(X_test).cpu().numpy()

# Inverse transform output
# y = [R1, RD, Vto, Kp]
y_pred = y_scaler.inverse_transform(y_pred)
y_pred_real = 10**y_pred[0]

print("Predicted parameters:")
print(f"R1: {y_pred_real[0]/1000:.2f} kOhms | RD: {y_pred_real[1]/1000:.2f} kOhms | Vto: {y_pred_real[2]:.2f} V | Kp: {y_pred_real[3]*1e3:.4f} mA/V^2")

net = lt.AscEditor(asc_path)
# Set X data in netlist
net.set_component_value("VDD", VDD)
net.set_component_value("CL", CL)
net.set_parameter("f", f)
net.set_parameter("Vin", Vin)

# Set y data in netlist
net.set_component_value("R1", y_pred_real[0])
net.set_component_value("RD", y_pred_real[1])
net.set_parameter("Vto", y_pred_real[2])
net.set_parameter("Kp", y_pred_real[3])
net.save_netlist(asc_path)

def processing_data(raw_file, log_file):
    global Vin, gain
    read_log = LTSpiceLogReader(log_file)
    peakVo = read_log.get_measure_value("peakvo")
    gain_sim = peakVo/Vin
    print(f"Target gain: {gain:.2f} V/V")
    print(f"Simulated gain: {gain_sim:.2f} V/V")
    
    abs_error = abs(gain - gain_sim)
    rel_error = abs_error/gain
    percent_error = rel_error * 100
    print(f"Absolute error: {abs_error:.4f} V/V")
    print(f"Relative error: {rel_error:.4f}")
    print(f"Percent error: {percent_error:.2f} %")

runner = SimRunner(output_folder="./circuit_sim/temp_files", simulator=LTspice)
runner.run(net, callback=processing_data)