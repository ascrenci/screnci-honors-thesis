#!/home/alex/miniconda3/envs/torch/bin/python

import torch
import PyLTSpice as lt
from PyLTSpice import SimRunner, LTspice
from PyLTSpice.log.ltsteps import LTSpiceLogReader
import numpy as np
from model_classes import ClassAModel, ResBlockMLP_Predictor
from config import asc_path, n_datapoints, branch, temp_path
import joblib
import os
from linecache import getline
os.environ["WINEDEBUG"] = "-all"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset and MinMaxScalers
#model = ResBlockMLP_Predictor(6,4,128).to(device)
model = ClassAModel(6,4,256).to(device)
model.load_state_dict(torch.load(f"{branch}/class_a_model_3_{n_datapoints}.pth", map_location=device))
for param in model.parameters():
    param.requires_grad = False
model.eval()

# X = [gain, bandwidth, VDD, Vto, THD=1]
# y = [R1, RD, Kp, CL]

# input array for testing
gain = np.random.uniform(5.0, 15.0)
bandwidth = np.random.uniform(100, 200)
VDD = np.random.uniform(5.0, 20.0)
Vin = np.random.uniform(10e-3, 200e-3)
Vto = np.random.uniform(0.5, 1.0)
THD = 1.0


print(f"\nTest input parameters:")
print(f"Gain: {gain:.2f} V/V | Bandwidth: {bandwidth:.2f} Hz | VDD: {VDD:.2f} V | Vin: {Vin*1000:.2f} mV | Vto: {Vto:.2f} V | THD: {THD:.2f} %")

data_scaler = joblib.load(f"{branch}/data_scaler_{n_datapoints}.pkl")
X_test = np.log10(np.array([[gain, bandwidth, VDD, Vin, Vto, THD]]))
temp_arr = np.zeros((1, 10))
temp_arr[:,:6] = X_test
temp_arr = data_scaler.transform(temp_arr)
X_test = temp_arr[:,:6]


X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

with torch.no_grad():
    y_pred = model(X_test).cpu().numpy()

# Inverse transform output
# y = [R1, RD, Kp, CL]
temp_arr = np.zeros((1,10))
temp_arr[0,6:] = y_pred[0]
temp_arr = data_scaler.inverse_transform(temp_arr)
y_pred_real = 10**temp_arr[0,6:]
#y_pred_real[:2] = 10**y_pred_real[:2]
#y_pred_real = 10**y_pred[0]

print("\nPredicted parameters:")
print(f"R1: {y_pred_real[0]/1000:.2f} kOhms | RD: {y_pred_real[1]/1000:.2f} kOhms | Kp: {y_pred_real[2]*1e3:.4f} mA/V^2 | CL: {y_pred_real[3]*1e9:.2f} nF")

net = lt.AscEditor(asc_path)
# Set X data in netlist
net.set_component_value("VDD", VDD)
net.set_parameter("Vin", Vin)
net.set_parameter("Vto", Vto)

# Set y data in netlist
net.set_component_value("R1", y_pred_real[0])
net.set_component_value("RD", y_pred_real[1])
net.set_parameter("Kp", y_pred_real[2])
net.set_component_value("CL", y_pred_real[3])
net.save_netlist(asc_path)

def processing_data(raw_file, log_file, analysis_mode):
    global gain, bandwidth
    if analysis_mode == "TRAN":
        THD = float(getline(f'./{log_file}', 32)[29:-2])
        print(f"\nMeasured THD: {THD}% | Error: {abs(THD-1.0):.2f}%")

    if analysis_mode == "AC":
        read_log = LTSpiceLogReader(log_file)
        gain_measured = np.abs(read_log.get_measure_value("a0"))
        bandwidth_measured = np.abs(read_log.get_measure_value("f_c"))
        print(f"Measured Gain: {gain_measured:.2f} V/V | Measured Bandwidth: {bandwidth_measured:.2f} Hz")
        print(f"\nGain Error: {abs(gain_measured-gain):.2f} V/V | Bandwidth Error: {abs(bandwidth_measured-bandwidth):.2f} Hz")
        print(f"Gain % Error: {abs(gain_measured-gain)/gain*100:.2f}% | Bandwidth % Error: {abs(bandwidth_measured-bandwidth)/bandwidth*100:.2f}%")

runner = SimRunner(output_folder=temp_path, simulator=LTspice)

net.add_instruction('.tran 0 11 10')
runner.run(net, callback=processing_data, callback_args=("TRAN",))
runner.wait_completion()

net.add_instruction('.ac dec 100 10 1G')
runner.run(net, callback=processing_data, callback_args=("AC",))
runner.wait_completion()