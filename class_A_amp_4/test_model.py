#!/home/alex/miniconda3/envs/torch/bin/python

import torch
import PyLTSpice as lt
from PyLTSpice import SimRunner, LTspice
from PyLTSpice.log.ltsteps import LTSpiceLogReader
import numpy as np
from model_classes import ClassAModel
from config import asc_path, n_datapoints, branch, temp_path, data_file
import joblib
import os
from linecache import getline
import re
os.environ["WINEDEBUG"] = "-all"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data = np.load(f"./{branch}/xval_{n_datapoints}.npy")

model = ClassAModel(5,3,32).to(device)
model.load_state_dict(torch.load(f"./{branch}/class_a_model_4_{n_datapoints}.pth", map_location=device))
for param in model.parameters():
    param.requires_grad = False
model.eval()

# input array for testing
CL = np.random.uniform(1e-9, 10e-9)
f = np.random.uniform(1e3, 10e3)
maxVo = np.random.uniform(4.0, 12.0)
gain = np.random.uniform(1.0, 14.0)
Vin = np.random.uniform(10e-3, 200e-3)

wC = 2*np.pi*f*CL
THD = 1.0

X = np.array([[wC, maxVo, gain, Vin, THD]])

#X = np.array([10**data[np.random.randint(len(data))]])
#CL, f, maxVo, gain, Vin, THD = X[0]

#wC, maxVo, gain, Vin, THD = X[0]
#CL = np.random.uniform(1e-9, 10e-9)
#f = wC/(np.pi*2*CL)

print(f"\nTest input parameters:")
print(f"CL = {CL*1e9:.2f} nF | f = {f/1e3:.2f} KHz | maxVo = {maxVo:.2f} V | gain = {gain:.2f} V/V | Vin = {Vin:.3f} | THD = {THD}")

X_scaler = joblib.load(f"./{branch}/x_scaler_{n_datapoints}.pkl")
X_scaled = X_scaler.transform(np.log10(X))


X_test = torch.tensor(X_scaled, dtype=torch.float32).to(device)

with torch.no_grad():
    y_pred = model(X_test).cpu().numpy()

y_scaler = joblib.load(f"./{branch}/y_scaler_{n_datapoints}.pkl")
y_pred_unscaled = y_scaler.inverse_transform(y_pred)
y_pred_real = 10**y_pred_unscaled[0]
R1, RD, VDD = y_pred_real

print("\nPredicted parameters:")
print(f"R1 = {R1/1000:.2f} kOhms | RD = {RD/1000:.2f} kOhms | VDD = {VDD:.2f}")

net = lt.AscEditor(asc_path)
# Set X data in netlist
net.set_parameter("CL", CL)
net.set_parameter("f", f)
net.set_parameter("R1", R1)
net.set_parameter("RD", RD)
net.set_parameter("VDD", VDD)
net.set_parameter("Vin", Vin)
net.save_netlist(asc_path)

def processing_data(raw_file, log_file):
    global gain, maxVo, Vin, THD

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
    THD_measured = float(m.group(1))

    maxVo_measured = np.abs(read_log.get_measure_value("maxvo"))
    gain_measured = (maxVo_measured - Vd)/Vin

    print("\nSim Results:")
    print(f"Measured maxVo: {maxVo_measured:.2f} V | Measured gain: {gain_measured:.2f} V/V | Measured THD: {THD_measured:.2f}")

    print("\nRel Measured Errors:")
    print(f"maxVo % error: {(maxVo_measured - maxVo)/maxVo*100:.2f} | Gain % error: {(gain_measured - gain)/gain*100:.2f} | THD Error: {(THD_measured - THD)/THD*100:.2f}")

runner = SimRunner(output_folder=temp_path, simulator=LTspice)

runner.run(net, callback=processing_data)
runner.wait_completion()