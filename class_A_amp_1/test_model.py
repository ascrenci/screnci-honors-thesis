#!/home/alex/miniconda3/envs/torch/bin/python

import torch
import PyLTSpice as lt
import numpy as np
from model_classes import ClassAModel
from config import asc_path
import joblib

# Load dataset and MinMaxScalers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ClassAModel().to(device)
model.load_state_dict(torch.load("class_a_model.pth"))
model.eval()

# Input array for testing: [RL, maxvo]
RL = 1e6
maxVo = 50e-3
x_scaler, y_scaler = joblib.load("x_scaler.pkl"), joblib.load("y_scaler.pkl")
X_test = x_scaler.transform(np.log10(np.array([[RL, maxVo]])))
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

with torch.no_grad():
    y_pred = model(X_test).cpu().numpy()

# Inverse transform output
y_pred_inv = y_scaler.inverse_transform(y_pred)
y_pred_real = 10**y_pred_inv[0] # VDD, RD, R1

print(y_pred_real)

net = lt.AscEditor(asc_path)
net.set_component_value("VDD", y_pred_real[0])
net.set_component_value("RD", y_pred_real[1])
net.set_component_value("R1", y_pred_real[2])
net.set_component_value("RL", RL)
net.save_netlist(asc_path)

'''
Next:
- Try diff values for Vt = [0.2V, 1V], Kp = [.01m, .2m]
'''