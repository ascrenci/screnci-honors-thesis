#!/home/alex/miniconda3/envs/torch/bin/python

import torch
import numpy as np
import joblib
from model_classes import ClassAModel
from config import n_datapoints, branch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained forward model
model = ClassAModel(input_dim=6, output_dim=3, hidden_dim=32).to(device)
model.load_state_dict(torch.load(f"./testing/class_a_model_4_forward_{n_datapoints}.pth", map_location=device))
model.eval()
for p in model.parameters():
    p.requires_grad = False   # we optimize inputs, not weights

x_scaler = joblib.load("./testing/x_scaler_forward_1000.pkl")
y_scaler = joblib.load("./testing/y_scaler_forward_1000.pkl")


def design(target_maxVo, target_gain, target_THD, CL, f, Vin,
           bounds=None, n_restarts=10, n_steps=500, lr=0.05):
    """
    Find (R1, RD, VDD) such that the forward model predicts
    (maxVo, gain, THD) close to the targets.
    Returns best parameters across multiple random restarts.
    """
    if bounds is None:
        # log10 bounds matching your training data ranges
        bounds = {
            "R1":  (np.log10(100e3), np.log10(2e6)),
            "RD":  (np.log10(1e3),   np.log10(10e3)),
            "VDD": (np.log10(5.0),   np.log10(20.0)),
        }

    # Target in scaled log-space (what the model outputs)
    target_log = np.log10([[target_maxVo, target_gain, target_THD]])
    target_scaled = torch.tensor(
        y_scaler.transform(target_log), dtype=torch.float32, device=device
    )

    # Fixed inputs in scaled log-space (CL, f, Vin)
    fixed_log = np.log10([[CL, f, Vin]])

    best_loss = float("inf")
    best_params = None

    for restart in range(n_restarts):
        # Random starting point inside bounds (log-space)
        R1_log  = np.random.uniform(*bounds["R1"])
        RD_log  = np.random.uniform(*bounds["RD"])
        VDD_log = np.random.uniform(*bounds["VDD"])

        # Build full input in log-space [R1, RD, VDD, CL, f, Vin]
        input_log = np.concatenate(
            [[[R1_log, RD_log, VDD_log]], fixed_log], axis=1
        )
        input_scaled = x_scaler.transform(input_log)

        # We optimize only the first 3 dims (params), keep last 3 fixed
        params = torch.tensor(
            input_scaled[0, :3], dtype=torch.float32,
            device=device, requires_grad=True
        )
        fixed = torch.tensor(
            input_scaled[0, 3:], dtype=torch.float32, device=device
        )

        opt = torch.optim.Adam([params], lr=lr)

        for step in range(n_steps):
            opt.zero_grad()
            x = torch.cat([params, fixed]).unsqueeze(0)
            pred = model(x)
            loss = ((pred - target_scaled) ** 2).sum()
            loss.backward()
            opt.step()

            # Project back into bounds (in scaled space)
            with torch.no_grad():
                # Convert bounds to scaled space using the scaler
                lo = x_scaler.transform([[
                    bounds["R1"][0], bounds["RD"][0], bounds["VDD"][0],
                    fixed_log[0,0], fixed_log[0,1], fixed_log[0,2]
                ]])[0, :3]
                hi = x_scaler.transform([[
                    bounds["R1"][1], bounds["RD"][1], bounds["VDD"][1],
                    fixed_log[0,0], fixed_log[0,1], fixed_log[0,2]
                ]])[0, :3]
                params.clamp_(
                    torch.tensor(lo, dtype=torch.float32, device=device),
                    torch.tensor(hi, dtype=torch.float32, device=device),
                )

        if loss.item() < best_loss:
            best_loss = loss.item()
            # Convert back to real-world units
            full_scaled = torch.cat([params, fixed]).cpu().numpy().reshape(1, -1)
            full_log    = x_scaler.inverse_transform(full_scaled)
            full_real   = 10 ** full_log
            best_params = {
                "R1":  full_real[0, 0],
                "RD":  full_real[0, 1],
                "VDD": full_real[0, 2],
                "loss": loss.item(),
            }

    return best_params


# --- Usage ---
result = design(
    target_maxVo=8.0,
    target_gain=5.0,
    target_THD=2.0,
    CL=10e-9, f=5e3, Vin=8.0/5.0,
    n_restarts=20, n_steps=500
)

print(f"Designed circuit:")
print(f"  R1  = {result['R1']/1e3:.1f} kΩ")
print(f"  RD  = {result['RD']/1e3:.1f} kΩ")
print(f"  VDD = {result['VDD']:.2f} V")
print(f"  Final loss (scaled): {result['loss']:.4f}")