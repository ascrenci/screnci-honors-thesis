import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import Dataset
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from config import branch, n_datapoints
import torch.nn.functional as F

class ClassADataset(Dataset):
    def __init__(self, data_file):
        #cols = [1,5,6,7]
        data = np.log10(np.load(data_file))
        #data[:,cols] = np.log10(data[:, cols])
        data_scaler = joblib.load(f"{branch}/data_scaler_{n_datapoints}.pkl")
        data = data_scaler.transform(data)
        
        X = data[:, :6] # X = [gain, bandwidth, VDD, Vin, Vto, THD=1]
        y = data[:, 6:] # y = [R1, RD, Kp, CL]

        """
        X_norm = self.xscaler.fit_transform(X)
        y_norm = self.yscaler.fit_transform(y)
        joblib.dump(self.xscaler, f"{branch}/xscaler_{n_datapoints}.pkl")
        joblib.dump(self.yscaler, f"{branch}/yscaler_{n_datapoints}.pkl")
        """

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ClassAModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

# X = [VDD, Vin, Vto, R1, RD, Kp, CL] 7
# y = [gain, bandwidth, THD] 3

#data = [gain, bandwidth, VDD, Vin, Vto, THD, R1, RD, Kp, CL]

class PerformancePredictorDataset(Dataset):
    def __init__(self, datafile):
        #cols = [1,5,6,7]
        data = np.log10(np.load(datafile))
        #data[:,cols] = np.log10(data[:,cols])
        data_scaler = MinMaxScaler()
        data = data_scaler.fit_transform(data)
        joblib.dump(data_scaler, f"{branch}/data_scaler_{n_datapoints}.pkl")

        X = np.zeros((n_datapoints, 7))
        y = np.zeros((n_datapoints, 3))

        y[:, :2] = data[:, :2] # gain, bandwidth
        y[:, 2] = data[:, 5] # THD

        X[:, :3] = data[:, 2:5] # VDD, Vin, Vto
        X[:, 3:] = data[:, 6:] # R1, RD, Kp, CL

        """
        xscaler = MinMaxScaler()
        yscaler = MinMaxScaler()

        X_norm = xscaler.fit_transform(X)
        y_norm = yscaler.fit_transform(y)
        joblib.dump(xscaler, f"{branch}/xscaler_predictor_{n_datapoints}.pkl")
        joblib.dump(yscaler, f"{branch}/yscaler_predictor_{n_datapoints}.pkl")
        """

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ResBlock_Surrogate(nn.Module):
    def __init__(self, hidden_dim, dropout_p=0.05):
        super().__init__()

        self.res = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1), # Allows gradients in the 'off' state
            nn.Dropout(dropout_p), # Regularizes against SPICE simulation noise
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

    def forward(self, x):
        # Crucial: No ReLU on the final addition. 
        # This allows the block to refine the prediction in both directions.
        return F.leaky_relu(self.res(x) + x, 0.1)

class ResBlockMLP_Surrogate(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()

        self.initial_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1)
        )

        # Using two blocks provides enough capacity for Gain, BW, and THD simultaneously
        self.res_blocks = nn.Sequential(
            ResBlock_Surrogate(hidden_dim),
            ResBlock_Surrogate(hidden_dim)
        )

        # Final output head - No activation here if using StandardScaler or MinMaxScaler
        # If you want to ensure positive outputs only, you can use nn.Softplus()
        self.output_head = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.initial_layer(x)
        x = self.res_blocks(x)
        return self.output_head(x)
    
class ResBlock_Predictor(nn.Module):
    def __init__(self, hidden_dim, dropout_p=0.05):
        super().__init__()

        self.res = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), # LayerNorm is key for stable inverse mapping
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, x):
        # Clean skip connection with LeakyReLU to avoid dead zones
        return F.leaky_relu(self.res(x) + x, 0.1)

class ResBlockMLP_Predictor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()

        self.initial_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1)
        )

        # 2-3 blocks are ideal for finding the inverse mapping
        self.res_blocks = nn.Sequential(
            ResBlock_Predictor(hidden_dim),
            ResBlock_Predictor(hidden_dim)
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid() # Forces output into the [0, 1] MinMaxScaler range
            #nn.Tanh()
        )
    
    def forward(self, x):
        x = self.initial_layer(x)
        x = self.res_blocks(x)
        #x = torch.tanh(x / 5.0) * 0.5 + 0.5
        return self.output_head(x)