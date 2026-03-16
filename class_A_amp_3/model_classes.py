import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import Dataset
import joblib
from sklearn.preprocessing import MinMaxScaler

class ClassADataset(Dataset):
    def __init__(self, data_file):
        data = np.log10(np.load(data_file))
        
        X = data[:, :4] # VDD, CL, frequency, gain
        y = data[:, 4:] # R1, RD, Vto, Kp

        self.xscaler = MinMaxScaler()
        self.yscaler = MinMaxScaler()

        X_norm = self.xscaler.fit_transform(X)
        y_norm = self.yscaler.fit_transform(y)
        joblib.dump(self.xscaler, "x_scaler.pkl")
        joblib.dump(self.yscaler, "y_scaler.pkl")

        self.X = torch.tensor(X_norm, dtype=torch.float32)
        self.y = torch.tensor(y_norm, dtype=torch.float32)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ClassAModel(nn.Module):
    def __init__(self, input_dim=4, output_dim=4):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)