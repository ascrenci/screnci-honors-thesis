import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import config

    
# new dataset class will min max scale magnitude to [-1, 1]
class LCDataset(Dataset):
    def __init__(self, data_file):
        data = np.load(data_file)

        mag = data[:, 0, 5:]
        mag = 20*np.log10(mag)
        phase = data[:, 1, 5:]/np.pi
        y = np.log10(data[:, 0, :5]).reshape(-1, 5)
        
        self.mag_max = mag.max()
        self.mag_min = mag.min()
        self.y_max = y.max(axis=0)
        self.y_min = y.min(axis=0)

        mag_norm = 2*(mag - self.mag_min)/(self.mag_max - self.mag_min) - 1
        y_norm = 2*(y - self.y_min)/(self.y_max - self.y_min) - 1
        
        
        #y_norm = self.normalize(y)

        #self.mag_scaler = StandardScaler().fit(mag)
        #self.phase_scaler = StandardScaler().fit(phase)
        #self.y_scaler = StandardScaler().fit(y)

        #mag = self.mag_scaler.transform(mag)
        #phase = self.phase_scaler.transform(phase)
        #y = self.y_scaler.transform(y)

        X = np.stack([mag_norm, phase], axis = 1)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y_norm, dtype=torch.float32)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    
class LCModel(nn.Module):
    def __init__(self, mag_max=None, mag_min=None, y_max=None, y_min=None, input_dim=2, output_dim=5):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),  # keep more frequency detail
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        '''
        w_np = np.logspace(np.log10(10), np.log10(10E9), config.freq_length)/10E9
        w_pt = torch.tensor(w_np, dtype=torch.float32)
        s_pt = torch.complex(torch.zeros_like(w_pt), w_pt)
        self.register_buffer('s', s_pt)

        self.register_buffer('mag_max', torch.tensor(mag_max, dtype=torch.float32))
        self.register_buffer('mag_min', torch.tensor(mag_min, dtype=torch.float32))
        self.register_buffer('y_max', torch.tensor(y_max, dtype=torch.float32))
        self.register_buffer('y_min', torch.tensor(y_min, dtype=torch.float32))
        '''

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

    '''
    def compute_mag_pred(self, y_pred):
        s = self.s.unsqueeze(0)
        comps = self.map_range(y_pred, self.y_min, self.y_max)
        #comps = (y_pred + 1)*(self.y_max - self.y_min)/2 + self.y_min

        R = torch.pow(10.0, comps[:,0].unsqueeze(1))
        L1 = torch.pow(10.0, comps[:,1].unsqueeze(1))*10E9
        L2 = torch.pow(10.0, comps[:,2].unsqueeze(1))*10E9
        C1 = torch.pow(10.0, comps[:,3].unsqueeze(1))*10E9
        C2 = torch.pow(10.0, comps[:,4].unsqueeze(1))*10E9

        H_s = (C1*s)/((C1*L1*s**2 + 1)*(1/R + (C1*s)/(C1*L1*s**2 + 1) + (C2*s)/(C2*L2*s**2 + 1)))
        mag_vv = torch.abs(H_s)
        mag_dB = 20*torch.log10(mag_vv + 1E-12)
        mag_dB_norm = 2*(mag_dB - self.mag_min)/(self.mag_max - self.mag_min) - 1

        return mag_dB_norm
    
    @staticmethod
    def map_range(y, y_min, y_max):
        x = torch.sigmoid(y)
        return y_min + (y_max - y_min)*x
    '''