import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import Dataset
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from config import branch, n_datapoints

class ClassADataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# data = [wC, maxVo, gain, Vin, THD, R1, RD, VDD]
def build_datasets(data_file, train_frac=0.7, seed=42):
    data = np.log10(np.load(data_file))
    # Data as is
    #X = data[:, :6]   # X = [c, f, maxVo, gain, Vin, THD]
    #y = data[:, 6:]   # y = [R1, RD, VDD]
    
    # Data with wC instead of CL and f
    X = data[:,:5] # X = [wC, maxVo, gain, Vin, THD] 5
    y = data[:,5:] # y = [R1, RD, VDD] 3

    # Data with wC and no THD
    #X = data[:,:4]
    #y = data[:,5:]

    # Split raw arrays first
    n = len(X)
    n_train = int(train_frac * n)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    np.save(f"./{branch}/xval_{n_datapoints}.npy", X_val)

    # Fit scalers ONLY on training data
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)

    # Transform both splits with the train-fit scalers
    X_train_s = x_scaler.transform(X_train)
    X_val_s   = x_scaler.transform(X_val)
    y_train_s = y_scaler.transform(y_train)
    y_val_s   = y_scaler.transform(y_val)

    joblib.dump(x_scaler, f"./{branch}/x_scaler_{n_datapoints}.pkl")
    joblib.dump(y_scaler, f"./{branch}/y_scaler_{n_datapoints}.pkl")

    train_dataset = ClassADataset(X_train_s, y_train_s)
    val_dataset   = ClassADataset(X_val_s,   y_val_s)
    return train_dataset, val_dataset

class ClassAModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            
            #nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),

            #nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

# data = [CL, f, maxVo, gain, Vin, THD, R1, RD, VDD]
def build_forward_datasets(data_file, train_frac=0.7, seed=42):
    data = np.log10(np.load(data_file))
    X_col = [0, 1, 4, 6, 7, 8] # 6 | [CL, f, Vin, R1, RD, VDD]
    y_col = [2, 3, 5] # 3 | [maxVo, gain, THD]
    X = data[:, X_col]
    y = data[:, y_col]

    # Split raw arrays first
    n = len(X)
    n_train = int(train_frac * n)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    np.save(f"./{branch}/xval_forward_{n_datapoints}.npy", X_val)

    # Fit scalers ONLY on training data
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)

    # Transform both splits with the train-fit scalers
    X_train_s = x_scaler.transform(X_train)
    X_val_s   = x_scaler.transform(X_val)
    y_train_s = y_scaler.transform(y_train)
    y_val_s   = y_scaler.transform(y_val)

    joblib.dump(x_scaler, f"./{branch}/x_scaler_forward_{n_datapoints}.pkl")
    joblib.dump(y_scaler, f"./{branch}/y_scaler_forward_{n_datapoints}.pkl")

    train_dataset = ClassADataset(X_train_s, y_train_s)
    val_dataset   = ClassADataset(X_val_s,   y_val_s)
    return train_dataset, val_dataset