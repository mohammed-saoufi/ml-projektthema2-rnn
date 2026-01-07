import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler


class JenaDataset(Dataset):
    def __init__(self, csv_path, seq_length, scaler=None, is_train=False):
        df = pd.read_csv(csv_path)
        data = df["T (degC)"].values.reshape(-1, 1)

        if is_train:
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(data)
        else:
            if scaler is None:
                raise ValueError("Scaler muss für val/test übergeben werden.")
            self.scaler = scaler
            self.data = self.scaler.transform(data)

        self.data = torch.FloatTensor(self.data)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_length]
        y = self.data[idx + self.seq_length]
        return x, y
