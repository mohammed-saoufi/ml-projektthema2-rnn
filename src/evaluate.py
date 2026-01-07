import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from src.dataset import JenaDataset
from src.model import LSTMModel

# -------------------------
# Konfiguration
# -------------------------
BATCH_SIZE = 64
TRAIN_CSV = Path("data/processed/train.csv")
TEST_CSV = Path("data/processed/test.csv")
MODEL_PATH = Path("results/lstm_model.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}. Bitte zuerst trainieren.")
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"Train CSV nicht gefunden: {TRAIN_CSV}")
    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Test CSV nicht gefunden: {TEST_CSV}")

    # --- Checkpoint laden (nur weights + config) ---
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    cfg = checkpoint.get("config", {})
    hidden_size = cfg.get("hidden_size", 64)
    num_layers = cfg.get("num_layers", 2)
    dropout = cfg.get("dropout", 0.2)
    seq_length = cfg.get("seq_length", 24)

    # --- Scaler NUR auf Train fitten (kein Leakage) ---
    train_df = pd.read_csv(TRAIN_CSV)
    scaler = StandardScaler().fit(train_df["T (degC)"].values.reshape(-1, 1))

    # --- Test Dataset / Loader ---
    test_ds = JenaDataset(str(TEST_CSV), seq_length, scaler=scaler, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # --- Modell instanziieren und weights laden ---
    model = LSTMModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # --- Metriken ---
    mse_criterion = nn.MSELoss(reduction="mean")

    mse_scaled_sum = 0.0
    mse_c_sum = 0.0
    n_batches = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            preds = model(x)

            # 1) MSE im skalierten Raum (StandardScaler)
            mse_scaled_sum += mse_criterion(preds, y).item()

            # 2) MSE in °C (zurücktransformiert)
            preds_np = preds.cpu().numpy()
            y_np = y.cpu().numpy()

            preds_c = scaler.inverse_transform(preds_np)
            y_c = scaler.inverse_transform(y_np)

            mse_c_sum += float(np.mean((preds_c - y_c) ** 2))
            n_batches += 1

    avg_mse_scaled = mse_scaled_sum / max(1, n_batches)
    avg_mse_c = mse_c_sum / max(1, n_batches)

    print("\n--- Evaluierung abgeschlossen ---")
    print(f"MSE (skaliert, StandardScaler-Raum): {avg_mse_scaled:.6f}")
    print(f"MSE (in °C, zurücktransformiert):     {avg_mse_c:.6f}")


if __name__ == "__main__":
    print("RUNNING FILE:", __file__)
    print("CWD:", os.getcwd())
    evaluate()
