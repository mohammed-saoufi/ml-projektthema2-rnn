import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import JenaDataset
from src.model import LSTMModel

# -------------------------
# Konfiguration
# -------------------------
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 10
SEQ_LENGTH = 24

TRAIN_CSV = Path("data/processed/train.csv")
VAL_CSV = Path("data/processed/validation.csv")
MODEL_PATH = Path("results/lstm_model.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    # Ordner sicherstellen
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"Train CSV nicht gefunden: {TRAIN_CSV}")
    if not VAL_CSV.exists():
        raise FileNotFoundError(f"Validation CSV nicht gefunden: {VAL_CSV}")

    # --- Datasets / Loader ---
    train_ds = JenaDataset(str(TRAIN_CSV), SEQ_LENGTH, is_train=True)
    val_ds = JenaDataset(str(VAL_CSV), SEQ_LENGTH, scaler=train_ds.scaler, is_train=False)

    # Zeitreihe: shuffle=False ist die sichere Wahl
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # --- Modell / Loss / Optimizer ---
    model = LSTMModel(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0

        for x, y in train_loader:
            x = x.to(DEVICE)  # (B, T, 1)
            y = y.to(DEVICE)  # (B, 1)

            optimizer.zero_grad(set_to_none=True)
            preds = model(x)  # (B, 1)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / max(1, len(train_loader))

        # --- Validation ---
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v = x_v.to(DEVICE)
                y_v = y_v.to(DEVICE)
                preds_v = model(x_v)
                val_loss_sum += criterion(preds_v, y_v).item()

        avg_val_loss = val_loss_sum / max(1, len(val_loader))

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss = {avg_train_loss:.6f}, "
            f"Val Loss = {avg_val_loss:.6f}"
        )

    # --- Speichern: NUR weights + config (kein Scaler!) ---
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "seq_length": SEQ_LENGTH,
            },
        },
        MODEL_PATH,
    )

    print(f"Training beendet. Modell gespeichert: {MODEL_PATH}")


if __name__ == "__main__":
    print("RUNNING FILE:", __file__)
    print("CWD:", os.getcwd())
    train()
