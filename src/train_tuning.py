import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

# Importe aus deinem Projekt-Ordner
from src.dataset import JenaDataset
from src.model import LSTMModel

# -------------------------
# Aufgabe 5: MEIN TUNING
# -------------------------
HIDDEN_SIZE = 128      # Mehr Kapazität
NUM_LAYERS = 3         # Tieferes Modell
DROPOUT = 0.3          # Schutz vor Overfitting
BATCH_SIZE = 32        # Kleinere Batches für bessere Generalisierung
LEARNING_RATE = 5e-4   # Langsameres Lernen
EPOCHS = 15            # Längeres Training
SEQ_LENGTH = 48        # Größerer Zeit-Kontext

# Pfade definieren
TRAIN_CSV = Path("data/processed/train.csv")
VAL_CSV = Path("data/processed/validation.csv")
MODEL_PATH = Path("results/lstm_model_tuning.pth") # Separates File

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # Ordner für Ergebnisse sicherstellen
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not TRAIN_CSV.exists() or not VAL_CSV.exists():
        print("Fehler: Daten nicht gefunden. Bitte data_preparation.py zuerst ausführen.")
        return

    # 1. Datasets & Loader vorbereiten
    train_ds = JenaDataset(str(TRAIN_CSV), SEQ_LENGTH, is_train=True)
    val_ds = JenaDataset(str(VAL_CSV), SEQ_LENGTH, scaler=train_ds.scaler, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Modell, Loss und Optimizer initialisieren
    model = LSTMModel(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"--- Starte Tuning-Training auf {DEVICE} ---")
    
    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validierung
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v, y_v = x_v.to(DEVICE), y_v.to(DEVICE)
                val_loss += criterion(model(x_v), y_v).item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.6f} | Val Loss: {val_loss/len(val_loader):.6f}")

    # 4. Speichern des optimierten Modells
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "seq_length": SEQ_LENGTH
        }
    }, MODEL_PATH)
    print(f"--- Tuning beendet! Modell gespeichert unter: {MODEL_PATH} ---")

if __name__ == "__main__":
    train()