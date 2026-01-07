import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from src.dataset import JenaDataset
from src.model import LSTMModel

# Konfiguration
MODEL_PATH = Path("results/lstm_model_tuning.pth")
TEST_CSV = Path("data/processed/test.csv")
TRAIN_CSV = Path("data/processed/train.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize():
    # --- 1. LOSS-KURVE PLOTTEN (Aufgabe 6.1) ---
    # Diese Werte stammen direkt aus deinem Terminal-Screenshot
    train_losses = [0.2999, 0.0518, 0.0266, 0.0199, 0.0175, 0.0168, 0.0159, 0.0151, 0.0152, 0.0140, 0.0142, 0.0136, 0.0131, 0.0131, 0.0129]
    val_losses = [0.0843, 0.0235, 0.0163, 0.0121, 0.0109, 0.0113, 0.0134, 0.0157, 0.0162, 0.0137, 0.0144, 0.0158, 0.0164, 0.0163, 0.0139]

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title('Trainings- und Validierungsverlust (Tuning)')
    plt.xlabel('Epoche')
    plt.ylabel('MSE (skaliert)')
    plt.legend()
    plt.grid(True)
    plt.savefig("results/loss_tuning.png")
    print("Loss-Kurve gespeichert: results/loss_tuning.png")

    # --- 2. VORHERSAGE VS. REALITÄT (Aufgabe 6.2) ---
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    cfg = checkpoint["config"]
    
    # Scaler wiederherstellen (wichtig für °C)
    train_df = pd.read_csv(TRAIN_CSV)
    scaler = StandardScaler().fit(train_df["T (degC)"].values.reshape(-1, 1))
    
    test_ds = JenaDataset(str(TEST_CSV), cfg["seq_length"], scaler=scaler)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    
    model = LSTMModel(1, cfg["hidden_size"], cfg["num_layers"], cfg["dropout"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds = []
    all_actuals = []

    with torch.no_grad():
        for x, y in test_loader:
            preds = model(x.to(DEVICE))
            all_preds.append(preds.cpu())
            all_actuals.append(y)

    # Zurück in Grad Celsius wandeln
    preds_c = scaler.inverse_transform(torch.cat(all_preds).numpy())
    actuals_c = scaler.inverse_transform(torch.cat(all_actuals).numpy())

    # Plot erstellen (die ersten 200 Stunden für bessere Sichtbarkeit)
    plt.figure(figsize=(12, 6))
    plt.plot(actuals_c[:200], label='Tatsächlich (°C)', color='black', alpha=0.6)
    plt.plot(preds_c[:200], label='Vorhersage Tuning (°C)', color='red', linestyle='--')
    plt.title('Temperaturvorhersage: Vergleich Tuning-Modell vs. Realität')
    plt.xlabel('Zeit (Stunden)')
    plt.ylabel('Temperatur (°C)')
    plt.legend()
    plt.savefig("results/prediction_comparison_tuning.png")
    print("Vergleichs-Plot gespeichert: results/prediction_comparison_tuning.png")

if __name__ == "__main__":
    visualize()