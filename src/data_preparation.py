import pandas as pd
from pathlib import Path

# =========================
# 2.2 Zeitraum filtern
# =========================

# 1) Pfade
RAW_PATH = Path("data/raw")
PROCESSED_PATH = Path("data/processed")
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

# 2) CSV-Datei finden (nimmt die erste .csv in data/raw)
csv_files = list(RAW_PATH.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(
        "Keine CSV in data/raw gefunden. Lege die Jena-CSV dort ab, z.B. data/raw/jena_climate_2009_2016.csv"
    )
DATA_PATH = csv_files[0]
print(f"Using dataset: {DATA_PATH}")

# 3) Laden
df = pd.read_csv(DATA_PATH)

# 4) Zeitspalte erkennen (im Jena-Datensatz typischerweise: 'Date Time')
time_col = "Date Time"
if time_col not in df.columns:
    raise KeyError(
        f"Zeitspalte '{time_col}' nicht gefunden. Erste Spalten: {list(df.columns)[:15]}"
    )

# 5) In datetime umwandeln (deutsches Format: Tag.Monat.Jahr Stunde:Minute:Sekunde)
# Beispiel aus deiner Fehlermeldung: "13.01.2009 00:00:00"
df[time_col] = pd.to_datetime(df[time_col], format="%d.%m.%Y %H:%M:%S")

# 6) Als Index setzen und sortieren
df = df.set_index(time_col).sort_index()

# 7) Zeitraum filtern: 2014-01-01 bis 2016-12-31
df_filtered = df.loc["2014-01-01":"2016-12-31"]

print("Filtered shape:", df_filtered.shape)
print("First timestamp:", df_filtered.index.min())
print("Last timestamp:", df_filtered.index.max())

# 8) Speichern
out_file = PROCESSED_PATH / "jena_climate_2014_2016.csv"
df_filtered.to_csv(out_file)
print(f"Saved filtered dataset to: {out_file}")

# =========================
# 2.3 Reduktion: 10 Min -> 1 Stunde (Mittelwert)
# =========================

# stündlich resamplen (Durchschnitt pro Stunde)
df_hourly = df_filtered.resample("1h").mean()
print("Columns:", df_hourly.columns.tolist())


print("Hourly shape:", df_hourly.shape)
print("Hourly first timestamp:", df_hourly.index.min())
print("Hourly last timestamp:", df_hourly.index.max())

# speichern
out_hourly = PROCESSED_PATH / "jena_climate_2014_2016_hourly.csv"
df_hourly.to_csv(out_hourly)
print(f"Saved hourly dataset to: {out_hourly}")

# =========================
# 2.4 EDA – Temperatur-Zeitreihe plotten
# =========================

import matplotlib.pyplot as plt

# Temperatur-Spalte
temp_col = "T (degC)"

# Plot erstellen
plt.figure(figsize=(12, 5))
plt.plot(df_hourly.index, df_hourly[temp_col], linewidth=0.5)
plt.title("Hourly Air Temperature (2014–2016)")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.tight_layout()

# Plot speichern
plot_path = "results/temperature_timeseries.png"
plt.savefig(plot_path, dpi=150)
plt.close()

print(f"Saved temperature time series plot to: {plot_path}")

# =========================
# 2.5 Train / Validation / Test Split
# =========================

# Anzahl der Datenpunkte
n = len(df_hourly)

# Split-Indizes
train_end = int(n * 0.70)
val_end = int(n * 0.85)

# Zeitbasierter Split (KEIN Shuffle!)
train_df = df_hourly.iloc[:train_end]
val_df = df_hourly.iloc[train_end:val_end]
test_df = df_hourly.iloc[val_end:]

# Prüfen
print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)
print("Test shape:", test_df.shape)

print("Train period:", train_df.index.min(), "to", train_df.index.max())
print("Validation period:", val_df.index.min(), "to", val_df.index.max())
print("Test period:", test_df.index.min(), "to", test_df.index.max())


# Speichern
train_df.to_csv(PROCESSED_PATH / "train.csv")
val_df.to_csv(PROCESSED_PATH / "validation.csv")
test_df.to_csv(PROCESSED_PATH / "test.csv")

print("Saved train / validation / test datasets.")
