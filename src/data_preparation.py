import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


# Pfade
RAW_PATH = Path("data/raw")
PROCESSED_PATH = Path("data/processed")
RESULTS_PATH = Path("results")

PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# CSV-Datei finden
csv_files = list(RAW_PATH.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(
        "Keine CSV in data/raw gefunden. Lege die Jena-CSV dort ab, z.B. data/raw/jena_climate_2009_2016.csv"
    )

DATA_PATH = csv_files[0]
print(f"Using dataset: {DATA_PATH}")

# Laden
df = pd.read_csv(DATA_PATH)

# Zeitspalte prüfen
time_col = "Date Time"
if time_col not in df.columns:
    raise KeyError(
        f"Zeitspalte '{time_col}' nicht gefunden. Erste Spalten: {list(df.columns)[:15]}"
    )

# Zeit parsen und als Index setzen
df[time_col] = pd.to_datetime(df[time_col], format="%d.%m.%Y %H:%M:%S")
df = df.set_index(time_col).sort_index()

# Zeitraum filtern: 2014-01-01 bis 2016-12-31
df_filtered = df.loc["2014-01-01":"2016-12-31"]
print("Filtered shape:", df_filtered.shape)

# Speichern (gefiltert)
filtered_path = PROCESSED_PATH / "jena_climate_2014_2016.csv"
df_filtered.to_csv(filtered_path)
print(f"Saved: {filtered_path}")

# Reduktion: 10-Min -> 1 Stunde (Mittelwerte)
df_hourly = df_filtered.resample("1H").mean().dropna()
print("Hourly shape:", df_hourly.shape)

# Speichern (stündlich)
hourly_path = PROCESSED_PATH / "jena_climate_2014_2016_hourly.csv"
df_hourly.to_csv(hourly_path)
print(f"Saved: {hourly_path}")

# EDA: Temperatur-Zeitreihe plotten
temp_col = "T (degC)"
if temp_col not in df_hourly.columns:
    raise KeyError(
        f"Temperaturspalte '{temp_col}' nicht gefunden. Verfügbare Spalten: {df_hourly.columns.tolist()}"
    )

plt.figure(figsize=(12, 5))
plt.plot(df_hourly.index, df_hourly[temp_col], linewidth=0.5)
plt.title("Hourly Air Temperature (2014–2016)")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.tight_layout()

plot_path = RESULTS_PATH / "temperature_timeseries.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"Saved plot: {plot_path}")

# Split: Train / Validation / Test (chronologisch)
n = len(df_hourly)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = df_hourly.iloc[:train_end]
val_df = df_hourly.iloc[train_end:val_end]
test_df = df_hourly.iloc[val_end:]

print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)
print("Test shape:", test_df.shape)

print("Train period:", train_df.index.min(), "to", train_df.index.max())
print("Validation period:", val_df.index.min(), "to", val_df.index.max())
print("Test period:", test_df.index.min(), "to", test_df.index.max())

# Speichern (Split)
train_df.to_csv(PROCESSED_PATH / "train.csv")
val_df.to_csv(PROCESSED_PATH / "validation.csv")
test_df.to_csv(PROCESSED_PATH / "test.csv")
print("Saved train/validation/test CSVs.")
