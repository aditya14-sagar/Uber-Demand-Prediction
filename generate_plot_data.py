"""
Regenerate data/external/plot_data.csv

This file is used only for the Streamlit map visualization (app.py).
It needs: pickup_latitude, pickup_longitude, region

We build it by:
1. Loading data/interim/df_without_outliers.csv (same source used to fit scaler/kmeans)
2. Scaling the coordinates with the existing models/scaler.joblib
3. Predicting region labels with models/mb_kmeans.joblib
4. Sampling a manageable number of points per region (for fast map rendering)
5. Saving to data/external/plot_data.csv
"""

import joblib
import pandas as pd
from pathlib import Path

# adjust this if running from a different location
root_path = Path(__file__).parent

data_path = root_path / "data/interim/df_without_outliers.csv"
scaler_path = root_path / "models/scaler.joblib"
kmeans_path = root_path / "models/mb_kmeans.joblib"
save_path = root_path / "data/external/plot_data.csv"

# how many points per region to keep for the map (avoids a huge/slow file)
SAMPLE_PER_REGION = 200

print(f"Loading {data_path} ...")
df = pd.read_csv(data_path, usecols=["pickup_latitude", "pickup_longitude"])
df = df.dropna()

print("Loading scaler and kmeans model ...")
scaler = joblib.load(scaler_path)
kmeans = joblib.load(kmeans_path)

print("Scaling coordinates and predicting regions ...")
scaled = scaler.transform(df[scaler.feature_names_in_])
df["region"] = kmeans.predict(scaled)

print(f"Sampling up to {SAMPLE_PER_REGION} points per region ...")
df_sampled = (
    df.groupby("region", group_keys=True)
    .apply(lambda g: g.sample(n=min(len(g), SAMPLE_PER_REGION), random_state=42), include_groups=False)
    .reset_index(level=0)  
    .reset_index(drop=True)
)

save_path.parent.mkdir(parents=True, exist_ok=True)
df_sampled.to_csv(save_path, index=False)

print(f"Saved {len(df_sampled)} rows to {save_path}")
print(df_sampled["region"].value_counts().sort_index())
