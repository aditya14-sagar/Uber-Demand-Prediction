import joblib
import pandas as pd
import logging
from pathlib import Path
from yaml import safe_load, YAMLError
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


# create a logger
logger = logging.getLogger("extract_features")
logger.setLevel(logging.INFO)

# attach a console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

# make a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)


def read_cluster_input(data_path, chunksize=100000, usecols=["pickup_latitude", "pickup_longitude"]):
    """Read CSV in chunks for clustering. Returns a TextFileReader iterator."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if data_path.stat().st_size == 0:
        raise ValueError(f"Data file is empty: {data_path}")
    try:
        df_reader = pd.read_csv(data_path, chunksize=chunksize, usecols=["pickup_latitude", "pickup_longitude"])
    except ValueError as e:
        raise ValueError(f"One or more expected columns {usecols} are missing from {data_path}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to read data file {data_path}: {e}") from e
    return df_reader


def save_model(model, save_path):
    """Persist a model to disk with joblib."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(model, save_path)
    except Exception as e:
        raise RuntimeError(f"Failed to save model to {save_path}: {e}") from e


def read_params(params_path="params.yaml"):
    """Load and return the YAML parameter file."""
    params_path = Path(params_path)
    if not params_path.exists():
        raise FileNotFoundError(f"Params file not found: {params_path}")
    try:
        with open(params_path, "r") as file:
            params = safe_load(file)
    except YAMLError as e:
        raise ValueError(f"Could not parse YAML in {params_path}: {e}") from e
    except OSError as e:
        raise RuntimeError(f"Could not open params file {params_path}: {e}") from e

    if not isinstance(params, dict):
        raise ValueError(f"Expected a mapping at the top level of {params_path}, got {type(params).__name__}")
    return params


def get_nested_param(params, *keys):
    """Safely retrieve a nested value from a dict; raises KeyError with a clear path if missing."""
    node = params
    path = []
    for key in keys:
        path.append(key)
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"Missing required parameter: {' -> '.join(path)}")
        node = node[key]
    return node


if __name__ == "__main__":
    # current path
    current_path = Path(__file__)
    # set the root path
    root_path = current_path.parent.parent.parent
    # data_path
    data_path = root_path / "data/interim/df_without_outliers.csv"


    # 1. Fit StandardScaler                                                

    logger.info("Fitting StandardScaler …")
    scaler = StandardScaler()
    chunks_seen = 0
    try:
        df_reader = read_cluster_input(data_path)
        for chunk in df_reader:
            if chunk.empty:
                logger.warning("Encountered an empty chunk while fitting scaler — skipping.")
                continue
            if chunk.isnull().values.any():
                logger.warning("Chunk contains NaN values; dropping rows before fitting scaler.")
                chunk = chunk.dropna()
            if chunk.empty:
                logger.warning("Chunk became empty after dropping NaN rows — skipping.")
                continue
            scaler.partial_fit(chunk)
            chunks_seen += 1
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error("Failed to read data for scaler fitting: %s", e)
        raise SystemExit(1)
    except Exception as e:
        logger.error("Unexpected error while fitting scaler: %s", e)
        raise SystemExit(1)

    if chunks_seen == 0:
        logger.error("No valid data chunks were found — cannot fit scaler.")
        raise SystemExit(1)

    scaler_save_path = root_path / "models/scaler.joblib"
    try:
        save_model(scaler, scaler_save_path)
    except RuntimeError as e:
        logger.error("Could not save scaler: %s", e)
        raise SystemExit(1)
    logger.info("Scaler saved to %s", scaler_save_path)

    
    # 2. Read parameters                                                   
   
    try:
        params = read_params()
        mini_batch_params = get_nested_param(params, "extract_features", "mini_batch_kmeans")
        ewma_params = get_nested_param(params, "extract_features", "ewma")
    except (FileNotFoundError, ValueError, KeyError) as e:
        logger.error("Parameter loading failed: %s", e)
        raise SystemExit(1)

    logger.info("MiniBatchKMeans parameters: %s", mini_batch_params)
    logger.info("EWMA parameters: %s", ewma_params)

  
    # 3. Train MiniBatchKMeans                                             

    logger.info("Training MiniBatchKMeans …")
    try:
        mini_batch = MiniBatchKMeans(**mini_batch_params)
    except TypeError as e:
        logger.error("Invalid MiniBatchKMeans parameter(s): %s", e)
        raise SystemExit(1)

    chunks_seen = 0
    try:
        df_reader = read_cluster_input(data_path)
        for chunk in df_reader:
            if chunk.empty:
                logger.warning("Empty chunk encountered during KMeans training — skipping.")
                continue
            if chunk.isnull().values.any():
                chunk = chunk.dropna()
            if chunk.empty:
                continue
            try:
                scaled_chunk = scaler.transform(chunk)
            except Exception as e:
                logger.error("Scaler transform failed on chunk: %s", e)
                raise SystemExit(1)
            mini_batch.partial_fit(scaled_chunk)
            chunks_seen += 1
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error("Failed to read data for KMeans training: %s", e)
        raise SystemExit(1)
    except SystemExit:
        raise
    except Exception as e:
        logger.error("Unexpected error during KMeans training: %s", e)
        raise SystemExit(1)

    if chunks_seen == 0:
        logger.error("No valid data chunks were found — cannot train KMeans model.")
        raise SystemExit(1)

    kmeans_save_path = root_path / "models/mb_kmeans.joblib"
    try:
        save_model(mini_batch, kmeans_save_path)
    except RuntimeError as e:
        logger.error("Could not save KMeans model: %s", e)
        raise SystemExit(1)
    logger.info("KMeans model saved to %s", kmeans_save_path)

 
    # 4. Assign cluster labels & resample to 15-minute intervals          

    logger.info("Loading full dataset for cluster predictions …")
    try:
        df_final = pd.read_csv(data_path, parse_dates=["tpep_pickup_datetime"])
    except FileNotFoundError as e:
        logger.error("Data file not found: %s", e)
        raise SystemExit(1)
    except ValueError as e:
        logger.error(
            "Could not parse 'tpep_pickup_datetime' as dates or a required column is missing: %s", e
        )
        raise SystemExit(1)
    except Exception as e:
        logger.error("Unexpected error reading full dataset: %s", e)
        raise SystemExit(1)

    if df_final.empty:
        logger.error("Full dataset is empty — nothing to process.")
        raise SystemExit(1)

    required_cols = {"pickup_longitude", "pickup_latitude", "tpep_pickup_datetime"}
    missing = required_cols - set(df_final.columns)
    if missing:
        logger.error("Required column(s) missing from dataset: %s", missing)
        raise SystemExit(1)

    location_subset = df_final.loc[:, ["pickup_longitude", "pickup_latitude"]]
    if location_subset.isnull().values.any():
        n_before = len(location_subset)
        location_subset = location_subset.dropna()
        df_final = df_final.loc[location_subset.index]
        logger.warning(
            "Dropped %d rows with NaN coordinates (%d remaining).",
            n_before - len(location_subset),
            len(location_subset),
        )

    try:
        scaled_location_subset = scaler.transform(location_subset)
        cluster_predictions = mini_batch.predict(scaled_location_subset)
    except Exception as e:
        logger.error("Cluster prediction step failed: %s", e)
        raise SystemExit(1)

    df_final = df_final.copy()
    df_final["region"] = cluster_predictions
    logger.info("Cluster labels added to dataset.")

    df_final = df_final.drop(columns=["pickup_latitude", "pickup_longitude"])
    logger.info("Dropped latitude and longitude columns.")

    # Validate datetime index before resampling
    if df_final["tpep_pickup_datetime"].isnull().any():
        n_null = df_final["tpep_pickup_datetime"].isnull().sum()
        logger.warning("Dropping %d rows with null pickup datetime.", n_null)
        df_final = df_final.dropna(subset=["tpep_pickup_datetime"])

    df_final.set_index("tpep_pickup_datetime", inplace=True)
    region_grp = df_final.groupby("region")

    try:
        resampled_data = (
            region_grp["region"]
            .resample("15min")
            .count()
        )
    except Exception as e:
        logger.error("Resampling to 15-minute intervals failed: %s", e)
        raise SystemExit(1)

    resampled_data.name = "total_pickups"
    resampled_data = resampled_data.reset_index(level=0)

    epsilon_val = 10
    resampled_data.replace({"total_pickups": {0: epsilon_val}}, inplace=True)
    logger.info("Resampling complete; zeros replaced with epsilon=%d.", epsilon_val)


    # 5. Compute EWMA-based average pickups                                

    try:
        resampled_data["avg_pickups"] = (
            resampled_data
            .groupby("region")["total_pickups"]
            .ewm(**ewma_params)
            .mean()
            .shift(1)
            .round()
            .values
        )
    except TypeError as e:
        logger.error("Invalid EWMA parameter(s): %s", e)
        raise SystemExit(1)
    except Exception as e:
        logger.error("EWMA computation failed: %s", e)
        raise SystemExit(1)

    if resampled_data["avg_pickups"].isnull().all():
        logger.warning(
            "All avg_pickups values are NaN — check EWMA parameters or whether "
            "there are enough data points per region."
        )
    logger.info("EWMA-based average pickups calculated successfully.")

    # 6. Save output                                                       

    save_path = root_path / "data/processed/resampled_data.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        resampled_data.to_csv(save_path, index=True)
    except OSError as e:
        logger.error("Failed to write output CSV to %s: %s", save_path, e)
        raise SystemExit(1)
    logger.info("Output saved to %s", save_path)


