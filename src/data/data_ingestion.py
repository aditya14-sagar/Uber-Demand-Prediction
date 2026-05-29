import dask.dataframe as dd
import logging
from pathlib import Path
import sys

# create a logger
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.INFO)
# attach a console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)
# make a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# inlier range for latitude and longitude
min_latitude = 40.60
max_latitude = 40.85
min_longitude = -74.05
max_longitude = -73.70
# inlier range for fare amount and trip distance
min_fare_amount_val = 0.50
max_fare_amount_val = 81.0
min_trip_distance_val = 0.25
max_trip_distance_val = 24.43


def read_dask_df(
    data_path: Path,
    parse_dates: list = ["tpep_pickup_datetime"],
    columns: list = [
        "trip_distance",
        "tpep_pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "fare_amount",
    ],
):
    if not isinstance(data_path, Path):
        data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if not data_path.is_file():
        raise ValueError(f"Path is not a file: {data_path}")

    if data_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file, got: {data_path.suffix}")

    try:
        dd_df = dd.read_csv(data_path, parse_dates=parse_dates, usecols=columns)
        logger.info(f"Successfully read file: {data_path.name}")
        return dd_df
    except ValueError as e:
        raise ValueError(
            f"Column mismatch or parse error reading '{data_path.name}': {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error reading '{data_path.name}': {e}") from e


def dask_pipeline(df):
    if df is None:
        raise ValueError("Input DataFrame is None.")

    required_cols = {
        "pickup_latitude", "pickup_longitude",
        "dropoff_latitude", "dropoff_longitude",
        "fare_amount", "trip_distance",
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"DataFrame is missing required columns: {missing_cols}")

    try:
        # remove outliers from lat/long columns
        df = df.loc[
            (df["pickup_latitude"].between(min_latitude, max_latitude, inclusive="both"))
            & (df["pickup_longitude"].between(min_longitude, max_longitude, inclusive="both"))
            & (df["dropoff_latitude"].between(min_latitude, max_latitude, inclusive="both"))
            & (df["dropoff_longitude"].between(min_longitude, max_longitude, inclusive="both")),
            :,
        ]

        # remove outliers from fare amount and trip distance columns
        df = df.loc[
            (df["fare_amount"].between(min_fare_amount_val, max_fare_amount_val, inclusive="both"))
            & (df["trip_distance"].between(min_trip_distance_val, max_trip_distance_val, inclusive="both"))
        ]
        logger.info("Outliers removed successfully.")

    except KeyError as e:
        raise KeyError(f"Column not found during outlier filtering: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error during outlier removal: {e}") from e

    try:
        cols_to_drop = ["trip_distance", "dropoff_longitude", "dropoff_latitude", "fare_amount"]
        missing_drop_cols = [c for c in cols_to_drop if c not in df.columns]
        if missing_drop_cols:
            raise ValueError(f"Cannot drop missing columns: {missing_drop_cols}")

        df = df.drop(cols_to_drop, axis=1)
        logger.info("Columns dropped successfully.")
    except Exception as e:
        raise RuntimeError(f"Error while dropping columns: {e}") from e

    try:
        df = df.compute()
        if df.empty:
            logger.warning("Computed DataFrame is empty — all rows may have been filtered out.")
        else:
            logger.info(f"Dask DataFrame computed successfully. Shape: {df.shape}")
        return df
    except MemoryError:
        raise MemoryError("Insufficient memory to compute the Dask DataFrame.")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during DataFrame computation: {e}") from e


if __name__ == "__main__":
    # current path
    current_path = Path(__file__)
    # set the root path
    root_path = current_path.parent.parent.parent
    # raw data path
    raw_data_dir = root_path / "data/raw"

    if not raw_data_dir.exists():
        logger.error(f"Raw data directory does not exist: {raw_data_dir}")
        sys.exit(1)

    # dataframe names
    df_names = [
        "yellow_tripdata_2016-01.csv",
        "yellow_tripdata_2016-02.csv",
        "yellow_tripdata_2016-03.csv",
    ]

    # read all dataframes
    dfs = []
    for df_name in df_names:
        df_path = raw_data_dir / df_name
        try:
            df = read_dask_df(df_path)
            dfs.append(df)
        except FileNotFoundError as e:
            logger.error(f"File not found, skipping: {e}")
        except (ValueError, RuntimeError) as e:
            logger.error(f"Failed to read '{df_name}', skipping: {e}")

    if not dfs:
        logger.error("No DataFrames were loaded. Exiting.")
        sys.exit(1)

    logger.info(f"Successfully read {len(dfs)}/{len(df_names)} DataFrames.")

    try:
        df_final = dd.concat(dfs, axis=0)
        logger.info("All datasets merged successfully.")
    except ValueError as e:
        logger.error(f"Failed to concatenate DataFrames (schema mismatch?): {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during concatenation: {e}")
        sys.exit(1)

    try:
        df_final = dask_pipeline(df_final)
        logger.info("Dask pipeline executed successfully.")
    except (ValueError, KeyError) as e:
        logger.error(f"Pipeline failed due to data/schema issue: {e}")
        sys.exit(1)
    except MemoryError as e:
        logger.error(f"Pipeline failed — out of memory: {e}")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"Pipeline failed with runtime error: {e}")
        sys.exit(1)

    # save the dataframe
    df_without_outliers_path = root_path / "data/interim/df_without_outliers.csv"
    try:
        df_without_outliers_path.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(df_without_outliers_path, index=False)
        logger.info(f"DataFrame saved successfully to: {df_without_outliers_path}")
    except PermissionError:
        logger.error(f"Permission denied when saving to: {df_without_outliers_path}")
        sys.exit(1)
    except OSError as e:
        logger.error(f"OS error while saving DataFrame: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error while saving DataFrame: {e}")
        sys.exit(1)