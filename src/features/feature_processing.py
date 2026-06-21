import logging
import sys
from pathlib import Path
import pandas as pd


# create a logger
logger = logging.getLogger("feature_processing")
logger.setLevel(logging.INFO)

# attach a console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

# make a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)


def load_data(data_path: Path) -> pd.DataFrame:
    """Read the resampled CSV, parsing the pickup datetime column."""
    try:
        df = pd.read_csv(data_path, parse_dates=["tpep_pickup_datetime"])
    except FileNotFoundError:
        logger.error("Data file not found at %s", data_path)
        raise
    except ValueError as e:
        
        logger.error("Failed to parse data file (check column names / format): %s", e)
        raise
    except pd.errors.EmptyDataError:
        logger.error("Data file at %s is empty", data_path)
        raise
    except pd.errors.ParserError as e:
        logger.error("CSV parsing error in %s: %s", data_path, e)
        raise

    if df.empty:
        logger.error("Loaded dataframe is empty: %s", data_path)
        raise ValueError(f"No rows found in {data_path}")

    required_cols = {"tpep_pickup_datetime", "region", "total_pickups"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.error("Missing required columns: %s", missing)
        raise KeyError(f"Required columns missing from input data: {missing}")

    logger.info("Data read successfully (%d rows, %d columns)", *df.shape)
    return df


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day_of_week and month columns derived from the pickup datetime."""
    try:
        df["day_of_week"] = df["tpep_pickup_datetime"].dt.day_of_week
        df["month"] = df["tpep_pickup_datetime"].dt.month
    except AttributeError as e:
        
        logger.error("tpep_pickup_datetime is not datetime-typed: %s", e)
        raise

    logger.info("Datetime features extracted successfully")
    return df


def generate_lag_features(df: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    """Group by region and generate lag features for total_pickups."""
    try:
        region_grp = df.groupby("region")
        lag_features = region_grp["total_pickups"].shift(periods)
    except KeyError as e:
        logger.error("Column required for grouping/lagging not found: %s", e)
        raise

    
    if isinstance(lag_features, pd.Series):
        lag_features = lag_features.to_frame()

    if lag_features.shape[1] != len(periods):
        logger.error(
            "Expected %d lag columns, got %d - pandas version mismatch?",
            len(periods), lag_features.shape[1]
        )
        raise ValueError("Unexpected shape returned from shift(); check pandas version")

    logger.info("Lag features generated successfully")
    return lag_features


def merge_and_clean(df: pd.DataFrame, lag_features: pd.DataFrame) -> pd.DataFrame:
    """Merge lag features with original df, rename lag columns, drop NaNs."""
    try:
        data = pd.concat([lag_features, df], axis=1)
    except Exception as e:
        logger.error("Failed to merge lag features with original data: %s", e)
        raise
    logger.info("Lagged features merged successfully")

    rows_before = len(data)
    data.dropna(inplace=True)
    rows_after = len(data)

    if rows_after == 0:
        logger.error("All rows dropped after dropna() - check lag/merge logic")
        raise ValueError("No rows remain after dropping missing values")

    if rows_before > 0:
        dropped_pct = (rows_before - rows_after) / rows_before * 100
        logger.info(
            "Dropped %d/%d rows (%.1f%%) with missing values",
            rows_before - rows_after, rows_before, dropped_pct
        )

    n_lag_cols = lag_features.shape[1]
    mapper = {name: f"lag_{ind + 1}" for ind, name in enumerate(data.columns[:n_lag_cols])}
    data = data.rename(columns=mapper)
    logger.info("Column names renamed successfully")
    return data


def split_train_test(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into Jan/Feb train and March test sets, slicing the lag/day_of_week columns."""
    try:
        trainset = data.loc[data["month"].isin([1, 2]), "lag_1":"day_of_week"]
        testset = data.loc[data["month"].isin([3]), "lag_1":"day_of_week"]
    except KeyError as e:
        logger.error("Expected column not found while slicing train/test sets: %s", e)
        raise

    if trainset.empty:
        logger.error("Train set is empty after filtering for months [1, 2]")
        raise ValueError("Train set is empty - check 'month' values in source data")
    if testset.empty:
        logger.error("Test set is empty after filtering for month [3]")
        raise ValueError("Test set is empty - check 'month' values in source data")

    logger.info("Train/test split complete (train=%d rows, test=%d rows)",
                len(trainset), len(testset))
    return trainset, testset


def save_csv(df: pd.DataFrame, path: Path, label: str) -> None:
    """Save a dataframe to CSV, ensuring the parent directory exists."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=True)
    except PermissionError:
        logger.error("Permission denied writing %s to %s", label, path)
        raise
    except OSError as e:
        logger.error("OS error writing %s to %s: %s", label, path, e)
        raise

    logger.info("%s data saved successfully to %s", label, path)


def main() -> None:
    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent
    data_path = root_path / "data/processed/resampled_data.csv"

    df = load_data(data_path)
    df = add_datetime_features(df)

    df.set_index("tpep_pickup_datetime", inplace=True)
    logger.info("Datetime column set as index successfully")

    periods = list(range(1, 5))
    lag_features = generate_lag_features(df, periods)
    data = merge_and_clean(df, lag_features)
    trainset, testset = split_train_test(data)

    train_data_save_path = root_path / "data/processed/train.csv"
    test_data_save_path = root_path / "data/processed/test.csv"

    save_csv(trainset, train_data_save_path, "Train")
    save_csv(testset, test_data_save_path, "Test")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Feature processing pipeline failed")
        sys.exit(1)