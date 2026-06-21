import sys
import pandas as pd
import joblib
import logging
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn import set_config
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


set_config(transform_output="pandas")

# create a logger
logger = logging.getLogger("train_model")
logger.setLevel(logging.INFO)

# attach a console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

# make a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)


def load_data(data_path: Path) -> pd.DataFrame:
    """Read the training CSV, parsing the pickup datetime column."""
    try:
        df = pd.read_csv(data_path, parse_dates=["tpep_pickup_datetime"])
    except FileNotFoundError:
        logger.error("Training data not found at %s", data_path)
        raise
    except ValueError as e:
        logger.error("Failed to parse training data (check columns/format): %s", e)
        raise
    except pd.errors.EmptyDataError:
        logger.error("Training data file is empty: %s", data_path)
        raise
    except pd.errors.ParserError as e:
        logger.error("CSV parsing error in %s: %s", data_path, e)
        raise

    if df.empty:
        logger.error("Loaded training dataframe has zero rows: %s", data_path)
        raise ValueError(f"No rows found in {data_path}")

    logger.info("Data read successfully (%d rows, %d columns)", *df.shape)
    return df


def make_xy(df: pd.DataFrame, target_col: str = "total_pickups") -> tuple[pd.DataFrame, pd.Series]:
    """Split the dataframe into features and target."""
    if target_col not in df.columns:
        logger.error("Target column '%s' not found in training data", target_col)
        raise KeyError(f"'{target_col}' missing from training data")

    X_train = df.drop(columns=[target_col])
    y_train = df[target_col]

    if y_train.isna().any():
        n_na = int(y_train.isna().sum())
        logger.error("Target column '%s' has %d missing values", target_col, n_na)
        raise ValueError(f"Target column contains {n_na} missing values")

    return X_train, y_train


def build_encoder(categorical_cols: list[str]) -> ColumnTransformer:
    """Construct the OneHotEncoder + passthrough ColumnTransformer."""
    return ColumnTransformer(
        [("ohe", OneHotEncoder(drop="first", sparse_output=False), categorical_cols)],
        remainder="passthrough",
        n_jobs=-1,
        force_int_remainder_cols=False,
    )


def fit_transform_encoder(
    encoder: ColumnTransformer, X_train: pd.DataFrame, categorical_cols: list[str]
) -> pd.DataFrame:
    """Fit the encoder once on X_train and return the transformed features."""
    missing = [c for c in categorical_cols if c not in X_train.columns]
    if missing:
        logger.error("Categorical columns missing from features: %s", missing)
        raise KeyError(f"Missing expected columns for encoding: {missing}")

    try:
        encoder.fit(X_train)
        X_train_encoded = encoder.transform(X_train)
    except ValueError as e:
        # e.g. unexpected dtypes, all-NaN categorical column, etc.
        logger.error("Encoder fit/transform failed: %s", e)
        raise

    logger.info("Data encoded successfully (%d -> %d columns)",
                X_train.shape[1], X_train_encoded.shape[1])
    return X_train_encoded


def train_model(X_train_encoded: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Fit a LinearRegression model."""
    if len(X_train_encoded) != len(y_train):
        logger.error(
            "Row count mismatch: X has %d rows, y has %d rows",
            len(X_train_encoded), len(y_train)
        )
        raise ValueError("Feature and target row counts do not match")

    if X_train_encoded.isna().any().any():
        logger.error("Encoded features contain NaN values after transform")
        raise ValueError("NaNs present in encoded training features")

    lr = LinearRegression()
    try:
        lr.fit(X_train_encoded, y_train)
    except ValueError as e:
        logger.error("Model fitting failed: %s", e)
        raise

    logger.info("Model trained successfully")
    return lr


def save_artifact(obj, save_path: Path, label: str) -> None:
    """Save a fitted object (encoder or model) to disk via joblib."""
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, save_path)
    except PermissionError:
        logger.error("Permission denied writing %s to %s", label, save_path)
        raise
    except OSError as e:
        logger.error("OS error writing %s to %s: %s", label, save_path, e)
        raise

    logger.info("%s saved successfully to %s", label, save_path)


def main() -> None:
    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent
    data_path = root_path / "data/processed/train.csv"

    df = load_data(data_path)
    df.set_index("tpep_pickup_datetime", inplace=True)

    X_train, y_train = make_xy(df)

    categorical_cols = ["region", "day_of_week"]
    encoder = build_encoder(categorical_cols)
    X_train_encoded = fit_transform_encoder(encoder, X_train, categorical_cols)

    encoder_save_path = root_path / "models/encoder.joblib"
    save_artifact(encoder, encoder_save_path, "Encoder")

    lr = train_model(X_train_encoded, y_train)

    model_save_path = root_path / "models/model.joblib"
    save_artifact(lr, model_save_path, "Model")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Model training pipeline failed")
        sys.exit(1)