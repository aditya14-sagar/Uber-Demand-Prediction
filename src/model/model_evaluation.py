import sys
import json
import logging
from pathlib import Path

import mlflow
import dagshub
import pandas as pd
import joblib
from sklearn import set_config
from sklearn.metrics import mean_absolute_percentage_error


# create a logger
logger = logging.getLogger("evaluate_model")
logger.setLevel(logging.INFO)

# attach a console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

# make a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)


def init_dagshub_mlflow(repo_owner: str, repo_name: str, experiment_name: str) -> None:
    """Initialize DagsHub + MLflow tracking. Raises if auth/connection fails."""
    try:
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    except Exception as e:
        logger.error("DagsHub initialization failed (check credentials/network): %s", e)
        raise RuntimeError("Could not initialize DagsHub tracking") from e

    try:
        mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        logger.error("Failed to configure MLflow tracking URI/experiment: %s", e)
        raise RuntimeError("Could not configure MLflow tracking") from e

    logger.info("DagsHub/MLflow tracking initialized successfully")


def load_dataset(data_path: Path, target_col: str = "total_pickups") -> pd.DataFrame:
    """Read a processed CSV with the pickup datetime parsed and set as index."""
    try:
        df = pd.read_csv(data_path, parse_dates=["tpep_pickup_datetime"])
    except FileNotFoundError:
        logger.error("Data file not found: %s", data_path)
        raise
    except ValueError as e:
        logger.error("Failed to parse %s (check columns/format): %s", data_path, e)
        raise
    except pd.errors.EmptyDataError:
        logger.error("Data file is empty: %s", data_path)
        raise
    except pd.errors.ParserError as e:
        logger.error("CSV parsing error in %s: %s", data_path, e)
        raise

    if df.empty:
        logger.error("Loaded dataframe has zero rows: %s", data_path)
        raise ValueError(f"No rows found in {data_path}")

    if target_col not in df.columns:
        logger.error("Target column '%s' missing from %s", target_col, data_path)
        raise KeyError(f"'{target_col}' missing from {data_path}")

    df.set_index("tpep_pickup_datetime", inplace=True)
    logger.info("Loaded %s (%d rows, %d columns)", data_path.name, *df.shape)
    return df


def load_joblib_artifact(path: Path, label: str):
    """Load a joblib artifact (encoder or model) with a clear error on failure."""
    if not path.exists():
        logger.error("%s file not found: %s", label, path)
        raise FileNotFoundError(f"{label} not found at {path}")
    try:
        artifact = joblib.load(path)
    except Exception as e:
        logger.error("Failed to load %s from %s: %s", label, path, e)
        raise RuntimeError(f"Could not load {label} from {path}") from e
    logger.info("%s loaded successfully", label)
    return artifact


def evaluate_predictions(model, X_test_encoded, y_test) -> tuple:
    """Generate predictions and compute MAPE."""
    try:
        y_pred = model.predict(X_test_encoded)
    except Exception as e:
        logger.error("Model prediction failed: %s", e)
        raise RuntimeError("Model.predict() failed on test data") from e

    if len(y_pred) != len(y_test):
        logger.error("Prediction length (%d) does not match y_test length (%d)",
                      len(y_pred), len(y_test))
        raise ValueError("Mismatch between prediction and target lengths")

    try:
        loss = mean_absolute_percentage_error(y_test, y_pred)
    except Exception as e:
        logger.error("MAPE computation failed: %s", e)
        raise RuntimeError("Could not compute MAPE") from e

    logger.info("Loss (MAPE): %s", loss)
    return y_pred, loss


def log_to_mlflow(model, X_test_encoded, y_pred, loss, train_data_path, test_data_path):
    """Run the full MLflow logging block: params, metric, datasets, model."""
    try:
        with mlflow.start_run(run_name="model") as run:
            try:
                mlflow.log_params(model.get_params())
            except Exception as e:
                # non-fatal: log a warning but continue the run
                logger.warning("Could not log model params: %s", e)

            try:
                mlflow.log_metric("MAPE", loss)
            except Exception as e:
                logger.error("Failed to log MAPE metric: %s", e)
                raise

            try:
                training_data = mlflow.data.from_pandas(
                    pd.read_csv(train_data_path, parse_dates=["tpep_pickup_datetime"])
                    .set_index("tpep_pickup_datetime"),
                    targets="total_pickups",
                )
                validation_data = mlflow.data.from_pandas(
                    pd.read_csv(test_data_path, parse_dates=["tpep_pickup_datetime"])
                    .set_index("tpep_pickup_datetime"),
                    targets="total_pickups",
                )
                mlflow.log_input(training_data, "training")
                mlflow.log_input(validation_data, "validation")
            except Exception as e:
                logger.error("Failed to log datasets to MLflow: %s", e)
                raise

            try:
                model_signature = mlflow.models.infer_signature(X_test_encoded, y_pred)
                logged_model = mlflow.sklearn.log_model(
                    model, "demand_prediction",
                    signature=model_signature,
                    pip_requirements="requirements.txt",
                )
            except Exception as e:
                logger.error("Failed to log sklearn model to MLflow: %s", e)
                raise

    except Exception as e:
        logger.error("MLflow run failed and will not be considered logged: %s", e)
        raise RuntimeError("MLflow logging failed") from e

    logger.info("MLflow logging complete (run_id=%s)", run.info.run_id)
    return logged_model


def save_run_information(run_id: str, artifact_path: str, model_uri: str, path: Path) -> None:
    """Write run metadata to a JSON file."""
    run_information = {
        "run_id": run_id,
        "artifact_path": artifact_path,
        "model_uri": model_uri,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(run_information, f, indent=4)
    except OSError as e:
        logger.error("Failed to write run information to %s: %s", path, e)
        raise

    logger.info("Run information saved successfully to %s", path)


def main() -> None:
    set_config(transform_output="pandas")

    init_dagshub_mlflow(
        repo_owner="aditya14-sagar",
        repo_name="Uber-Demand-Prediction",
        experiment_name="DVC Pipeline",
    )

    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent
    train_data_path = root_path / "data/processed/train.csv"
    test_data_path = root_path / "data/processed/test.csv"

    df_test = load_dataset(test_data_path)
    X_test = df_test.drop(columns=["total_pickups"])
    y_test = df_test["total_pickups"]

    encoder_path = root_path / "models/encoder.joblib"
    encoder = load_joblib_artifact(encoder_path, "Encoder")

    try:
        X_test_encoded = encoder.transform(X_test)
    except Exception as e:
        logger.error("Encoder transform failed on test data: %s", e)
        raise RuntimeError("Could not transform test features") from e
    logger.info("Data transformed successfully")

    model_path = root_path / "models/model.joblib"
    model = load_joblib_artifact(model_path, "Model")

    y_pred, loss = evaluate_predictions(model, X_test_encoded, y_test)

    logged_model = log_to_mlflow(
        model, X_test_encoded, y_pred, loss, train_data_path, test_data_path
    )

    run_id = logged_model.run_id
    artifact_path = logged_model.artifact_path
    model_uri = logged_model.model_uri

    json_file_save_path = root_path / "run_information.json"
    save_run_information(
        run_id=run_id,
        artifact_path=artifact_path,
        model_uri=model_uri,
        path=json_file_save_path,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Model evaluation pipeline failed")
        sys.exit(1)