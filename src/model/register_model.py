import sys
import json
import logging
from pathlib import Path

import mlflow
import dagshub
from mlflow.client import MlflowClient


# create a logger
logger = logging.getLogger("register_model")
logger.setLevel(logging.INFO)

# attach a console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

# make a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)


def init_dagshub_mlflow(repo_owner: str, repo_name: str) -> None:
    """Initialize DagsHub + MLflow tracking. Raises if auth/connection fails."""
    try:
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    except Exception as e:
        logger.error("DagsHub initialization failed (check credentials/network): %s", e)
        raise RuntimeError("Could not initialize DagsHub tracking") from e

    try:
        mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")
    except Exception as e:
        logger.error("Failed to set MLflow tracking URI: %s", e)
        raise RuntimeError("Could not configure MLflow tracking") from e

    logger.info("DagsHub/MLflow tracking initialized successfully")


def load_run_info(path: Path) -> dict:
    """Load run_information.json and validate it has the required key."""
    try:
        with open(path, "r") as f:
            run_info = json.load(f)
    except FileNotFoundError:
        logger.error("Run information file not found: %s", path)
        raise
    except json.JSONDecodeError as e:
        logger.error("Error decoding JSON from %s: %s", path, e)
        raise
    except OSError as e:
        logger.error("Could not read %s: %s", path, e)
        raise

    if "model_uri" not in run_info:
        logger.error("'model_uri' key missing from %s", path)
        raise KeyError(f"'model_uri' missing from {path}")

    logger.info("Run information loaded successfully")
    return run_info


def register_model(model_uri: str, model_name: str):
    """Register the model with the MLflow model registry."""
    try:
        model_version = mlflow.register_model(model_uri, model_name)
    except Exception as e:
        logger.error("Model registration failed for uri=%s, name=%s: %s", model_uri, model_name, e)
        raise RuntimeError("mlflow.register_model() failed") from e

    logger.info(
        "Model registered successfully with version: %s and name: %s",
        model_version.version, model_version.name
    )
    return model_version


def transition_stage(client: MlflowClient, model_name: str, model_version: str, stage: str):
    """Transition a registered model version to a new stage."""
    try:
        stage_version = client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage=stage,
            archive_existing_versions=False,
        )
    except Exception as e:
        logger.error(
            "Stage transition failed for name=%s, version=%s, stage=%s: %s",
            model_name, model_version, stage, e
        )
        raise RuntimeError("transition_model_version_stage() failed") from e

    logger.info(
        "Model moved to stage: %s with version: %s and name: %s",
        stage_version.current_stage, stage_version.version, stage_version.name
    )
    return stage_version


def main() -> None:
    init_dagshub_mlflow(
        repo_owner="aditya14-sagar",
        repo_name="Uber-Demand-Prediction",
    )

    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent
    file_name = "run_information.json"

    run_info = load_run_info(root_path / file_name)
    model_uri = run_info["model_uri"]

    model_name = "uber_demand_prediction_model"
    model_version_obj = register_model(model_uri, model_name)

    model_name = model_version_obj.name
    model_version = model_version_obj.version

    client = MlflowClient()
    model_stage = "Staging"
    transition_stage(client, model_name, model_version, model_stage)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Model registration pipeline failed")
        sys.exit(1)