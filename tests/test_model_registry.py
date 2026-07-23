import warnings
import mlflow
import dagshub
import json
import numpy as np
from sklearn.exceptions import InconsistentVersionWarning
import pandas as pd

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

dagshub.init(repo_owner='aditya14-sagar', repo_name='Uber-Demand-Prediction', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/aditya14-sagar/Uber-Demand-Prediction.mlflow")


def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
    return run_info


run_info = load_model_information("run_information.json")
model_uri = run_info["artifact_path"]
model = mlflow.sklearn.load_model(model_uri)


def test_load_model_from_registry():
    assert model is not None, "Failed to load model from registry"



def test_model_predicts():
    feature_names = model.feature_names_in_
    dummy_input = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
    pred = model.predict(dummy_input)
    assert pred is not None, "Model failed to produce a prediction"


if __name__ == "__main__":
    test_load_model_from_registry()
    test_model_predicts()
    print("✅ All tests passed")