"""
projimmo/registry.py



Module for saving and loading XGBoost models.

This module provides functionality to save a trained XGBoost model either locally or to Google Cloud Storage (GCS),
and to load the most recent model from the specified target (local or GCS). The target storage location is
configured via the `MODEL_TARGET` variable, which determines whether the model is stored locally or on GCS.

Functions:
    - save_model: Saves the trained XGBoost model to a specified location (local or GCS).
    - load_model: Loads the most recent XGBoost model from the specified location (local or GCS).

Configuration:
    - `MODEL_TARGET`: Specifies where the model should be saved or loaded from ('local' or 'gcs').
    - `LOCAL_REGISTRY_PATH`: Path to the local directory where models are stored.
    - `BUCKET_NAME`: Name of the Google Cloud Storage bucket for storing/loading models.

Usage:
    1. Call `save_model(model)` to save a trained model. The model is saved with a timestamp in its name.
    2. Call `load_model()` to load the most recent model either from the local registry or from GCS.

Dependencies:
    - `joblib`: For saving and loading models locally.
    - `google.cloud.storage`: For interacting with Google Cloud Storage.
    - `xgboost`: For working with XGBoost models.
    - `colorama`: For colorful output in the console.
    - `glob`: For finding files in the local model directory.

Note:
    The functions depend on the configuration of `MODEL_TARGET`, which should be set to either 'local' or 'gcs'.
    The local model files should be saved in the directory specified by `LOCAL_REGISTRY_PATH`,
    and GCS storage requires proper configuration with `BUCKET_NAME`.
"""






import os
import time
import joblib
from google.cloud import storage
import xgboost as xgb
from colorama import Fore, Style
import glob
from projimmo.params import *

def save_model(model: xgb.Booster) -> None:
    """
    Persist a trained XGBoost model locally and optionally to Google Cloud Storage.

    This function saves the trained XGBoost model both locally (with a unique timestamp-based name)
    and to Google Cloud Storage if the target is configured as 'gcs' in the environment variables.

    Args:
        model (xgb.Booster): The trained XGBoost model to save.
    """
    # Generate a timestamp to create a unique filename for the model
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save the model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.pkl")
    joblib.dump(model, model_path)
    print("✅ Model saved locally")

    # If the target is Google Cloud Storage, upload the model
    if MODEL_TARGET == "gcs":
        model_filename = os.path.basename(model_path)
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to GCS")

    return None


def load_model() -> xgb.Booster:
    """
    Load the most recent XGBoost model either from local storage or Google Cloud Storage.

    - If MODEL_TARGET is 'local', loads the most recent model from the local registry.
    - If MODEL_TARGET is 'gcs', loads the most recent model from Google Cloud Storage.

    Returns:
        xgb.Booster: The most recent trained XGBoost model, or None if no model is found.
    """
    print(Fore.BLUE + f"\nLoading model from {MODEL_TARGET}" + Style.RESET_ALL)

    # Load the latest model from local storage
    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoading latest model from local registry..." + Style.RESET_ALL)

        # Get the list of all model files in the local model directory
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*.pkl")

        # If no models are found locally, return None
        if not local_model_paths:
            return None

        # Sort models by filename to get the most recent one based on timestamp
        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        print(Fore.BLUE + f"\nLoading latest model from disk..." + Style.RESET_ALL)
        latest_model = joblib.load(most_recent_model_path_on_disk)
        print("✅ Model loaded from local disk")

        return latest_model

    # Load the latest model from Google Cloud Storage
    elif MODEL_TARGET == "gcs":
        print(Fore.BLUE + f"\nLoading latest model from GCS..." + Style.RESET_ALL)

        # Connect to GCS and get a list of blobs under the 'models/' prefix
        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="models/"))

        # If no blobs are found, return None
        if not blobs:
            print(f"❌ No models found in GCS bucket {BUCKET_NAME}")
            return None

        # Find the most recently uploaded model
        latest_blob = max(blobs, key=lambda x: x.updated)

        # Download the latest model to the local machine
        latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, "models", os.path.basename(latest_blob.name))
        os.makedirs(os.path.dirname(latest_model_path_to_save), exist_ok=True)
        latest_blob.download_to_filename(latest_model_path_to_save)

        # Load the model from the downloaded file
        latest_model = joblib.load(latest_model_path_to_save)
        print("✅ Latest model downloaded from cloud storage")

        return latest_model

    else:
        print(f"❌ Invalid MODEL_TARGET: {MODEL_TARGET}")
        return None
