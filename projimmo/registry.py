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
    Persist trained XGBoost model locally and optionally to Google Cloud Storage.
    """


#Création d'un horodatage avec la date et l'heure actuelles, ce qui garantit que
#chaque modèle sauvegardé aura un nom unique basé sur le moment où il a été sauvegardé.
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.pkl")
    joblib.dump(model, model_path)
#    model.save_model(model_path)

    print("✅ Model saved locally")

    if MODEL_TARGET == "gcs":
        # Upload model to Google Cloud Storage
        model_filename = os.path.basename(model_path)
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to GCS")

    return None



def load_model() -> xgb.Booster:
    """
    Return a saved XGBoost model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'
    """
    print(Fore.BLUE + f"\nDans la fonction {MODEL_TARGET}" + Style.RESET_ALL)

    if MODEL_TARGET == "local":

        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

        # Get the latest model version name by the timestamp on disk
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*.pkl")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)
        #latest_model = xgb.Booster()
        #latest_model.load_model(most_recent_model_path_on_disk)
        latest_model = joblib.load(most_recent_model_path_on_disk)
        print("✅ Model loaded from local disk")

        return latest_model

    elif MODEL_TARGET == "gcs":
        print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)

        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="models/"))
            #if not blobs:
        print(f"❌ {blobs}")
        try:
            print(f"❌❌❌❌❌ {BUCKET_NAME}")


            #    return None
            latest_blob = max(blobs, key=lambda x: x.updated)
            #if not latest_blob:
            print(f"❌ {latest_blob}")
            #    return None
            latest_model_path_to_save = os.path.join(latest_blob.name) #LOCAL_REGISTRY_PATH,
            print(f"{latest_model_path_to_save}")
            #if not latest_model_path_to_save:
            #    print(f"❌ No latest_model_path_to_save")
            #    return None
            os.makedirs(os.path.dirname(latest_model_path_to_save), exist_ok=True)
            latest_blob.download_to_filename(latest_model_path_to_save)
            latest_model = joblib.load(latest_model_path_to_save)
            print(f"{latest_model}")
            #latest_model = xgb.Booster()
            #latest_model.load_model(latest_model_path_to_save)

            print("✅ Latest model downloaded from cloud storage")

            return latest_model
        except:
            print(f"❌ No model found in GCS bucket {BUCKET_NAME}")

            return None