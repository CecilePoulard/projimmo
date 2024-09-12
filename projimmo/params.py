#projimmo/params.py
import os
import numpy as np

##################  VARIABLES  ##################
DATA_SIZE = os.environ.get("DATA_SIZE")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE"))
DATA_DIR = os.environ.get("DATA_DIR")
DATA_YEAR = os.environ.get("DATA_YEAR")
MODEL_TARGET = os.environ.get("MODEL_TARGET")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_PROJECT_WAGON = os.environ.get("GCP_PROJECT_WAGON")
GCP_REGION = os.environ.get("GCP_REGION")
BQ_DATASET = os.environ.get("BQ_DATASET")
BQ_REGION = os.environ.get("BQ_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
INSTANCE = os.environ.get("INSTANCE")
#MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
#MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
#MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")
#PREFECT_FLOW_NAME = os.environ.get("PREFECT_FLOW_NAME")
#PREFECT_LOG_LEVEL = os.environ.get("PREFECT_LOG_LEVEL")
#EVALUATION_START_DATE = os.environ.get("EVALUATION_START_DATE")
#GAR_IMAGE = os.environ.get("GAR_IMAGE")
#GAR_MEMORY = os.environ.get("GAR_MEMORY")

##################  CONSTANTS  #####################
LOCAL_DATA_PATH = "raw_data/"#valeursfoncieres-2023.txt
#os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "data")
#LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "training_outputs")

COLUMN_NAMES_RAW = ['Date mutation', 'Nature mutation', 'Valeur fonciere',
        'Type de voie', 'Code postal', 'Surface Carrez du 1er lot',
        'Surface Carrez du 2eme lot', 'Surface Carrez du 3eme lot',
        'Surface Carrez du 4eme lot', 'Surface Carrez du 5eme lot',
        'Nombre de lots', 'Code type local', 'Surface reelle bati',
        'Nombre pieces principales', 'Surface terrain']

DEPARTEMENTS=['75', '13', '69', '31', '06', '44', '34', '67', '33', '59']

DTYPES_RAW = {
    'valeur_fonciere':"float",
    'type_de_voie': "str",
    'departement':"int",
    'nombre_de_lots': "int",
    'code_type_local': "int",
    'surface_reelle_bati': "float",
    'nombre_pieces_principales': "float",
       'surface_terrain': "float",
       'somme_surface_carrez': "float"}

PARAM_GRID_KNN= {
    'n_neighbors': [3, 5, 7, 9],
    #'weights': ['uniform', 'distance'],
    #'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

PARAM_GRID_LR= {
    'alpha': [0.1, 1, 10, 100]  # Pour Ridge ou Lasso,à modifier selon modele
}


PARAM_GRID_XGB = {
    'objective': ['reg:squarederror', 'reg:squaredlogerror'],
    'max_depth': [3, 4, 5, 6],
    'eta': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 150]
}

#DTYPES_PROCESSED = np.float32



################## VALIDATIONS #################

#env_valid_options = dict(
#    DATA_SIZE=["1k", "200k", "all"],
#    MODEL_TARGET=["local", "gcs", "mlflow"],
#)

#def validate_env_value(env, valid_options):
#    env_value = os.environ[env]
#    if env_value not in valid_options:
#        raise NameError(f"Invalid value for {env} in `.env` file: {env_value} must be in {valid_options}")


#for env, valid_options in env_valid_options.items():
#    validate_env_value(env, valid_options)
