# projimmo/params.py

import os
import numpy as np

##################  VARIABLES  ##################
# Définition des variables liées à l'environnement et à la configuration des paramètres.

# Taille des données (par défaut, la taille est définie par une variable d'environnement)
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1_000_000))  # Taille des morceaux de données à traiter.
DATA_DIR = os.environ.get("DATA_DIR")  # Répertoire des données brutes.
DATA_YEAR = os.environ.get("DATA_YEAR")  # Année des données à traiter.
MODEL_TARGET = os.environ.get("MODEL_TARGET")  # Cible du modèle (ex : "gcs", "mlflow", etc.).
GCP_PROJECT = os.environ.get("GCP_PROJECT")  # Projet GCP (Google Cloud Platform).
GCP_PROJECT_WAGON = os.environ.get("GCP_PROJECT_WAGON")  # Projet GCP pour Wagon.
GCP_REGION = os.environ.get("GCP_REGION")  # Région GCP où les services sont hébergés.
BQ_DATASET = os.environ.get("BQ_DATASET")  # Dataset BigQuery à utiliser.
BQ_REGION = os.environ.get("BQ_REGION")  # Région de BigQuery.
BUCKET_NAME = os.environ.get("BUCKET_NAME")  # Nom du bucket GCS (Google Cloud Storage).
INSTANCE = os.environ.get("INSTANCE")  # Nom de l'instance à utiliser.
GAR_IMAGE = os.environ.get("GAR_IMAGE")  # Nom de l'image Google Artifact Registry.
GAR_MEMORY = os.environ.get("GAR_MEMORY")  # Quantité de mémoire allouée à l'image.

##################  CONSTANTS  #####################
# Constantes locales pour les chemins de fichiers et les configurations de données.

LOCAL_DATA_PATH = "raw_data/"  # Chemin local pour les données brutes.
LOCAL_REGISTRY_PATH = "registry/"  # Répertoire local pour l'enregistrement des modèles.

# Noms des colonnes dans les données brutes.
COLUMN_NAMES_RAW = ['Date mutation', 'Nature mutation', 'Valeur fonciere', 'Type de voie',
                    'Code postal', 'Surface Carrez du 1er lot', 'Surface Carrez du 2eme lot',
                    'Surface Carrez du 3eme lot', 'Surface Carrez du 4eme lot', 'Surface Carrez du 5eme lot',
                    'Nombre de lots', 'Code type local', 'Surface reelle bati',
                    'Nombre pieces principales', 'Surface terrain']

# Liste des départements utilisés dans les données.
DEPARTEMENTS = ['75', '13', '69', '31', '06', '44', '34', '67', '33', '59']

# Types de données pour les colonnes des données brutes.
DTYPES_RAW = {
    'valeur_fonciere': "float",
    'type_de_voie': "str",
    'departement': "int",
    'nombre_de_lots': "int",
    'code_type_local': "int",
    'surface_reelle_bati': "float",
    'nombre_pieces_principales': "float",
    'surface_terrain': "float",
    'somme_surface_carrez': "float"
}

# Grilles de paramètres pour les modèles de machine learning.
PARAM_GRID_KNN = {
    'n_neighbors': [3, 5, 7, 9],  # Nombre de voisins à tester pour KNN.
    # 'weights': ['uniform', 'distance'],  # Poids des voisins à tester (commenté pour l'instant).
    # 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']  # Algorithmes de recherche à tester (commenté pour l'instant).
}

PARAM_GRID_LR = {
    'alpha': [0.1, 1, 10, 100]  # Valeurs de régularisation à tester pour Ridge ou Lasso (régression linéaire).
}

PARAM_GRID_XGB = {
    # 'objective': ['reg:squarederror', 'reg:squaredlogerror'],  # Objectifs à tester (commenté pour l'instant).
    'max_depth': [3, 4, 5, 6],  # Profondeur des arbres à tester.
    'eta': [0.01, 0.1, 0.2],  # Taux d'apprentissage à tester.
    'n_estimators': [50, 100, 150]  # Nombre d'estimateurs (arbres) à tester.
}

################## VALIDATIONS #################
# Section commentée pour valider les valeurs d'environnement.
# Si nécessaire, vous pouvez décommenter et personnaliser cette section pour valider les configurations.

# Définition des options valides pour les variables d'environnement.
# env_valid_options = dict(
#     DATA_SIZE=["1k", "200k", "all"],
#     MODEL_TARGET=["local", "gcs", "mlflow"],
# )

# Fonction de validation des valeurs des variables d'environnement.
# def validate_env_value(env, valid_options):
#     env_value = os.environ[env]
#     if env_value not in valid_options:
#         raise NameError(f"Invalid value for {env} in `.env` file: {env_value} must be in {valid_options}")

# Validation des options pour chaque variable d'environnement.
# for env, valid_options in env_valid_options.items():
#     validate_env_value(env, valid_options)
