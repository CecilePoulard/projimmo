"""
main.py

Ce script réalise l'ingestion, le prétraitement, l'entraînement et la sauvegarde d'un modèle prédictif
basé sur les données de transactions immobilières (DVF). Le script suit les étapes suivantes :
1. Chargement des fichiers locaux de données DVF.
2. Nettoyage et vérification des données avant chargement dans BigQuery.
3. Prétraitement des caractéristiques et création d'une table prétraitée dans BigQuery.
4. Entraînement de plusieurs modèles (régression linéaire, XGBoost, KNN), sélection du meilleur
   modèle en fonction du score R² et sauvegarde du modèle optimisé.

Fonctions principales :
- `load_all_files()`: Charge les fichiers locaux de données, applique le nettoyage et le chargement dans BigQuery.
- `clean_and_load(df, table)`: Nettoie et charge un DataFrame dans BigQuery.
- `preprocess(table)`: Applique des transformations aux données, enregistre les données prétraitées dans BigQuery.
- `train_and_save_best_model(X, y)`: Entraîne et évalue plusieurs modèles, sauvegarde le modèle ayant la meilleure performance.

Exécution :
Le script peut être exécuté directement, en important et traitant les données puis en entraînant le modèle final,
ou en utilisant chaque fonction de façon indépendante.

Prérequis :
- Bibliothèques externes : `pandas`, `bigquery`, `joblib`, `xgboost`, `sklearn`
- Variables globales : `LOCAL_DATA_PATH`, `BQ_DATASET`, `GCP_PROJECT`, `DATA_YEAR`, et autres paramètres définis
  dans les modules de `projimmo`.
"""


import os
import pandas as pd
from google.cloud import bigquery
from pathlib import Path
import joblib
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from projimmo.params import *
from projimmo.data import *
from projimmo.preprocessor import *
from projimmo.model import *
from projimmo.registry import *
from projimmo.optimize_model import *

def load_all_files():
    """
    Charge tous les fichiers CSV présents dans le répertoire spécifié par `LOCAL_DATA_PATH`.
    Concatène les données en un DataFrame, nettoie les données, et les charge dans BigQuery
    si elles n'y sont pas déjà.

    Returns:
        str: Nom de la table chargée dans BigQuery.
    """
    data_local_path = Path(LOCAL_DATA_PATH)

    # Vérifie que le chemin spécifié est un répertoire
    if not data_local_path.is_dir():
        raise ValueError(f"{data_local_path} n'est pas un répertoire valide.")

    table = f"DVF_cleaned_{DATA_YEAR}"
    all_files = list(data_local_path.glob('*.txt'))

    if not all_files:
        print("Aucun fichier CSV trouvé dans le répertoire spécifié.")
        return table

    client = bigquery.Client()

    # Vérifie si la table existe déjà dans BigQuery
    if table_exists(client, BQ_DATASET, table):
        print(f"La table {table} existe déjà. Aucune donnée ne sera chargée.")
        return table

    # Chargement par morceaux pour éviter de surcharger la mémoire
    for file in all_files:
        for chunk in pd.read_csv(file, sep="|", dtype=str, chunksize=CHUNK_SIZE):
            clean_and_load(chunk, table)

    return table


def clean_and_load(df, table):
    """
    Nettoie et charge un DataFrame dans BigQuery.

    Args:
        df (pd.DataFrame): Les données à nettoyer et charger.
        table (str): Nom de la table BigQuery cible.
    """
    df = clean_data(df)
    print("✅ Nettoyage effectué")
    df = clean_outliers(df)
    print("✅ Suppression des valeurs aberrantes terminée")

    load_data_to_bq(
        data=df,
        gcp_project=GCP_PROJECT,
        bq_dataset=BQ_DATASET,
        table=table,
        truncate=False
    )
    print("✅ Chargement dans BigQuery effectué")


def data_already_loaded(table, df):
    """
    Vérifie si les données du DataFrame sont déjà présentes dans une table BigQuery.

    Args:
        table (str): Nom de la table BigQuery.
        df (pd.DataFrame): Les données à vérifier.

    Returns:
        bool: True si les données sont déjà présentes, False sinon.
    """
    client = bigquery.Client()
    unique_ids = df['id'].unique().tolist()

    query = f"""
        SELECT id
        FROM `{GCP_PROJECT}.{BQ_DATASET}.{table}`
        WHERE id IN UNNEST(@unique_ids)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("unique_ids", "STRING", unique_ids)
        ]
    )

    results = client.query(query, job_config=job_config).result().to_dataframe()

    if not results.empty:
        print("Les données existent déjà dans la table.")
        return True
    return False


def table_exists(client, dataset_id, table_id):
    """
    Vérifie si une table existe dans un dataset BigQuery.

    Args:
        client (bigquery.Client): Client BigQuery.
        dataset_id (str): Nom du dataset.
        table_id (str): Nom de la table.

    Returns:
        bool: True si la table existe, False sinon.
    """
    try:
        client.get_table(f"{dataset_id}.{table_id}")
        return True
    except Exception as e:
        if "Not found" in str(e):
            return False
        raise


def preprocess(table):
    """
    Prépare les données pour le modèle en appliquant des transformations,
    puis les enregistre dans BigQuery.

    Args:
        table (str): Nom de la table à prétraiter.

    Returns:
        tuple: Nom de la table prétraitée dans BigQuery et DataFrame des données prétraitées.
    """
    query = f"SELECT * FROM `{GCP_PROJECT}`.{BQ_DATASET}.{table}"
    table_out = f"DVF_preproc_{DATA_YEAR}"
    client = bigquery.Client()

    if table_exists(client, BQ_DATASET, table_out):
        print(f"La table {table_out} existe déjà. Les données ont déjà été prétraitées.")
        return table_out

    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath(f"query_{DATA_YEAR}.csv")

    df_to_preproc = get_data_with_cache(
        query=query,
        gcp_project=GCP_PROJECT,
        cache_path=data_query_cache_path,
        data_has_header=True
    )

    # Prétraitement des caractéristiques
    df_preproc, preproc_baseline = preprocess_features(df_to_preproc)
    joblib.dump(preproc_baseline, 'preproc_baseline.pkl')

    load_data_to_bq(
        data=df_preproc,
        gcp_project=GCP_PROJECT,
        bq_dataset=BQ_DATASET,
        table=table_out,
        truncate=True
    )
    return table_out, df_preproc


def train_and_save_best_model(X, y, model_types=['KNR', 'LR', 'XGB'], test_size=0.2):
    """
    Entraîne plusieurs modèles, sélectionne le meilleur en fonction du score R²,
    et sauvegarde ce modèle.

    Args:
        X (pd.DataFrame): Caractéristiques des données.
        y (pd.Series): Cibles des données.
        model_types (list, optional): Liste des types de modèles à entraîner.
        test_size (float, optional): Taille de l'ensemble de test.

    Returns:
        tuple: Le meilleur modèle entraîné et son type.
    """
    X_train, X_test, y_train, y_test = split_df(X, y, test_size=test_size)
    best_model = None
    best_model_type = None
    best_r2_score = -float('inf')

    for model_type in model_types:
        print(f"Entraînement du modèle: {model_type}")
        model = initialize_model(model_type)
        model = train_model(X_train, y_train, model)
        metrics = evaluate_model(X_test, y_test, model, model_type)

        # Mise à jour du meilleur modèle si R² est supérieur
        if metrics['r2'] > best_r2_score:
            best_r2_score = metrics['r2']
            best_model = model
            best_model_type = model_type

    if best_model is None:
        raise ValueError("Aucun modèle n'a été entraîné correctement.")

    save_model(best_model)
    return best_model, best_model_type


if __name__ == "__main__":
    # Charger et traiter les fichiers de données
    table_to_preproc = load_all_files()
    table_preproc, df_preproc = preprocess(table_to_preproc)

    # Préparer les caractéristiques pour le modèle
    X = df_preproc.drop(columns='valeur_fonciere')
    y = df_preproc['valeur_fonciere']

    # Entraîner et sauvegarder le meilleur modèle
    best_model, best_model_type = train_and_save_best_model(X, y)
    print(f"Le meilleur modèle est de type {best_model_type}")
