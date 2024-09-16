#main.py

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
    Charge tous les fichiers CSV dans le répertoire local_path,
    les concatène en un seul DataFrame.
    """
    data_local_path = Path(LOCAL_DATA_PATH)
    # Verif que data_local_path bien un répertoire
    if not data_local_path.is_dir():
        raise ValueError(f"{data_local_path} n'est pas un répertoire valide.")

    table=f"DVF_cleaned_{DATA_YEAR}"
    all_files = list(data_local_path.glob('*.txt'))
    if not all_files:
        print("Aucun fichier CSV trouvé dans le répertoire spécifié.")
        return table
    client = bigquery.Client()
    if table_exists(client, BQ_DATASET, table):
        print(f"La table {table} existe déjà. Aucune donnée ne sera chargée.")
        return table

    for file in all_files:
        for chunk in pd.read_csv(file, sep="|", dtype=str, chunksize=CHUNK_SIZE):
            #if not data_already_loaded(table, chunk):
            clean_and_load(chunk, table)
            #else:
            #    print(f"Les données de {file} sont déjà chargées dans la table.")


    return table


def clean_and_load(df, table):
    """
    Nettoie le df et le charge dans bigQuery
    """
    #df = pd.concat(df_list, ignore_index=True)
    #print("concat ok")
    df = clean_data(df)
    print("✅  clean ok \n")
    df = clean_outliers(df)
    print("✅ outliers ok\n")

    load_data_to_bq(
    data=df,
    gcp_project=GCP_PROJECT,
    bq_dataset=BQ_DATASET,
    table=table,
    truncate=False ) # Append the data to the existing table
    print("✅ load ok\n")


def data_already_loaded(table, df):
    """
    Vérifie si les données sont déjà présentes dans la table BigQuery
    Retourne True si les données sont déjà présentes, False sinon
    """
    client = bigquery.Client()

    # Extraire une ou plusieurs colonnes uniques de ton dataframe pour vérifier s'ils existent déjà
    # Assumons que tu as une colonne 'id' ou un identifiant unique
    unique_ids = df['id'].unique().tolist()

    # Créer une requête pour vérifier les ids déjà présents
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

    query_job = client.query(query, job_config=job_config)

    results = query_job.result().to_dataframe()

    if not results.empty:
        print("Les données existent déjà dans la table.")
        return True
    else:
        return False




def table_exists(client, dataset_id, table_id):
    """
    Vérifie si la table existe dans le dataset BigQuery.
    """
    try:
        client.get_table(f"{dataset_id}.{table_id}")
        return True
    except Exception as e:
        if "Not found" in str(e):
            return False
        else:
            raise








def preprocess(table):
    #table_cleaned=load_all_files()
    query = f"""
        SELECT *
        FROM `{GCP_PROJECT}`.{BQ_DATASET}.{table}
    """
    table_out= f"DVF_preproc_{DATA_YEAR}"
    client = bigquery.Client()
    if table_exists(client, BQ_DATASET, table_out):
        print(f"La table {table_out} existe déjà. \nLes datas ont déjà été préprocess.")
        return table_out
    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath(f"query_{DATA_YEAR}.csv")
    df_to_preproc=get_data_with_cache(
        query=query,
        gcp_project=GCP_PROJECT,
       cache_path=data_query_cache_path,
        data_has_header=True
   )
    # Prétraiter les caractéristiques
    global preproc_baseline
    df_preproc, preproc_baseline =preprocess_features(df_to_preproc)

    # Sauvegarder le transformateur
    joblib.dump(preproc_baseline, 'preproc_baseline.pkl')
    load_data_to_bq(
    data=df_preproc,
    gcp_project=GCP_PROJECT,
    bq_dataset=BQ_DATASET,
    table=table_out,
    truncate=True )
    return table_out,df_preproc




def train_and_save_best_model(X, y, model_types=['KNR', 'LR', 'XGB'], test_size=0.2):
    """
    Entraîne les modèles spécifiés, évalue leur performance, récupère le meilleur modèle, et l'enregistre en local.

    Parameters:
    - X: Les caractéristiques des données.
    - y: Les cibles des données.
    - model_types: Liste des types de modèles à entraîner.
    - test_size: Taille de l'ensemble de test pour la séparation des données.

    Returns:
    - best_model: Le meilleur modèle entraîné.
    - best_model_type: Le type de modèle du meilleur modèle.
    """

    # Séparer les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = split_df(X, y, test_size=test_size)

    best_model = None
    best_model_type = None
    best_r2_score = -float('inf')  # Commencer avec une valeur très basse pour les scores R²

    # Entraîner et évaluer chaque modèle
    for model_type in model_types:
        print(f"Entraînement du modèle: {model_type}")

        # Initialiser le modèle
        model = initialize_model(model_type)

        # Entraîner le modèle
        model = train_model(X_train, y_train, model)

        # Évaluer le modèle
        metrics = evaluate_model(X_test, y_test, model, model_type)

        # Comparer les performances
        if metrics['r2'] > best_r2_score:
            best_r2_score = metrics['r2']
            best_model = model
            best_model_type = model_type

    if best_model is None:
        raise ValueError("Aucun modèle n'a été entraîné correctement.")

    # Enregistrer le meilleur modèle
    #model_filename = f"best_model_{best_model_type}.pkl"
    #joblib.dump(best_model, model_filename)
    #print(f"Le meilleur modèle ({best_model_type}) a été enregistré sous le nom {model_filename}.")
    # Sauvegarder le modèle avec save_model
    save_model(best_model)

    return best_model, best_model_type



#def pre_model(X,y,model_type):
#    X_train,X_test,y_train,y_test=split_df(X,y)

#    model=initialize_model(model_type)
    # Effectuer la validation croisée
#    cv_results = cross_validate_model(X_train, y_train, model)
#    print("\nSans validation croisée:")
 #   fitted_model=train_model(X_train,y_train,model)
#    metrics=evaluate_model(X_test,y_test,fitted_model, model_type)
#    return metrics
if __name__ == "__main__":
    table_to_preproc=load_all_files()
    table_preproc=preprocess(table_to_preproc)


  #  query = f"""
  #      SELECT *
  #      FROM `{GCP_PROJECT}`.{BQ_DATASET}.{table_preproc}
  #  """
  #  data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath(f"query_{DATA_YEAR}.csv")
  #  df_preproc=get_data_with_cache(
  #      query=query,
  #      gcp_project=GCP_PROJECT,
  #     cache_path=data_query_cache_path,
  #      data_has_header=True
   #)
  #  X = df_preproc.drop(columns='valeur_fonciere')
  #  y = df_preproc['valeur_fonciere']

  #  best_model, best_model_type = train_and_save_best_model(X, y)
  #  X_train, X_test, y_train, y_test = split_df(X, y)
  #  model_type=best_model_type
    #best_model=initialize_model(best_model_type)#, max_depth=max_depth, eta=eta, n_estimators=n_estimators)

  #  best_model=load_model()
# Validation croisée sur les données transformées par PCA
  #  cv_results_best_model = cross_validate_model(X_train, y_train, best_model)
  #  mean_cv_best_mode = {'mean MSE': np.mean(- cv_results_best_model['test_score']),'STD MSE':np.std(- cv_results_best_model['test_score'])}

# Entraîner le modèle sur les données réduites
    #fitted_model_XGB_opti = train_model(X_train, y_train, model_XGB_opti)
  #  metrics_base= evaluate_model(X_test, y_test, best_model, model_type)
  #  best_params=grid_search(X_train, y_train, model_type,best_model)

  #  eta=best_params['eta']
  #  max_depth=best_params['max_depth']
  #  n_estimators=best_params['n_estimators']
   # model_XGB_opti=initialize_model(model_type, max_depth=max_depth, eta=eta, n_estimators=n_estimators)
        # Validation croisée sur les données transformées par PCA
   # cv_results_XGB_opti = cross_validate_model(X_train, y_train, model_XGB_opti)
  #  mean_cv_XGB_opti = {'mean MSE': np.mean(- cv_results_XGB_opti['test_score']),'STD MSE':np.std(- cv_results_XGB_opti['test_score'])}

        # Entraîner le modèle sur les données réduites
  #  fitted_model_XGB_opti = train_model(X_train, y_train, model_XGB_opti)

        # Évaluer le modèle sur les données de test transformées
   # metrics_XGB_opti = evaluate_model(X_test, y_test, fitted_model_XGB_opti, model_type)

  #  save_model(fitted_model_XGB_opti)
    # Create a dictionary to hold the results
   # results_dict = {
   #     'Model': ['Base', 'Optimized'],
   #     'MSE': [metrics_base['mse'], metrics_XGB_opti['mse']],
   #     'MAE': [metrics_base['mae'], metrics_XGB_opti['mae']],
  #      'R2': [metrics_base['r2'], metrics_XGB_opti['r2']],
  #      'RMSE': [metrics_base['rmse'], metrics_XGB_opti['rmse']],
   #     'MSLE': [metrics_base['msle'], metrics_XGB_opti['msle']],
   #     'Best Params': ['N/A', str(best_params)],
                # Ajout des scores de validation croisée pour les deux modèles
  #      'CV_MEAN_MSE': [mean_cv_best_mode['mean MSE'], mean_cv_XGB_opti['mean MSE']],
  #      'CV_STD_MSE': [mean_cv_best_mode['STD MSE'], mean_cv_XGB_opti['STD MSE']],
  #  }


    # Convert the dictionary to a pandas DataFrame
  #  results_df = pd.DataFrame(results_dict)

    # Save the DataFrame to a CSV file
   # results_df.to_csv('model_evaluation_results.csv', index=False)
   # print("✅ Model evaluation results saved to 'model_evaluation_results.csv'")



#def train(
 #       min_date:str = '2009-01-01',
 #       max_date:str = '2015-01-01',
 #       split_ratio: float = 0.02, # 0.02 represents ~ 1 month of validation data on a 2009-2015 train set
 #       learning_rate=0.0005,
 #       batch_size = 256,
 #       patience = 2
 #   ) -> float:
#
 #   """
 #   - Download processed data from your BQ table (or from cache if it exists)
 #   - Train on the preprocessed dataset (which should be ordered by date)
 #   - Store training results and model weights
#
 #   Return val_mae as a float
 #   """







#def evaluate(
#        min_date:str = '2014-01-01',
 #       max_date:str = '2015-01-01',
#        stage: str = "Production"
#    ) -> float:
#    """
 #   Evaluate the performance of the latest production model on processed data
#    Return MAE as a float
 #   """




#def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
#"""
#    Make a prediction using the latest trained model
#    """



# if __name__=='__main__':
#   preprocess()
#   train()
#   evaluate()
#   pred()
