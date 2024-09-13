#main.py

import os
import pandas as pd
from google.cloud import bigquery
from pathlib import Path

from projimmo.params import *
from projimmo.data import *

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
    #all_files = [f for f in data_local_path if f.is_file()]
    all_files = list(data_local_path.glob('*.txt'))

    if not all_files:
        print("Aucun fichier CSV trouvé dans le répertoire spécifié.")
        return

    for file in all_files:
        df = pd.read_csv(file, sep="|", dtype=str)
        print(file)
        clean_and_load(df)
    return table


def clean_and_load(df, table):
    """
    Nettoie le df en param et la charge dans bigQuery
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

def preprocess():
    table_cleaned=load_all_files()
    query = f"""
        SELECT *
        FROM `{GCP_PROJECT}`.{BQ_DATASET}.{table_cleaned}
    """
    #Pour écrire dans big query le df notre choix
#load_data_to_bq(
#        df_de_notre_choix,
 #       gcp_project=GCP_PROJECT,
 #       bq_dataset=BQ_DATASET,
 #       table=table_cleaned',
 #       truncate=True
  #  )
    df_to_preproc=get_data_with_cache(
        query=query,
        gcp_project=GCP_PROJECT,
       # cache_path:Path,
        data_has_header=True
   )


if __name__ == "__main__":
#    concat_and_load() # A faire qu'une fois!! Sinon va ajouter















#def preprocess(min_date:str = '2009-01-01', max_date:str = '2015-01-01') -> None:
 #   """
 #   - Query the raw dataset from Le Wagon's BigQuery dataset
 #   - Cache query result as a local CSV if it doesn't exist locally
#    - Process query data
#    - Store processed data on your personal BQ (truncate existing table if it exists)
 #   - No need to cache processed data as CSV (it will be cached when queried back from BQ during training)
#    """










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
