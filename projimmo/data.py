# projimmo/data.py

import numpy as np
import pandas as pd

from google.cloud import bigquery
from pathlib import Path
from colorama import Fore, Style


from projimmo.params import *


#def clean_data(df: pd.DataFrame) -> pd.DataFrame:
 #   """
#    Clean raw data by
#    - assigning correct dtypes to each column
#    - removing buggy or irrelevant transactions
#    """

#def concat
# """
# concatene les df clean et preprocess pour chaque année

#recupere fichier
def get_data_with_cache(
        gcp_project:str,
        query:str,
        cache_path:Path,
        data_has_header=True
   ) -> pd.DataFrame:
    """
    Retrieve `query` data from BigQuery, or from `cache_path` if the file exists
    Store at `cache_path` if retrieved from BigQuery for future use
    """

    if cache_path.is_file():
        print(Fore.BLUE + "\nLoad data from local CSV..." + Style.RESET_ALL)
        df = pd.read_csv(cache_path, header='infer' if data_has_header else None)
    else:
        print(Fore.BLUE + "\nLoad data from BigQuery server..." + Style.RESET_ALL)
        client = bigquery.Client(project=gcp_project)
        query_job = client.query(query)
        result = query_job.result()
        df = result.to_dataframe()
        # Store as CSV if the BQ query returned at least one valid line
        if df.shape[0] > 1:
            df.to_csv(cache_path, header=data_has_header, index=False)

    print(f"✅ Data loaded, with shape {df.shape}")

    return df


#charger les datas sur bigquery
def load_data_to_bq(
        data: pd.DataFrame,
        gcp_project:str,
        bq_dataset:str,
        table: str,
        truncate: bool
    ) -> None:
    """
    - Save the DataFrame to BigQuery
    - Empty the table beforehand if `truncate` is True, append otherwise
    """
#truncate : Un booléen qui indique si la table doit être vidée avant de charger
# les nouvelles données. Si True, les anciennes données sont supprimées. Si False,
# les nouvelles données sont ajoutées à la suite des existantes.

#test si les data d'entree est bien sous forme de DF
    assert isinstance(data, pd.DataFrame)
#construction du non complet de la table bigquery
    full_table_name = f"{gcp_project}.{bq_dataset}.{table}"
    print(Fore.BLUE + f"\nSave data to BigQuery @ {full_table_name}...:" + Style.RESET_ALL)


#Nettoyage des noms de colonnes.
#BigQuery impose des regles strictes sur le nom des colonnes
#doivent commencer par '_' ou une lettre et nom un chiffre ou un
# symbole
    data.columns = [f"_{column}" if not str(column)[0].isalpha() and not str(column)[0] == "_" else str(column) for column in data.columns]
#initialisation de bigquery
    client = bigquery.Client()

#Cette partie configure la manière dont les données seront écrites dans la table BigQuery
    write_mode = "WRITE_TRUNCATE" if truncate else "WRITE_APPEND"
    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

    print(f"\n{'Write' if truncate else 'Append'} {full_table_name} ({data.shape[0]} rows)")

# charge les données dans bigquery
    job = client.load_table_from_dataframe(data, full_table_name, job_config=job_config)
#Cette méthode attend que le chargement soit terminé avant de continuer.
    result = job.result()

    print(f"✅ Data saved to bigquery, with shape {data.shape}")
