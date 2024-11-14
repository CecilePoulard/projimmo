"""
 projimmo/data.py



Ce module contient des fonctions pour le nettoyage, la transformation et le traitement de données immobilières,
en particulier celles provenant de la base de données BigQuery. Les étapes incluent la préparation des données,
la suppression des valeurs aberrantes (outliers), et le chargement des données dans BigQuery. Il est conçu pour
faciliter l'intégration, la gestion et l'analyse des données brutes en vue de leur utilisation dans des analyses
immobilières.

Les principales fonctions du module sont les suivantes :
1. `clean_data`: Nettoie les données brutes en attribuant les bons types de données, en filtrant les lignes
   pertinentes, en créant de nouvelles colonnes et en supprimant les doublons.
2. `clean_outliers`: Supprime les valeurs aberrantes basées sur le Z-score de la colonne 'valeur_fonciere'.
3. `get_data_with_cache`: Récupère les données depuis BigQuery ou depuis un fichier CSV local en utilisant un cache
   pour éviter des requêtes redondantes.
4. `load_data_to_bq`: Charge un DataFrame dans BigQuery, soit en vidant la table existante, soit en ajoutant les
   nouvelles données à la table.

Les données traitées sont destinées à être utilisées dans des analyses liées au marché immobilier,
avec des informations sur des transactions immobilières spécifiques telles que la valeur foncière, les surfaces
des lots, et des informations géographiques.

Chaque fonction est dotée de sa propre logique métier pour répondre à des besoins spécifiques du nettoyage de
données et du traitement d'analyses sur des données immobilières.

Modules externes utilisés :
- `numpy`: Pour des transformations numériques.
- `pandas`: Pour la manipulation des DataFrames.
- `scipy.stats`: Pour le calcul du Z-score afin de détecter les outliers.
- `google.cloud.bigquery`: Pour interagir avec BigQuery.
- `colorama`: Pour améliorer l'affichage en couleur dans la console.
- `pathlib`: Pour la gestion des chemins de fichiers.

"""



import numpy as np
import pandas as pd
from scipy import stats
from google.cloud import bigquery
from pathlib import Path
from colorama import Fore, Style
from google.api_core.exceptions import NotFound

from projimmo.params import *

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les données brutes en effectuant les étapes suivantes:
    - Attribution des bons types de données pour chaque colonne
    - Suppression des transactions erronées ou non pertinentes
    - Création de nouvelles colonnes pour les mois et années des mutations
    - Conversion des colonnes de surface en numérique
    - Nettoyage des noms de colonnes et suppression des doublons

    Args:
        df (pd.DataFrame): DataFrame contenant les données brutes à nettoyer.

    Returns:
        pd.DataFrame: DataFrame nettoyée avec les bonnes colonnes et types de données.
    """

    # Sélectionner uniquement les colonnes nécessaires
    df = df[COLUMN_NAMES_RAW]

    # Filtrer les données pour ne garder que celles des départements des 10 plus grandes villes
    df = df[df['Code postal'].astype(str).str.startswith(tuple(DEPARTEMENTS))]

    # Filtrer pour ne garder que les transactions 'Vente' et certains types de biens
    df = df[df['Nature mutation'] == 'Vente']
    df = df[(df['Code type local'] == '1') | (df['Code type local'] == '2')]

    # Ajouter les colonnes 'Month mutation' et 'Year mutation'
    df['Date mutation'] = pd.to_datetime(df['Date mutation'], format='%d/%m/%Y')
    df['Month mutation'] = df['Date mutation'].dt.month
    df['Year mutation'] = df['Date mutation'].dt.year

    # Colonnes Carrez à traiter (surface des lots)
    carrez_cols = ['Surface Carrez du 1er lot', 'Surface Carrez du 2eme lot',
                   'Surface Carrez du 3eme lot', 'Surface Carrez du 4eme lot',
                   'Surface Carrez du 5eme lot']

    # Remplacer les virgules par des points pour la conversion en valeurs numériques
    df['Valeur fonciere'] = df['Valeur fonciere'].str.replace(',', '.')
    df[carrez_cols] = df[carrez_cols].apply(lambda col: col.str.replace(',', '.'))

    # Remplacer les NaN dans les colonnes de surface Carrez par 0
    df[carrez_cols] = df[carrez_cols].fillna(0)

    # Remplir les NaN dans d'autres colonnes avec des 0
    df[['Surface reelle bati', 'Nombre pieces principales', 'Surface terrain',
        'Nombre de lots', 'Code type local']] = df[['Surface reelle bati',
                                                    'Nombre pieces principales',
                                                    'Surface terrain',
                                                    'Nombre de lots',
                                                    'Code type local']].fillna(0)

    # Convertir les colonnes pertinentes en numérique
    df[carrez_cols] = df[carrez_cols].apply(pd.to_numeric, errors='coerce')
    df['Valeur fonciere'] = pd.to_numeric(df['Valeur fonciere'], errors='coerce')

    # Calculer la somme des surfaces Carrez pour tous les lots
    df['somme surface carrez'] = df[carrez_cols].sum(axis=1)

    # Supprimer les colonnes de surface Carrez et les colonnes inutiles
    df.drop(columns=carrez_cols, inplace=True)
    df.drop(columns=['Date mutation', 'Nature mutation'], inplace=True)

    # Supprimer les lignes où il manque la 'Valeur fonciere' ou le 'Code postal'
    df.dropna(subset=['Valeur fonciere', 'Code postal'], inplace=True)

    # Filtrer les lignes où certaines surfaces ou quantités sont égales à 0
    df = df[(df['Nombre de lots'] != 0) &
            (df['Surface reelle bati'] != 0) &
            (df['somme surface carrez'] != 0)]

    # Nettoyage du type de voie (ex. 'RUE', 'AV', 'CHE', etc.)
    type_voie_to_keep = ['RUE', 'AV', 'CHE', 'BD', 'RTE', 'ALL', 'IMP', 'PL', 'RES', 'CRS']
    df['Type de voie'] = df['Type de voie'].where(df['Type de voie'].isin(type_voie_to_keep), 'AUTRE')

    # Appliquer une transformation log sur la 'Valeur fonciere' pour réduire l'impact des valeurs extrêmes
    df['Valeur fonciere'] = np.log1p(df['Valeur fonciere'])

    # Fonction pour nettoyer les noms des colonnes (mettre en minuscules et remplacer les espaces par des underscores)
    def clean_column_names(df):
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        return df

    df = clean_column_names(df)

    # Ajouter une colonne 'departement' en extrayant les deux premiers chiffres du code postal
    df['departement'] = df['code_postal'].str[:2]
    df = df.drop(columns=['code_postal'])

    # Supprimer les doublons
    df_cleaned = df.drop_duplicates()

    # Appliquer les types de données définis dans DTYPES_RAW
    df_cleaned = df_cleaned.astype(DTYPES_RAW)

    return df_cleaned

def clean_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les valeurs aberrantes (outliers) basées sur le Z-score de la colonne 'valeur_fonciere'.
    Les lignes ayant un Z-score supérieur à 3 ou inférieur à -3 sont considérées comme des outliers et sont supprimées.

    Args:
        df (pd.DataFrame): DataFrame contenant les données à traiter.

    Returns:
        pd.DataFrame: DataFrame sans les valeurs aberrantes.
    """

    df = df.copy()

    # Calcul du Z-score pour la colonne 'valeur_fonciere'
    df['z_score'] = stats.zscore(df['valeur_fonciere'])

    # Filtrer les outliers en fonction du Z-score
    df = df[df['z_score'].abs() <= 3]

    # Supprimer la colonne 'z_score' après traitement
    df = df.drop(['z_score'], axis=1)

    return df

def get_data_with_cache(
        gcp_project: str,
        query: str,
        cache_path: Path,
        data_has_header: bool = True
   ) -> pd.DataFrame:
    """
    Récupère les données depuis BigQuery ou un fichier local, selon l'existence du cache.
    Si les données sont récupérées depuis BigQuery, elles sont ensuite stockées dans un fichier CSV pour un usage futur.

    Args:
        gcp_project (str): Le projet Google Cloud Platform à utiliser pour la requête BigQuery.
        query (str): La requête SQL à exécuter sur BigQuery.
        cache_path (Path): Le chemin où les données sont stockées en cache (fichier CSV).
        data_has_header (bool): Indique si le fichier CSV a une ligne d'entête. Par défaut True.

    Returns:
        pd.DataFrame: Les données récupérées sous forme de DataFrame.
    """

    if cache_path.is_file():
        print(Fore.BLUE + "\nChargement des données depuis le fichier local CSV..." + Style.RESET_ALL)
        df = pd.read_csv(cache_path, sep="|", dtype=str)
    else:
        print(Fore.BLUE + "\nChargement des données depuis BigQuery..." + Style.RESET_ALL)
        client = bigquery.Client(project=gcp_project)
        query_job = client.query(query)
        result = query_job.result()
        df = result.to_dataframe()
        # Sauvegarder les données dans un fichier CSV si elles ont été récupérées depuis BigQuery
        # df.to_csv(cache_path, header=data_has_header, index=False)

    print(f"✅ Données chargées, shape: {df.shape}")

    return df

def load_data_to_bq(
        data: pd.DataFrame,
        gcp_project: str,
        bq_dataset: str,
        table: str,
        truncate: bool
    ) -> None:
    """
    Charge un DataFrame dans BigQuery, en vidant la table si 'truncate' est True ou en ajoutant les nouvelles données sinon.

    Args:
        data (pd.DataFrame): Le DataFrame à charger dans BigQuery.
        gcp_project (str): Le projet Google Cloud Platform.
        bq_dataset (str): Le dataset BigQuery.
        table (str): Le nom de la table BigQuery.
        truncate (bool): Si True, la table est vidée avant d'insérer de nouvelles données, sinon on ajoute les nouvelles données.

    Returns:
        None
    """

    # Vérifier que l'argument data est bien un DataFrame
    assert isinstance(data, pd.DataFrame)

    # Nom complet de la table BigQuery
    full_table_name = f"{gcp_project}.{bq_dataset}.{table}"
    print(Fore.BLUE + f"\nSauvegarde des
