"""
fast.py



Ce fichier contient l'implémentation d'une API RESTful utilisant FastAPI pour prédire la valeur foncière d'une propriété immobilière.

Les prédictions sont réalisées à partir des informations sur le bien immobilier fournies par l'utilisateur, notamment le type de voie, le nombre de lots, la surface du terrain, la surface réelle du bâtiment, et d'autres caractéristiques spécifiques à la mutation d'une propriété.

Les principales fonctionnalités de ce fichier incluent :
- L'initialisation de l'application FastAPI et la configuration du middleware CORS pour permettre les requêtes depuis toutes les origines.
- Le chargement d'un modèle pré-entraîné et d'un préprocesseur de données au démarrage de l'application pour effectuer les prédictions.
- Un endpoint `/predict` qui prend en entrée les caractéristiques d'un bien immobilier et renvoie une prédiction de la valeur foncière basée sur un modèle machine learning.
- La gestion des données d'entrée via un prétraitement (transformation et nettoyage des colonnes), suivi d'une prédiction avec un modèle préchargé.
- Un endpoint racine `/` pour vérifier que le serveur fonctionne.

Modules utilisés :
- `fastapi`: Framework pour créer l'API RESTful.
- `pandas`: Pour manipuler les données sous forme de DataFrame.
- `re`: Pour le nettoyage des noms de colonnes.
- `joblib`: Pour charger des objets pré-enregistrés tels que le préprocesseur et le modèle.
- `projimmo`: Un module contenant des fonctions et des classes spécifiques pour les paramètres, les données, le prétraitement, et l'optimisation du modèle.

L'API est conçue pour être utilisée par des clients front-end ou d'autres services backend pour intégrer des prédictions de valeur foncière dans leurs processus.

Les entrées à l'API doivent inclure des informations sur la propriété, telles que le type de voie, le nombre de lots, la surface réelle du bâtiment, et d'autres données spécifiques à la mutation. Ces données sont transformées avant de passer dans le modèle qui génère une estimation de la valeur foncière de la propriété.

"""








import pandas as pd
from fastapi import FastAPI
import re
from fastapi.middleware.cors import CORSMiddleware
import joblib
from projimmo.params import *
from projimmo.data import *
from projimmo.preprocessor import *
from projimmo.model import *
from projimmo.registry import *
from projimmo.optimize_model import *

# Initialisation de l'application FastAPI
app = FastAPI()

# Configuration du middleware CORS pour autoriser toutes les origines, méthodes et headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permet toutes les origines
    allow_credentials=True,
    allow_methods=["*"],  # Permet toutes les méthodes HTTP
    allow_headers=["*"],  # Permet tous les headers
)

# Chargement du modèle au démarrage du serveur
app.state.model = load_model()

# Chargement du préprocesseur pour la transformation des données
app.state.preproc_baseline = joblib.load('preproc_baseline.pkl')

@app.get("/predict")
def predict(
    type_de_voie: str,  # Type de voie (ex. RUE, AV, CHE, etc.)
    nombre_de_lots: int,  # Nombre de lots
    code_type_local: int,  # Type de bien : maison (1) ou appartement (2)
    surface_reelle_bati: float,  # Surface réelle du bâtiment
    nombre_pieces_principales: float,  # Nombre de pièces principales
    month_mutation: int,  # Mois de la mutation
    year_mutation: int,  # Année de la mutation
    somme_surface_carrez: float,  # Somme des surfaces Carrez des lots
    departement: int,  # Code du département (ex. 75 pour Paris)
    surface_terrain: float = 0.0,  # Surface du terrain
    valeur_fonciere: float = 0.0  # Valeur foncière de la transaction
):
    """
    Prédire la valeur foncière à partir des informations immobilières fournies.

    Args:
        type_de_voie (str): Type de voie (ex. 'RUE', 'AV', etc.).
        nombre_de_lots (int): Nombre de lots.
        code_type_local (int): Type de bien, 'maison' ou 'appartement'.
        surface_reelle_bati (float): Surface réelle du bâtiment.
        nombre_pieces_principales (float): Nombre de pièces principales.
        month_mutation (int): Mois de la mutation.
        year_mutation (int): Année de la mutation.
        somme_surface_carrez (float): Somme des surfaces Carrez des lots.
        departement (int): Code du département (ex. 75 pour Paris).
        surface_terrain (float, optional): Surface du terrain, par défaut 0.0.
        valeur_fonciere (float, optional): Valeur foncière de la transaction, par défaut 0.0.

    Returns:
        dict: Dictionnaire contenant la prédiction de la valeur foncière.
    """

    # Création d'un DataFrame à partir des données fournies
    X_pred = pd.DataFrame({
        'valeur_fonciere': [valeur_fonciere],
        'type_de_voie': [type_de_voie],
        'nombre_de_lots': [nombre_de_lots],
        'code_type_local': [code_type_local],
        'surface_reelle_bati': [surface_reelle_bati],
        'nombre_pieces_principales': [nombre_pieces_principales],
        'surface_terrain': [surface_terrain],
        'month_mutation': [month_mutation],
        'year_mutation': [year_mutation],
        'somme_surface_carrez': [somme_surface_carrez],
        'departement': [departement]
    })

    # Transformation des données avec le préprocesseur chargé
    preproc_baseline = app.state.preproc_baseline
    X_processed = preproc_baseline.transform(X_pred)

    # Si la matrice est creuse, la convertir en dense pour la prédiction
    if hasattr(X_processed, 'toarray'):
        X_processed = X_processed.toarray()
    elif hasattr(X_processed, 'todense'):
        X_processed = X_processed.todense()

    # Obtenir les noms des features après transformation
    if hasattr(preproc_baseline, 'get_feature_names_out'):
        feature_names = preproc_baseline.get_feature_names_out()
    else:
        feature_names = ["Feature_" + str(i) for i in range(X_processed.shape[1])]

    # Convertir en DataFrame pour afficher les colonnes après transformation
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

    # Fonction pour nettoyer les noms de colonnes
    def clean_column_names(columns):
        """
        Nettoie les noms de colonnes en supprimant certains préfixes et suffixes inutiles.

        Args:
            columns (Index): Liste des noms de colonnes.

        Returns:
            List: Liste des noms de colonnes nettoyés.
        """
        cleaned_columns = []
        for col in columns:
            # Supprimer les préfixes 'pipeline-<num>__' ou 'remainder__'
            col = re.sub(r'^(pipeline-\d+__|remainder__)', '', col)
            # Supprimer le suffixe '.0' en le remplaçant par '_0'
            col = re.sub(r'\.0$', '_0', col)
            cleaned_columns.append(col)
        return cleaned_columns

    # Appliquer le nettoyage des noms de colonnes
    cleaned_columns = clean_column_names(X_processed_df.columns)

    # Renommer les colonnes du DataFrame transformé
    X_processed_df.columns = cleaned_columns
    # Supprimer la colonne 'valeur_fonciere' après transformation
    X_processed_df.drop(columns='valeur_fonciere', inplace=True)

    # Vérifier que le modèle est chargé
    model = app.state.model
    assert model is not None

    # Prédire avec le modèle chargé
    y_pred = model.predict(X_processed_df)

    # Retourner la prédiction sous forme de dictionnaire
    return {
        'prediction': float(np.expm1(y_pred[0]))  # Appliquer la transformation inverse de log sur la prédiction
    }

@app.get("/")
def root():
    """
    Endpoint racine pour vérifier que l'application fonctionne.

    Returns:
        dict: Un message de salutation.
    """
    return {"greeting": "Hello"}
