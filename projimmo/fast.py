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

# Configuration du middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permet toutes les origines
    allow_credentials=True,
    allow_methods=["*"],  # Permet toutes les méthodes
    allow_headers=["*"],  # Permet tous les headers
)

# Chargement du modèle au démarrage du serveur
app.state.model = load_model()

# Chargement du préprocesseur pour transformation des données
app.state.preproc_baseline = joblib.load('preproc_baseline.pkl')

@app.get("/predict")
def predict(

    type_de_voie: str,  # ['RUE', 'AV', 'CHE', 'BD', 'RTE', 'ALL', 'IMP', 'PL', 'RES', 'CRS','AUTRE']
    nombre_de_lots: int,
    code_type_local: int,  # {'maison': 1, 'appartement': 2}
    surface_reelle_bati: float ,
    nombre_pieces_principales: float,
    surface_terrain: float,
    month_mutation: int ,
    year_mutation: int ,
    somme_surface_carrez: float ,
    departement: int # ['75', '13', '69', '31', '06', '44', '34', '67', '33', '59']
    ,valeur_fonciere: float=0.
):
    # Création du DataFrame à partir des données fournies
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

    # Transformation des données
    preproc_baseline = app.state.preproc_baseline
    X_processed = preproc_baseline.transform(X_pred)
    #print(X_processed.columns)
    # Si la matrice est creuse, la convertir en dense
    if hasattr(X_processed, 'toarray'):
        X_processed = X_processed.toarray()
    elif hasattr(X_processed, 'todense'):
        X_processed = X_processed.todense()
    # Obtenir les noms de colonnes après transformation
    if hasattr(preproc_baseline, 'get_feature_names_out'):
        feature_names = preproc_baseline.get_feature_names_out()
    else:
        feature_names = ["Feature_" + str(i) for i in range(X_processed.shape[1])]

    # Convertir en DataFrame pour afficher les colonnes
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

# Fonction pour nettoyer les noms de colonnes
    def clean_column_names(columns):
        cleaned_columns = []
        for col in columns:
            # Supprimer les préfixes
            col = re.sub(r'^(pipeline-\d+__|remainder__)', '', col)
            # Supprimer le suffixe '.0'
            col = re.sub(r'\.0$', '_0', col)
            cleaned_columns.append(col)
        return cleaned_columns

# Nettoyer les noms de colonnes
    cleaned_columns = clean_column_names(X_processed_df.columns)
    # Prédiction avec le modèle
   #
   #
    # Renommer les colonnes de X_processed_df
    X_processed_df.columns = cleaned_columns
    X_processed_df.drop(columns='valeur_fonciere',inplace=True)
    model = app.state.model
    assert model is not None
    y_pred = model.predict(X_processed_df)
    # Retourner la prédiction sous forme de dictionnaire
    return {
        'prediction': float(np.expm1(y_pred[0])),
        #'columns': X_processed_df.columns.tolist()
    }

@app.get("/")
def root():
    return {"greeting": "Hello"}
