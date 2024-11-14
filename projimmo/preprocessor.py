"""
projimmo/preprocessor.py


Prétraitement des données pour le projet Immo.

Ce module contient des fonctions et des pipelines pour appliquer des transformations aux caractéristiques
des données d'entrée avant d'entraîner un modèle. Il inclut des étapes de prétraitement spécifiques pour
les variables numériques et catégorielles, telles que la standardisation, la normalisation, et l'encodage.

Fonctionnalités principales :
- Prétraitement des colonnes numériques : standardisation et mise à l'échelle robuste pour les colonnes
  contenant des valeurs numériques.
- Prétraitement des colonnes catégorielles : encodage one-hot pour les variables catégorielles et
  encodage ordinal pour certaines colonnes spécifiques.
- Construction d'un pipeline de transformation des données via `ColumnTransformer` de scikit-learn pour
  appliquer ces transformations de manière modulaire et cohérente.
- Retourne un DataFrame transformé avec les caractéristiques prétraitées prêtes à être utilisées pour
  l'entraînement d'un modèle.

Fonctions :
- `preprocess_features(X: pd.DataFrame) -> pd.DataFrame` : Fonction principale pour prétraiter les
  caractéristiques d'entrée. Applique une série de transformations aux colonnes numériques et catégorielles
  et retourne le DataFrame transformé.
"""


import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, OrdinalEncoder
from colorama import Fore, Style


def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Applique un prétraitement aux caractéristiques d'entrée (X).

    Cette fonction utilise un pipeline pour transformer les colonnes numériques,
    catégorielles et ordinales de manière appropriée. Elle applique les transformations suivantes :
    - Normalisation pour les colonnes numériques
    - Standardisation et mise à l'échelle robuste pour certaines colonnes numériques
    - Encodage one-hot pour les colonnes catégorielles
    - Encodage ordinal pour une colonne spécifique

    Args:
        X (pd.DataFrame): Le DataFrame contenant les caractéristiques à prétraiter.

    Returns:
        X_trans_df (pd.DataFrame): Un DataFrame avec les caractéristiques transformées.
        preproc_baseline (ColumnTransformer): L'objet de transformation des colonnes utilisé pour effectuer les transformations.
    """

    # Définir les pipelines de prétraitement pour les différentes transformations
    preproc_robust = make_pipeline(RobustScaler())  # Mise à l'échelle robuste pour 'surface_terrain' et 'somme_surface_carrez'
    preproc_standard = make_pipeline(StandardScaler())  # Standardisation pour 'surface_reelle_bati'
    preproc_categorical_baseline = make_pipeline(OneHotEncoder(handle_unknown="ignore"))  # Encodage one-hot pour les variables catégorielles
    preproc_ordinal = make_pipeline(OrdinalEncoder())  # Encodage ordinal pour 'nombre_pieces_principales'

    # Définir la transformation des colonnes avec les différents préprocesseurs
    preproc_baseline = make_column_transformer(
        (preproc_standard, ['surface_reelle_bati']),  # Applique StandardScaler sur 'surface_reelle_bati'
        (preproc_robust, ['surface_terrain', 'somme_surface_carrez']),  # Applique RobustScaler sur 'surface_terrain' et 'somme_surface_carrez'
        (preproc_categorical_baseline, ['code_type_local', 'type_de_voie', 'month_mutation', 'year_mutation', 'departement']),  # OneHotEncoder sur les variables catégorielles
        (preproc_ordinal, ['nombre_pieces_principales']),  # OrdinalEncoder sur 'nombre_pieces_principales'
        remainder='passthrough'  # Les autres colonnes non spécifiées sont laissées inchangées
    )

    # Appliquer la transformation sur les données d'entrée X
    X_trans = preproc_baseline.fit_transform(X)

    # Fonction interne pour obtenir les noms des colonnes après transformation
    def get_feature_names(column_transformer, original_columns):
        """
        Récupère les noms des colonnes après transformation.

        Args:
            column_transformer (ColumnTransformer): Le transformer de colonnes appliqué sur les données.
            original_columns (Index): Les colonnes originales avant transformation.

        Returns:
            feature_names (list): La liste des noms des colonnes après transformation.
        """
        feature_names = []
        for name, transformer, columns in column_transformer.transformers_:
            if hasattr(transformer, 'get_feature_names_out'):
                # Récupérer les nouveaux noms des colonnes pour les transformeurs qui le supportent
                feature_names.extend(transformer.get_feature_names_out())
            else:
                # Si le transformer n'a pas de fonction 'get_feature_names_out', conserver les noms d'origine des colonnes
                feature_names.extend(columns)
        return feature_names

    # Obtenir les noms des nouvelles colonnes après transformation
    feature_names = get_feature_names(preproc_baseline, X.columns)

    # Si le résultat de la transformation est une matrice creuse, la convertir en dense pour créer un DataFrame
    if hasattr(X_trans, "toarray"):
        X_trans_dense = X_trans.toarray()
    else:
        X_trans_dense = X_trans

    # Renommer la dernière colonne en "nombre_de_lots" si des colonnes ont été transformées
    if len(feature_names) > 0:
        feature_names[-1] = "nombre_de_lots"

    # Créer un DataFrame avec les nouvelles colonnes transformées
    X_trans_df = pd.DataFrame(X_trans_dense, columns=feature_names)

    # Afficher la forme des données transformées
    print(f"✅ X_processed, with shape {X_trans_df.shape}")

    # Assigner un nom spécifique à la première colonne (valeur_fonciere) si nécessaire
    X_trans_df.columns = ['valeur_fonciere' if col == 0 else col for col in X_trans_df.columns]

    # Retourner le DataFrame transformé et le préprocesseur utilisé
    return X_trans_df, preproc_baseline
