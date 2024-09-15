#projimmo/preprocessor.py

#module pour préprocesser les datas (i.e. pipeline)

import numpy as np
import pandas as pd



from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler,OrdinalEncoder
from colorama import Fore, Style


def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
  # Définir les pipelines de prétraitement
    preproc_robust = make_pipeline(RobustScaler())
    preproc_standard = make_pipeline(StandardScaler())
    preproc_categorical_baseline = make_pipeline(OneHotEncoder(handle_unknown="ignore"))
    preproc_ordinal = make_pipeline(OrdinalEncoder())

    # Définir le transformer de colonnes
    preproc_baseline = make_column_transformer(
        (preproc_standard, ['surface_reelle_bati']),
        (preproc_robust, ['surface_terrain', 'somme_surface_carrez']),#,'valeur_fonciere'
        (preproc_categorical_baseline, ['code_type_local', 'type_de_voie', 'month_mutation', 'year_mutation','departement']),
        (preproc_ordinal, ['nombre_pieces_principales']),
        remainder='passthrough')
    # Transformer les données
    X_trans = preproc_baseline.fit_transform(X)

    # Fonction pour obtenir les noms des colonnes
    def get_feature_names(column_transformer, original_columns):
        feature_names = []
        for name, transformer, columns in column_transformer.transformers_:
            if hasattr(transformer, 'get_feature_names_out'):
                # Obtenir les noms de fonctionnalités pour chaque transformeur
                feature_names.extend(transformer.get_feature_names_out())
            else:
                feature_names.extend(columns)
        # Ajouter les noms des colonnes non transformées

        return feature_names

    # Obtenir les noms des colonnes
    feature_names = get_feature_names(preproc_baseline,X.columns)

    # Convertir en DataFrame si la sortie est une matrice creuse
    if hasattr(X_trans, "toarray"):
        X_trans_dense = X_trans.toarray()
    else:
        X_trans_dense = X_trans
    # Renommer la dernière colonne en "nombre_de_lots"
    if len(feature_names) > 0:
        feature_names[-1] = "nombre_de_lots"
    # Créer la DataFrame avec les noms de colonnes
    X_trans_df = pd.DataFrame(X_trans_dense, columns=feature_names)
    print("✅ X_processed, with shape", X_trans_df.shape)
    X_trans_df.columns = ['valeur_fonciere' if col == 0 else col for col in X_trans_df.columns]

    return X_trans_df
