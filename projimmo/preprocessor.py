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
#    def create_sklearn_preprocessor() -> ColumnTransformer:
 #       """
 #       Scikit-learn pipeline that transforms a cleaned dataset of shape (_, 7)
 #       into a preprocessed one of fixed shape (_, 65).
#
 #       Stateless operation: "fit_transform()" equals "transform()".
 #      """

 # ..............

 #    preprocessor = create_sklearn_preprocessor()
 #   X_processed = preprocessor.fit_transform(X)

 #

  #  return X_processed
  # Définir les pipelines de prétraitement
    preproc_robust = make_pipeline(RobustScaler())
    preproc_standard = make_pipeline(StandardScaler())
    preproc_categorical_baseline = make_pipeline(OneHotEncoder(handle_unknown="ignore"))
##################################### A modifier!!!!!!!! ################
    preproc_ordinal = make_pipeline(OrdinalEncoder())

    # Définir le transformer de colonnes
    preproc_baseline = make_column_transformer(
        (preproc_standard, ['surface_reelle_bati']),
        (preproc_robust, ['surface_terrain', 'somme_surface_carrez','valeur_fonciere']),
        (preproc_categorical_baseline, ['code_type_local', 'type_de_voie', 'departement']),
        (preproc_ordinal, ['nombre_pieces_principales', 'month_mutation', 'year_mutation']),
        remainder='passthrough')
    # Transformer les données
    X_trans = preproc_baseline.fit_transform(X)

    # Fonction pour obtenir les noms des colonnes
    def get_feature_names(column_transformer):
        feature_names = []
        for name, transformer, columns in column_transformer.transformers_:
            if hasattr(transformer, 'get_feature_names_out'):
                # Obtenir les noms de fonctionnalités pour chaque transformeur
                feature_names.extend(transformer.get_feature_names_out())
            else:
                feature_names.extend(columns)
        return feature_names

    # Obtenir les noms des colonnes
    feature_names = get_feature_names(preproc_baseline)

    # Convertir en DataFrame si la sortie est une matrice creuse
    if hasattr(X_trans, "toarray"):
        X_trans_dense = X_trans.toarray()
    else:
        X_trans_dense = X_trans

    # Créer la DataFrame avec les noms de colonnes
    X_trans_df = pd.DataFrame(X_trans_dense, columns=feature_names)
    print("✅ X_processed, with shape", X_trans_df.shape)

    return X_trans_df
