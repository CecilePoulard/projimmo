#projimmo/preprocessor.py

#module pour préprocesser les datas (i.e. pipeline)

import numpy as np
import pandas as pd



from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer


#from colorama import Fore, Style
#def preprocess_features(X: pd.DataFrame) -> np.ndarray:
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

 #   print("✅ X_processed, with shape", X_processed.shape)

  #  return X_processed
