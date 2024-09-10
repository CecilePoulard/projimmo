#module pour préprocesser les datas

import numpy as np
import pandas as pd



from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer


#from colorama import Fore, Style
