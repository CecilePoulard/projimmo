#projimmo/model.py

import xgboost as xgb  # Pour utiliser XGBoost
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error
from sklearn.metrics import accuracy_score  # Pour les tâches de classification

import numpy as np

def split_df(df,test_size=0.2):
    """
    Split dataset into train_test_split
    - possibilité de changer le test_size (par défaut 0.2)
    """
# Split the data into training and testing sets
    X=df.drop(columns='valeur_fonciere').copy()
    y=df.valeur_fonciere.copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


def initialize_model(model_type,n_neighbors=5,objective='reg:squarederror', max_depth=4, eta=0.1, n_estimators=100):
    """
    Initialise le model selon le choix de l'utilisateur
    - nombre de cluster pour le KNR: n_neighbors=5
    - parametres par défaut pour le XGboost objective='reg:squarederror', max_depth=4, eta=0.1, n_estimators=100
    """
    if model_type=="KNR":
        model = KNeighborsRegressor(n_neighbors=n_neighbors)
    elif model_type=="LR":
        model = LinearRegression()
    elif model_type == "XGB":
        model = xgb.XGBRegressor(objective=objective, max_depth=max_depth, eta=eta, n_estimators=n_estimators)  # XGBoost

    return model


def train_model(X_train,y_train,model):
    """
    entraine le modele
    """
    fitted_model=model.fit(X_train, y_train)
    return fitted_model




def evaluate_model(X_test,y_test,model, model_type):
    """
    Evalue le modele selon le type de modele
    retourne différentes metrics
    """
    # Make predictions on the testing data
    y_pred = model.predict(X_test)
    if model_type in ["KNR", "LR", "XGB"]:

    # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
    # Vérifier si toutes les prédictions sont positives pour calculer MSLE
        if np.all(y_pred > 0) and np.all(y_test > 0):
            msle = mean_squared_log_error(y_test, y_pred)
        else:
            msle=None

    # Print the evaluation metrics
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared (R²) Score: {r2}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Squared Logarithmic Error (MSLE): {msle}")
            # Retourner les métriques pour les réutiliser
    return {'mse': mse, 'mae': mae, 'r2': r2, 'rmse': rmse, 'msle': msle}





#1. Boosting en gradient
#XGBoost appartient à la famille des algorithmes de boosting, une méthode d'ensemble qui combine plusieurs modèles faibles (généralement des arbres de décision peu profonds) pour créer un modèle puissant. Le boosting fonctionne de manière séquentielle :

#Chaque modèle est construit en tentant de corriger les erreurs faites par les modèles précédents.
#XGBoost utilise le gradient des erreurs pour ajuster les nouveaux modèles, en minimisant progressivement les résidus (les différences entre les prédictions et les valeurs réelles).
#2. Arbres de décision
#Le modèle de base de XGBoost est généralement un arbre de décision (tree-based learners). Cependant, il s'agit souvent d'arbres peu profonds, ce qui permet d'éviter le surapprentissage (overfitting) tout en conservant une certaine puissance de prédiction.

#3. Optimisation avancée
#XGBoost est optimisé à plusieurs niveaux, ce qui en fait un algorithme très performant :

#Régularisation : L'ajout de termes de régularisation (L1 et L2) aide à éviter le surapprentissage.
#Pruning : XGBoost effectue un élagage intelligent des branches des arbres pour éviter des modèles inutiles.
#Handling des données manquantes : Il gère automatiquement les données manquantes, en les traitant de manière optimisée.
#Importance des caractéristiques : XGBoost peut calculer l'importance des caractéristiques, ce qui permet d'identifier quelles variables ont le plus d'impact sur les prédictions.
