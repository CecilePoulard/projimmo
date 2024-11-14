"""
projimmo/model.py


Module de traitement et d'évaluation des modèles de régression pour le projet immobilier (projimmo).

Ce module fournit des fonctions pour :
- Diviser les données en ensembles d'entraînement et de test.
- Initialiser et entraîner différents types de modèles de régression (K-nearest neighbors, régression linéaire, XGBoost).
- Évaluer les performances des modèles avec différentes métriques (MSE, MAE, R², RMSE, MSLE).
- Effectuer une validation croisée pour évaluer les performances du modèle.

Fonctions disponibles :
1. split_df(X, y, test_size=0.2):
    Divise les données en ensembles d'entraînement et de test.

2. cross_validate_model(X, y, model, cv=5, scoring='neg_mean_squared_error'):
    Effectue une validation croisée pour évaluer les performances du modèle.

3. initialize_model(model_type, n_neighbors=5, objective='reg:squarederror', max_depth=4, eta=0.1, n_estimators=100):
    Initialise le modèle spécifié par l'utilisateur (KNN, Régression Linéaire, ou XGBoost).

4. train_model(X_train, y_train, model):
    Entraîne le modèle sur les données d'entraînement.

5. evaluate_model(X_test, y_test, model, model_type):
    Évalue les performances du modèle sur les données de test et affiche plusieurs métriques d'évaluation.

6. Notes générales :
    - **Boosting en gradient (XGBoost)** : XGBoost utilise la méthode de boosting pour combiner plusieurs modèles faibles (arbres de décision peu profonds), ce qui permet d'améliorer la précision des prédictions.
    - **Arbres de décision** : Le modèle de base pour XGBoost est un arbre de décision peu profond, ce qui limite le surapprentissage.
    - **Optimisation avancée** : XGBoost utilise la régularisation, le pruning, gère les données manquantes et calcule l'importance des caractéristiques pour une meilleure interprétation du modèle.

"""




import xgboost as xgb  # Pour utiliser XGBoost
from sklearn.neighbors import KNeighborsRegressor  # Pour le K-nearest neighbors
from sklearn.linear_model import LinearRegression  # Pour la régression linéaire
from sklearn.model_selection import train_test_split  # Pour diviser les données en ensembles d'entraînement et de test
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error  # Pour les métriques d'évaluation
from sklearn.metrics import accuracy_score  # Pour les tâches de classification
from sklearn.model_selection import cross_val_score, cross_validate  # Pour la validation croisée

import numpy as np


def split_df(X, y, test_size=0.2):
    """
    Divise les données en ensembles d'entraînement et de test.

    Args:
        X (DataFrame): Les caractéristiques des données.
        y (Series): Les cibles des données.
        test_size (float, optionnel): La proportion des données utilisées pour le test (par défaut 0.2).

    Returns:
        tuple: Les ensembles d'entraînement et de test pour X et y.
    """
    # Séparation des données en train et test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


def cross_validate_model(X, y, model, cv=5, scoring='neg_mean_squared_error'):
    """
    Effectue une validation croisée pour évaluer les performances du modèle.

    Args:
        X (DataFrame): Les caractéristiques des données.
        y (Series): Les cibles des données.
        model (estimator): Le modèle à évaluer.
        cv (int, optionnel): Le nombre de plis (folds) pour la validation croisée (par défaut 5).
        scoring (str, optionnel): La méthode de scoring à utiliser pour évaluer les performances (par défaut 'neg_mean_squared_error').

    Returns:
        dict: Les résultats de la validation croisée, comprenant les scores et les métriques.
    """
    # Effectuer la validation croisée
    results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)

    # Convertir les scores négatifs en positifs pour une meilleure interprétation
    mse_scores = -results['test_score']
    mean_mse = np.mean(mse_scores)
    std_mse = np.std(mse_scores)

    # Affichage des résultats
    print(f"Validation croisée :")
    print(f"MSE moyen: {mean_mse:.2f}")
    print(f"Écart-type du MSE: {std_mse:.2f}")

    return results


def initialize_model(model_type, n_neighbors=5, objective='reg:squarederror', max_depth=4, eta=0.1, n_estimators=100):
    """
    Initialise un modèle en fonction du type choisi.

    Args:
        model_type (str): Le type de modèle à initialiser ('KNR', 'LR', 'XGB').
        n_neighbors (int, optionnel): Nombre de voisins pour K-nearest neighbors (par défaut 5).
        objective (str, optionnel): L'objectif du modèle XGBoost (par défaut 'reg:squarederror').
        max_depth (int, optionnel): La profondeur maximale des arbres pour XGBoost (par défaut 4).
        eta (float, optionnel): Le taux d'apprentissage pour XGBoost (par défaut 0.1).
        n_estimators (int, optionnel): Le nombre d'estimateurs pour XGBoost (par défaut 100).

    Returns:
        estimator: Le modèle initialisé.
    """
    if model_type == "KNR":
        model = KNeighborsRegressor(n_neighbors=n_neighbors)
    elif model_type == "LR":
        model = LinearRegression()
    elif model_type == "XGB":
        model = xgb.XGBRegressor(objective=objective, max_depth=max_depth, eta=eta, n_estimators=n_estimators)
    return model


def train_model(X_train, y_train, model):
    """
    Entraîne le modèle sur les données d'entraînement.

    Args:
        X_train (DataFrame): Les caractéristiques d'entraînement.
        y_train (Series): Les cibles d'entraînement.
        model (estimator): Le modèle à entraîner.

    Returns:
        estimator: Le modèle entraîné.
    """
    fitted_model = model.fit(X_train, y_train)
    return fitted_model


def evaluate_model(X_test, y_test, model, model_type):
    """
    Évalue les performances du modèle sur l'ensemble de test et affiche plusieurs métriques.

    Args:
        X_test (DataFrame): Les caractéristiques de test.
        y_test (Series): Les cibles de test.
        model (estimator): Le modèle à évaluer.
        model_type (str): Le type de modèle ('KNR', 'LR', 'XGB').

    Returns:
        dict: Un dictionnaire avec les métriques d'évaluation du modèle.
    """
    # Prédictions sur les données de test
    y_pred = model.predict(X_test)

    # Calcul des métriques d'évaluation
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Vérification si toutes les prédictions et cibles sont positives pour calculer le MSLE
    if np.all(y_pred > 0) and np.all(y_test > 0):
        msle = mean_squared_log_error(y_test, y_pred)
    else:
        msle = None

    # Affichage des résultats
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared (R²) Score: {r2}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Squared Logarithmic Error (MSLE): {msle}")

    return {'mse': mse, 'mae': mae, 'r2': r2, 'rmse': rmse, 'msle': msle}
