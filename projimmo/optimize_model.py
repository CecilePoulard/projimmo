# projimmo/optimize_model.py

from sklearn.model_selection import GridSearchCV
from projimmo.params import *  # Importation des paramètres spécifiques à chaque modèle

def grid_search(X_train, y_train, model_type, initialized_model):
    """
    Effectue une recherche par grille (Grid Search) pour optimiser les hyperparamètres du modèle initialisé.

    Cette fonction utilise `GridSearchCV` pour tester différents ensembles d'hyperparamètres
    pour un modèle donné, en fonction du type de modèle spécifié. Elle renvoie les meilleurs
    paramètres pour le modèle, ainsi que le meilleur score obtenu lors de la validation croisée.

    Args:
        X_train (DataFrame): Les caractéristiques de l'ensemble d'entraînement.
        y_train (Series): Les cibles de l'ensemble d'entraînement.
        model_type (str): Le type de modèle pour lequel effectuer la recherche ('KNR', 'LR', 'XGB').
        initialized_model (estimator): Le modèle déjà initialisé avant la recherche par grille.

    Returns:
        dict: Les meilleurs paramètres trouvés pour le modèle, selon la recherche par grille.
    """

    # Sélection du modèle et des paramètres à optimiser en fonction du type de modèle
    model = initialized_model
    if model_type == "KNR":
        param_grid = PARAM_GRID_KNN  # Paramètres spécifiques pour K-nearest neighbors
    elif model_type == "LR":
        param_grid = PARAM_GRID_LR   # Paramètres spécifiques pour la régression linéaire
    elif model_type == "XGB":
        param_grid = PARAM_GRID_XGB  # Paramètres spécifiques pour XGBoost

    # Initialisation du GridSearchCV avec les paramètres et le modèle
    grid_search = GridSearchCV(
        estimator=model,              # Le modèle à optimiser
        param_grid=param_grid,        # La grille des hyperparamètres à tester
        cv=5,                          # Validation croisée à 5 plis
        scoring='neg_mean_squared_error',  # Critère de performance à optimiser
        verbose=1                      # Affichage des informations pendant l'exécution
    )

    # Entraînement du GridSearchCV sur l'ensemble d'entraînement
    grid_search.fit(X_train, y_train)

    # Affichage des meilleurs paramètres et du meilleur score
    print(f"Best parameters for {model_type}: {grid_search.best_params_}")
    print(f"Best score for {model_type}: {-grid_search.best_score_}")

    # Retour des meilleurs paramètres trouvés
    return grid_search.best_params_
