#projimmo/optimize_model.py

from sklearn.model_selection import GridSearchCV
from projimmo.params import *

def grid_search(X_train, y_train, model_type,initialized_model):
    """
    Grid_search pour model initialisé
    renvoie les meilleurs estimateurs
    """
    model = initialized_model
    if model_type == "KNR":
        param_grid = PARAM_GRID_KNN
    elif model_type == "LR":
        model = initialized_model
        param_grid = PARAM_GRID_LR
    elif model_type == "XGB":
        model = initialized_model
        param_grid = PARAM_GRID_XGB
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        verbose=1)
    grid_search.fit(X_train, y_train)
    print(f"Best parameters for {model_type}: {grid_search.best_params_}")
    print(f"Best score for {model_type}: {-grid_search.best_score_}")

    return grid_search.best_params_
