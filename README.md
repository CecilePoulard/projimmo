# Projimmo

Projimmo est un projet de mise en application de nos connaissances acquises lors du bootcamp Data Science/IA ayant eu lieu l'été 2024.
L'objectif est d'estimer le prix d'un bien immobilier situé dans une des 10 plus grandes villes françaises.

## Equipe
Cécile Poulard

Lallie Dintillac

Fayssal Abdallah Hassan


## Présentation vidéo du projet:
https://www.youtube.com/watch?v=WVA62Tycitk&list=PLEELjdEITejmDCV9e4X6l2iJOhULXoCDl&index=4


## Data

Nous avons utilisé les données disponibles sur le site de data.gouv.fr et disponibles via ce lien:
https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/

## Déroulement du projet


### Prise en main du dataset
- Visualisation des données avant nettoyage: [Voir le notebook](Exploration_notebook/dataviz.ipynb)
- Nettoyage des données: [Voir le module][Voir le fichier data.py](projimmo/prepocess.py)
- Visualisation des données  après nettoyage: [Voir le notebook](Exploration_notebook/datavizpostclean.ipynb)
- Feature engeniring : [Voir le notebook](Exploration_notebook/Feature%20engineering.ipynb)



### Choix du modèle et optimisation des paramètres

- Choix du modèles : Nous testons 3 modèles: **Linear regression**, **KNR** (K-Nearest Neighbors Regressor) et **XGBoost** ( eXtreme Gradient Boosting).
- Une fois le modèle sélectionné (XGB), nous optimisons les hyper-paramètres à l'aide d'un GridSearch.


### Déploiement

- Le meilleur modèle avec les meilleurs hyper paramètres sont stockés dans `best_model_XGB.pkl` (en local ou sur GCS selon la décision de l'utilisateur)

- Création d'une API à l'aide de FastAPI

- Déploiement de l'API sur streamlit

## Structure de la librairie


``` bash
projimmo/
│
├── app.py                      # Script principal pour exécuter l'application
├── main.py                     # Fichier principal pour le lancement
├── .env                        # Fichier de configuration des variables d'environnement
├── Dockerfile                  # Fichier Docker pour containeriser l'application
├── requirements.txt            # Liste des dépendances Python
├── setup.py                    # Script d'installation pour le package
├── best_model_XGB.pkl          # Modèle ML pré-entrainé pour les prédictions
├── preproc_baseline.pkl        # Préprocesseur des données
├──  projimmo/                 # Répertoire racine de la bibliothèque
    ├── data.py                   # Gestion des données brutes et chargement
    ├── fast.py                   # Implémentations de méthodes ou algorithmes optimisés pour améliorer la vitesse
    ├── model.py                  # Définition des modèles d'estimation des prix immobiliers
    ├── optimize_model.py         # Fonctions pour ajuster et optimiser les modèles existants
    ├── params.py                 # Gestion des paramètres ou hyperparamètres utilisés dans le projet
    ├── preprocessor.py           # Prétraitement des données pour les rendre exploitables par les modèles
    └── registry.py               # Gestion d'enregistrements
└── Exploration_notebook/       # Notebooks Jupyter pour l'exploration et le test
    ├── Best-params.ipynb       # Test pour sélectionner les meilleurs paramètres du modèle
    ├── Best-params.ipynb       # Test pour sélectionner les meilleurs paramètres du modèle
    ├── Feature engineering.ipynb # Notebook pour l'ingénierie des fonctionnalités
    ├── explore-datas-DVF.ipynb # Notebook d'exploration des données DVF
    ├── data_final.ipynb                         # Notebook traitant les étapes finales de préparation et nettoyage des données avant modélisation
    ├── dataviz.ipynb                            # Notebook dédié à la visualisation des données brutes, pour identifier les tendances ou anomalies
    └── datavizpostclean.ipynb                   # Visualisation des données après nettoyage, pour valider la qualité des transformations effectuées

```
