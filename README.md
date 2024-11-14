# Projimmo

Projimmo est un projet de mise en application de nos connaissances acquises lors du bootcamp Data Science/IA ayant eu lieu l'été 2024.
L'objectif était d'estimer le prix d'un bien immobilier situé dans une des 10 plus grandes villes françaises.

## Data

Nous avons utilisé les données disponibles sur le site de data.gouv.fr et disponibles via ce lien:
https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/

## Structure de la librairie

preparation des données: data.py
puis preprocess data grace à la librairie sklearn preprocessor.py



params.py: variables d'environnement/blobales








# Instructions pour l’environnement et installation du package

## 1. Créer un environnement virtuel avec pyenv
# pyenv virtualenv 3.10.6 projimmoenv

## 2. Activer l’environnement pour le projet projimmo
# Dans le répertoire du projet, créer un fichier `.python-version` pour activer automatiquement l'environnement
# pyenv local projimmoenv

## 3. Gérer les versions des packages
# - Lister les packages installés : `pip list`
# - Mettre à jour pip et les packages : `pip install --upgrade pip`

## 4. Installer des packages de base
# pip install pandas
# pip install ipython    # Requis pour IPython
# pip install ipykernel  # Requis pour Jupyter Notebooks

## 5. Installer le package projimmo dans l'environnement
# pip install .          # Installation standard
# pip install -e .       # Installation en mode édition pour suivre les modifications du code

## 6. Vérifier l’installation
# pip freeze             # Vérifie que `projimmo` est bien installé

## 7. Pour les notebooks, activer le rechargement automatique
# Dans les notebooks, ajouter au début :
# %load_ext autoreload
# %autoreload 2






















## Installation
Via PyPI

    pip install rubiks-cube-gym
Or build from source

    git clone https://github.com/DoubleGremlin181/RubiksCubeGym.git
    cd RubiksCubeGym
    pip install -e .

## Requirements

 - gym
 - numpy
 - opencv-python
 - wget

## Scrambling
You can pass the scramble as a parameter for the reset function
`self.reset(scramble="R U R' U'")`

The scramble should follow the [WCA Notation](https://www.worldcubeassociation.org/regulations/#article-12-notation)

##  Example
    import gym



``` python
import streamlit as st
```
Installation du package:
dans le terminal:
make install_package


ne pas oublier de lancer
direnv allow
puis direnv reload quand on le modifie
