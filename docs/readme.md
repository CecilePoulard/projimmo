


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
