"""
setup.py



Ce script configure et initialise le package Python `projimmo` pour la modélisation des prix de l'immobilier.

Le fichier `setup.py` est utilisé par setuptools pour empaqueter le projet avec les dépendances, métadonnées
et informations nécessaires. Une fois configuré, il permet d'installer `projimmo` dans un environnement Python
en utilisant `pip install .` ou `pip install -e .` pour le mode édition.

Sections principales :
----------------------
1. Lecture des dépendances depuis le fichier `requirements.txt`
   - Lit chaque ligne du fichier `requirements.txt` pour collecter les packages requis.
   - Exclut les dépendances installées directement depuis un dépôt git (`git+`), car elles doivent être installées
     manuellement.

2. Définition des métadonnées et des dépendances via `setup()`
   - `name` : Nom du package (ici, `projimmo`).
   - `description` : Brève description du package, précisant son objectif (modélisation de prix immobiliers).
   - `author` : Liste des auteurs du projet.
   - `install_requires` : Spécifie les dépendances requises, issues de `requirements.txt`.
   - `packages` : Inclut tous les sous-modules du package en utilisant `find_packages()`.

Utilisation :
-------------
Pour installer le package et ses dépendances, exécutez l'une des commandes suivantes depuis le répertoire
contenant `setup.py` :

- Installation standard : `pip install .`
- Installation en mode édition (permet de détecter les modifications du code) : `pip install -e .`

Ces commandes créent une entrée `projimmo` dans les packages Python installés, visible avec `pip list`.

Exemples supplémentaires :
--------------------------
- Pour inclure des scripts, ajouter le paramètre `scripts` dans `setup()`.
- Pour spécifier une version de Python minimale requise, ajouter `python_requires='>=3.x'`.

"""

from setuptools import find_packages, setup

# Lecture du fichier des dépendances (requirements.txt)
with open("requirements.txt") as f:
    content = f.readlines()
# Exclut les dépendances installées depuis un dépôt git
requirements = [x.strip() for x in content if "git+" not in x]

# Configuration de la distribution du package
setup(
    name='projimmo',                                  # Nom du package
    description="Modélisation du prix de l'immobilier", # Description du projet
    author="Lallie, Fayssal, Cecile",                 # Auteurs du projet
    install_requires=requirements,                    # Dépendances du projet
    packages=find_packages()                          # Recherche des packages inclus dans le projet
)
