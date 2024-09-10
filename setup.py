#setup.py

from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]


setup(name='projimmo',
      description="Modelisation du prix de l'immobilier",
      author="Lallie, Fayssal, Cecile",
      install_requires=requirements, #va chercher les package dont on a besoin listé dans requirements.txt
      packages=find_packages(),#librairies dans notre environnement. NB: ici que ["projimmo"]
      )











############ TERMINAL ################

##create new projimmoenv inside python 3.10.6
#pyenv virtualenv 3.10.6 projimmoenv

##In project projimmo, create '.python-version' that
##activates projimmoenv when present
#pyenv local projimmo

##Pour savoir les version des librairies
#pip list
## Pour mettre à jour les librairies présentes dans notre environnement
#pip install --upgrade pip

## Installations de base
#pip install pandas
#pip install ipython   # needed for ipython
#pip install ipykernel # needed for notebooks

## Installation de la librairies projimmo dans notre environnement virtuel
#pip install . ou pip install -e . #-> '-e' permet de mettre à jour la librairie
                                # quand les fichiers sont modifiés (mieux!)
## normalement installe la libraire, à verifier avec:
#pip freeze #-->projimmo doit être dans la liste



## Quand on utilise un ipynb, mettre au début (pour eviter de devoir relancer le kernel à chaque modif de la librairie):
#%load_ext autoreload
#%autoreload_2
