# Makefile pour automatiser les tâches courantes du projet.

# Utilisation :
# - Pour installer le package en mode développement : `make install_package`
# - Pour nettoyer les fichiers générés automatiquement et cachés : `make clean`

# ===================================
# INSTALLATION DU PACKAGE
# ===================================

## Installe le package en mode développement en le désinstallant d'abord
install_package:
	@pip uninstall -y projimmo || :     # Désinstalle projimmo s'il est déjà installé
	@pip install -e .                   # Installe le package en mode édition (développement)

# ===================================
# NETTOYAGE DES FICHIERS TEMPORAIRES
# ===================================

## Supprime les fichiers temporaires et les fichiers générés pour nettoyer le projet
clean:
	@rm -f */version.txt                # Supprime les fichiers version.txt
	@rm -f .coverage                     # Supprime le fichier de couverture des tests
	@rm -fr **/__pycache__ **/*.pyc      # Supprime les caches Python compilés
	@rm -fr **/build **/dist             # Supprime les dossiers de distribution et de build
	@rm -fr proj-*.dist-info             # Supprime les dossiers d'informations de distribution de projimmo
	@rm -fr proj.egg-info                # Supprime les dossiers egg-info du projet
	@rm -f **/.DS_Store                  # Supprime les fichiers .DS_Store générés par macOS
	@rm -f **/*Zone.Identifier           # Supprime les fichiers Zone.Identifier (Windows)
	@rm -f **/.ipynb_checkpoints         # Supprime les checkpoints générés par Jupyter Notebook
