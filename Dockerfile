# Utiliser une image de base officielle de Python 3.10
FROM python:3.10.6

# Définir un répertoire de travail dans le conteneur
WORKDIR /prod

# Copier le fichier requirements.txt pour installer les dépendances
COPY requirements.txt requirements.txt

# Installer les dépendances listées dans requirements.txt
RUN pip install -r requirements.txt

# Copier le code de l'application dans le répertoire de travail
COPY projimmo projimmo

# Copier le fichier de prétraitement des données (modèle) nécessaire pour les prédictions
COPY preproc_baseline.pkl preproc_baseline.pkl

# Commande pour lancer l'application FastAPI avec Uvicorn
CMD ["uvicorn", "projimmo.fast:app", "--host", "0.0.0.0", "--port", "$PORT"]

# Notes supplémentaires :
# Pour lancer le build de cette image, vous pouvez utiliser une variable d'environnement pour le tag:
# docker build --tag=$GAR_IMAGE:dev .
# Vérifier que l'image est bien créée avec la commande : docker images

# Commande pour lancer un conteneur Docker :
# docker run -it -e PORT=8000 -p 8000:8000 $GAR_IMAGE:dev bash
# Cela ouvre un terminal dans l'image Docker pour tester si le code et les dépendances sont correctement installés.

# Si vous utilisez un fichier .env pour les variables d'environnement, vous pouvez l'ajouter comme suit :
# docker run -it -e PORT=8000 -p 8000:8000 --env-file your/path/to/.env $GAR_IMAGE:dev
