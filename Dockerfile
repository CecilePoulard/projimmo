# Utiliser une image de base Python officielle
FROM python:3.10.6

# Créer un répertoire de travail
#WORKDIR /projimmo


WORKDIR /prod
# Copier les fichiers de dépendances
COPY requirements.txt requirements.txt

# Installer les dépendances
RUN pip install -r requirements.txt

# Copier le code de l'application dans le conteneur
COPY projimmo projimmo

# Copier les fichiers de modèles et autres ressources nécessaires
COPY preproc_baseline.pkl preproc_baseline.pkl
#/projimmo/preproc_baseline.pkl
#COPY /home/cpoulard/code/CecilePoulard/gcp/wagon-bootcamp-429817-511f35e74646.json /projimmo/gcp/wagon-bootcamp-429817-511f35e74646.json

# Commande pour exécuter l'application FastAPI avec Uvicorn
CMD uvicorn projimmo.fast:app --host 0.0.0.0 --port $PORT
#["uvicorn", "projimmo.fast:app", "--host", "0.0.0.0", "--port", $PORT]



# Utiliser une image de base Python officielle
#FROM python:3.10.6
# Copier le code de l'application dans le conteneur
#COPY projimmo /projimmo
# Copier les fichiers de dépendances
#COPY requirements.txt /requirements.txt
# Installer les dépendances
#RUN pip install -r requirements.txt

# Copier le fichier `preproc_baseline.pkl` dans le répertoire de travail
#COPY preproc_baseline.pkl /preproc_baseline.pkl
#ENV GOOGLE_APPLICATION_CREDENTIALS="/home/cpoulard/code/CecilePoulard/gcp/wagon-bootcamp-429817-511f35e74646.json"
# Dockerfile
#COPY /home/cpoulard/code/CecilePoulard/gcp/wagon-bootcamp-429817-511f35e74646.json preproc_baseline.pkl /app/gcp
# Commande pour exécuter l'application FastAPI avec Uvicorn
#
#CMD uvicorn projimmo.fast:app --host 0.0.0.0 --port 8000

# Utiliser une image de base Python officielle
#FROM python:3.10.6

# Créer un répertoire de travail
#WORKDIR /app

# Copier le code de l'application dans le conteneur
#COPY projimmo /app/projimmo

# Copier les fichiers de dépendances
#COPY requirements.txt /app/requirements.txt

# Installer les dépendances
#RUN pip install -r /app/requirements.txt

# Copier le fichier `preproc_baseline.pkl` dans le répertoire de travail
#COPY preproc_baseline.pkl /app/preproc_baseline.pkl

# Copier les fichiers de credentials GCP
#COPY /home/cpoulard/code/CecilePoulard/gcp/wagon-bootcamp-429817-511f35e74646.json /app/gcp/

# Définir la variable d'environnement pour les credentials GCP
#ENV GOOGLE_APPLICATION_CREDENTIALS="/app/gcp/wagon-bootcamp-429817-511f35e74646.json"

# Commande pour exécuter l'application FastAPI avec Uvicorn
#CMD ["uvicorn", "projimmo.fast:app", "--host", "0.0.0.0", "--port", "8000"]




#######
#Creer une variable d'env dans .env
# e.g. GAR_IMAGE=projimmo
# puis dans terminale lancer
# docker build --tag=$GAR_IMAGE:dev .
# verifier que l'image existe bien: docker images


# lancer docker
#docker run -it -e PORT=8000 -p 8000:8000 $GAR_IMAGE:dev bash
#--> ouvre le terminal de l'image docker
# verifier que projimmo est bien installé, ainsi que requirements.txt
# exit pour sortir

#copier le .env dans notre image
#docker run -it -e PORT=8000 -p 8000:8000 --env-file your/path/to/.env $GAR_IMAGE:dev
