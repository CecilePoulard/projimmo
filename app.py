"""
app.py

app.py contient une application Streamlit pour prédire les prix immobiliers.
Les utilisateurs peuvent entrer des critères d'une propriété pour recevoir une estimation de prix.

Dépendances:
- streamlit
- requests
"""

import streamlit as st
import requests

# Titre de l'application
st.title("Projimmo - Estimation de Prix Immobilier")

# Description de l'interface utilisateur
st.markdown("### Sélectionnez les critères")

# Mapping des départements français vers leurs codes numériques
departments = {
    '75-Paris': '75',
    '13-Marseille': '13',
    '69-Lyon': '69',
    '31-Toulouse': '31',
    '06-Nice': '06',
    '44-Nantes': '44',
    '34-Montpellier': '34',
    '67-Strasbourg': '67',
    '33-Bordeaux': '33',
    '59-Lille': '59'
}

# Dictionnaire pour les types de voies avec leurs codes abréviés
types_de_voies = {
    'Rue': 'RUE',
    'Avenue': 'AV',
    'Chemin': 'CHE',
    'Boulevard': 'BD',
    'Route': 'RTE',
    'Allée': 'ALL',
    'Impasse': 'IMP',
    'Place': 'PL',
    'Résidence': 'RES',
    'Cours': 'CRS',
    'Autre': 'AUTRE'
}

# Mapping des mois de l'année vers leurs codes numériques
mois = {
    'Janvier': '1', 'Février': '2', 'Mars': '3', 'Avril': '4',
    'Mai': '5', 'Juin': '6', 'Juillet': '7', 'Aout': '8',
    'Septembre': '9', 'Octobre': '10', 'Novembre': '11', 'Décembre': '12'
}

# Types de propriétés (maison ou appartement) avec leurs codes
types_de_local = {'Maison': '1', 'Appartement': '2'}

# Création d'un formulaire pour la saisie des paramètres
with st.form(key='params_for_api'):
    # Section 1 : Adresse et type de voie
    st.subheader("Informations de l'adresse")

    # Sélection du type de voie
    type_de_voie_label = st.selectbox("Sélectionner le type de voie", options=list(types_de_voies.keys()), index=10)
    type_de_voie = types_de_voies[type_de_voie_label]

    # Sélection du département
    departement_label = st.selectbox("Sélectionner le département", options=list(departments.keys()))
    departement = departments[departement_label]

    # Section 2 : Informations sur les surfaces
    st.subheader("Informations sur les surfaces")
    surface_reelle_bati = st.number_input("Surface réelle (m²)", value=0.0, min_value=0.0)
    somme_surface_carrez = st.number_input("Surface Carrez (m²)", value=0.0, min_value=0.0)

    # Section 3 : Informations générales sur le logement
    st.subheader("Informations générales sur le logement")

    # Saisie du nombre de lots dans le bien
    nombre_de_lots = st.number_input("Nombre de lots", format="%d", value=1, min_value=1)

    # Sélection du type de logement
    code_type_local_label = st.selectbox("Sélectionner le type de logement", options=list(types_de_local.keys()))
    code_type_local = types_de_local[code_type_local_label]

    # Saisie du nombre de pièces principales (sans les cuisines et dépendances)
    nombre_pieces_principales = st.number_input("Nombre de pièces principales", format="%d", value=0)
    st.markdown("*Les cuisines, salles d’eau et dépendances ne sont pas prises en compte.*")

    # Section 4 : Informations sur la date d'achat
    st.subheader("Informations générales sur la date d'achat")

    # Sélection de l'année de mutation (ou d'achat)
    year_mutation = st.number_input("Année", format="%d", value=2024, min_value=2019)

    # Sélection du mois de mutation
    month_mutation_label = st.selectbox("Sélectionner le mois", options=list(mois.keys()), index=0)
    month_mutation = mois[month_mutation_label]

    # Bouton pour soumettre le formulaire
    submitted = st.form_submit_button('Faire une prédiction')

# Vérification de la soumission du formulaire
if submitted:
    # Vérification que les surfaces ne sont pas toutes égales à zéro
    if somme_surface_carrez == 0.0 and surface_reelle_bati == 0.0:
        st.error("Veuillez entrer au moins une surface (Surface Carrez ou Surface réelle bâtie).")
    else:
        # Construction des paramètres pour l'appel API
        ride_params = {
            'type_de_voie': type_de_voie,
            'nombre_de_lots': nombre_de_lots,
            'code_type_local': code_type_local,
            'surface_reelle_bati': surface_reelle_bati,
            'nombre_pieces_principales': nombre_pieces_principales,
            'month_mutation': month_mutation,
            'year_mutation': year_mutation,
            'somme_surface_carrez': somme_surface_carrez,
            'departement': departement
        }

        # URL de l'API de prédiction de prix immobilier
        url = 'https://projimmo-639648797396.europe-west1.run.app/predict'

        # Appel à l'API pour obtenir la prédiction
        response = requests.get(url, params=ride_params)
        data = response.json()

        # Affichage du résultat de la prédiction
        pred = data['prediction']
        st.header(f'Prédiction du prix: {round(pred, 2)} €')
