import streamlit as st
import requests

'''
# Projimmo
'''

st.markdown('''
### Sélectionnez les critères
''')
# Dictionary to map department names to numeric values
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
    'Autre':'AUTRE'
}
mois= {
    'Janvier': '1',
    'Février': '2',
    'Mars': '3',
    'Avril': '4',
    'Mai': '5',
    'Juin': '6',
    'Juillet': '7',
    'Aout': '8',
    'Septembre': '9',
    'Octobre': '10',
    'Novembre':'11',
    'Décembre':'12'
}
types_de_local={'Maison':'1','Appartement':'2'}
 # ['RUE', 'AV', 'CHE', 'BD', 'RTE', 'ALL', 'IMP', 'PL', 'RES', 'CRS','AUTRE']
 # {'maison': 1, 'appartement': 2}
# Create a form for input parameters
with st.form(key='params_for_api'):
        # Predefined department options for the user to select

    # Section 1: Adresse et Voie
    st.subheader("Informations de l'adresse")

    type_de_voie_label = st.selectbox(
        "Sélectionner le type de voie",
        options=list(types_de_voies.keys()),
        index=10
    )
        # Extract the numeric department code
    type_de_voie = types_de_voies[type_de_voie_label]

    # Predefined department options for the user to select
    departement_label = st.selectbox(
        "Select departement",
        options=list(departments.keys())
    )
        # Extract the numeric department code
    departement = departments[departement_label]

    # Section 2: Surfaces
    st.subheader("Informations sur les surfaces")
    surface_reelle_bati = st.number_input("Surface réelle", value=0.0, min_value=0.0 )
    #La surface réelle est la surface mesurée au sol entre les murs
#Surface réelle ou séparations et arrondie au bâti mètre carré inférieur. Les
#surfaces des dépendances ne sont pas prises en compte.
    somme_surface_carrez = st.number_input("Surface carrez", value=0.0, min_value=0.0 )
   # surface_terrain = st.number_input("Surface du terrain", value=0.0, min_value=0.0 )



    # Section 3: Info générales sur le logement
    st.subheader("Informations générales sur le logement")
    #type_de_voie = st.number_input("Pickup type_de_voie", value=0)
    #nombre_de_lots = st.number_input("Pickup nombre_de_lots", value=0)
    #code_type_local = st.number_input("Pickup code_type_local", value=0)
        # Input for number of lots
    nombre_de_lots = st.number_input("Nombre de lots",format="%d", value=1, min_value=1 )

    # Dropdown for type of property (local)
    code_type_local_label = st.selectbox(
        "Sélectionner le type de local",
        options=list(types_de_local.keys())
    )
    code_type_local = types_de_local[code_type_local_label]




    nombre_pieces_principales = st.number_input("Nombre de pièces",format="%d", value=0)
    st.markdown("*Les cuisines, salles d’eau et dépendances ne sont pas prises en compte.*")
    # Section 3: Info générales sur la période
    st.subheader("Informations générales sur le date d'achat")
#Les cuisines, salles d’eau et dépendances ne sont pas prises en compte.
    year_mutation = st.number_input("Année", format="%d",value=2024, min_value=2019 )

    month_mutation_label = st.selectbox(
        "Sélectionner le mois",
        options=list(mois.keys()),
        index=0
    )
        # Extract the numeric department code
    month_mutation = mois[month_mutation_label]




    submitted = st.form_submit_button('Make prediction')

# Check if the form is submitted
if submitted:
        # Check if any of the surface inputs are zero  and surface_terrain == 0.0
    if somme_surface_carrez == 0.0 and surface_reelle_bati == 0.0:
        st.error("Veuillez entrer au moins une surface (Surface Carrez, Surface réelle bâtie ou Surface du terrain).")

    # Check if any of the inputs are zero or empty surface_terrain == 0 and
    if (type_de_voie == 0 and nombre_de_lots == 0 and code_type_local == 0 and
        surface_reelle_bati == 0 and nombre_pieces_principales == 0 and
        month_mutation == 0 and year_mutation == 0 and
        somme_surface_carrez == 0 and departement == 0):

        # If no parameters are entered, set prediction to 0
        st.header(f'Prédiction du prix: 0 €')

    else:
        # Make the API call only if parameters are provided
        ride_params = {
            'type_de_voie': type_de_voie,
            'nombre_de_lots': nombre_de_lots,
            'code_type_local': code_type_local,
            'surface_reelle_bati': surface_reelle_bati,
            'nombre_pieces_principales': nombre_pieces_principales,
           # 'surface_terrain': surface_terrain,
            'month_mutation': month_mutation,
            'year_mutation': year_mutation,
            'somme_surface_carrez': somme_surface_carrez,
            'departement': departement
        }

        url = 'https://projimmo-639648797396.europe-west1.run.app/predict'

        # API call
        response = requests.get(url, params=ride_params)
        data = response.json()

        # Retrieve and display the prediction
        pred = data['prediction']
        #st.header(data)
        st.header(f'Prédiction du prix: {round(pred, 2):,} €'.replace(',', '  '))
