import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dataset", page_icon="💽")
st.sidebar.header("Dataset")
st.sidebar.info("Auteurs : \n\n - Isabelle EVRARD [Linkedin](https://www.linkedin.com/in/isabelle-evrard-82a6b2253/) \n\n - Anthony LEBRET [Linkedin](https://linkedin.com/in/anthony-lebret-a7aabb176)")
st.sidebar.info("Données : [Site de l'Agence européenne pour l'environnement](https://www.eea.europa.eu/data-and-maps/data/co2-cars-emission-22)")

# Cacher "Made with Streamlit"
hide_footer_style = """ 
    <style>
    footer {visibility: hidden; }
    </style>
    """
st.markdown(hide_footer_style, unsafe_allow_html=True)

# Image CO2
from PIL import Image
image = Image.open('Streamlit/CO2_wide.jpg')
st.image(image)

# Titre
st.title('Présentation du jeu de données')
st.markdown("#### Données brutes")

st.write("Compte tenu de l’important volume de données générées au niveau européen, le projet s’est concentré sur **l’analyse des données françaises de 2019**. L’année 2020 a été écartée compte tenu de l’impact probable de l’épidémie de Covid sur la vente des véhicules neufs.")
st.write("Ainsi, les informations complètes du parc des véhicules particuliers et utilitaires légers neufs immatriculés en France pour l'année 2019 ont été téléchargées sur le site de l’AEE. Ce jeu de données est composé de **2 305 720 immatriculations**.")
st.write("Notre dataset dénommé **“Cars”** est constitué de **16 975 véhicules neufs** et a été créé à partir du jeu de données source en **supprimant les doublons** sur un identifiant véhicule.")

# Import dataset
cars = pd.read_csv("data/cars_FR2019.csv", index_col=0)
cars_clean = pd.read_csv("data/cars_FR2019_clean.csv", index_col=0)
cars_clean = cars_clean.drop(['Cr', 'r', 'size', 'indicator', 'Mh', 'T', 'Va', 'Ve'], axis=1)
# Renommer les variables
rename = {'m': 'Masse (kg)',
          'Mt': 'Masse test (kg)',
          'ec': 'Cylindrée en cm3',
          'ep': 'Puissance moteur en KW',
          'Enedc': 'Emissions de CO2 en g/km (norme NEDC)',
          'Ewltp': 'Emissions de CO2 en g/km (norme WLTP)',
          'Ft': 'Type de carburant',
          'Fm': 'Mode de carburation',
          'IT': 'Eco-innovation',
          'W': 'Empattement en mm',
          'At1': "Largeur de voie essieu directeur (mm)",
          'At2': "Largeur de voie autre essieu (mm)",
          'At2supAt1': 'Essieu arrière supérieur essieu avant',
          'Mp': 'Pool de constructeur',
          'Cn': 'Nom commercial',
          'Ct': 'Motricité'}
cars_clean_rename = cars_clean.rename(rename,axis=1)

if st.checkbox("Afficher les données brutes “Cars”"):
    if st.checkbox("Appliquer le pre-processing"):
        if st.checkbox("Renommer les variables"):
            st.dataframe(cars_clean_rename)
            st.write("Dimensions du dataset :", cars_clean_rename.shape)
            st.info("La variable 'Nom commercial' sera retirée avant la modélisation, nous avons choisi de l'inclure ici pour le côté interactif avec la base de données.")
        else:
            st.dataframe(cars_clean)
            st.write("Dimensions du dataset :", cars_clean.shape)
            st.info("La variable 'Cn' sera retirée avant la modélisation, nous avons choisi de l'inclure ici pour le côté interactif avec la base de données.")
    else:
        st.dataframe(cars)
        st.write("Dimensions du dataset :", cars.shape)
        

st.markdown("#### Description du pre-processing")

st.markdown("##### Suppression de données")
st.write("Dans un objectif de prédiction des émissions de CO₂ des véhicules, de nombreuses variables sont inutiles à nos modèles de Machine Learning. Notamment, les variables liées à l'identification d'un véhicule ou autres mesures administratives seront supprimées.") 
with st.expander("Afficher/Cacher les variables supprimées"):
    st.write("- Numéro d'homologation et identifiant de famille d’interpolation.")
    st.write("- Variables en lien avec les contrôles de simulation CO₂MPAS.")
    st.write("- Consommation d’énergie électrique qui ne concerne que les véhicules électriques.")
    st.write("- Marque des véhicules et variables correspondant au nom du constructeur, seul l’item pool de constructeur est conservé.")

with st.expander("Afficher/Cacher les véhicules supprimés"):
    st.write("Les 3 404 véhicules (20%) suivants sont supprimés :")
    st.write("- 3 147 véhicules sans mesure d’émissions de CO₂ selon le protocole WLTP : ces véhicules sont essentiellement des véhicules de fin de série, la nouvelle réglementation ne leur étant pas imposée.")
    st.write("- 76 véhicules codés “out of scope” et “duplicated” pour le nom du constructeur. Les guidelines précisant que les caravanes, corbillards et ambulances sont considérés hors champs.")
    st.write("- 153 véhicules électriques et 3 véhicules à hydrogène n’émettant pas de CO₂.")
    st.write("- 41 véhicules alimentés avec un carburant d’utilisation marginale et dont l’impact sur les émissions de CO₂ ne pourra pas être exploré : gaz naturel (26), gaz de pétrole liquéfié (19), mélange gaz-naturel et biométhane (4), super-éthanol (2).")

st.markdown("##### Traitement des données manquantes")
st.write("Les États membres devant suivre des procédures réglementaires et la collecte des données ayant débuté en 2010 on peut considérer que le circuit de production est bien rodé. On observe donc sans surprise des taux de données manquantes très faibles.")
with st.expander("Afficher/Cacher les données ajoutées"):
    st.write("- Type de carburant et mode de carburation manquants pour 8 véhicules BMW. Les valeurs ont été recherchées sur internet.")
    st.write("- La puissance nette maximale manquante pour deux véhicules est recherchée sur internet. La BMW i3 dans sa version de base ou spéciale dispose d’une puissance identique de 125 kW.")
    st.write("- 698 données manquantes pour la masse test. Celle-ci étant fortement corrélée à la masse (coefficient de corrélation de Pearson égal à 0.98), les données manquantes sont implémentées par régression linéaire.")

st.markdown("##### Recodage de variabes")
st.write("Certaines modalités dans les variables catégorielles sont rarissimes. L’impossibilité formelle d’observer des individus dotés de certaines caractéristiques peut provoquer des difficultés d’estimation des coefficients dans les modèles linéaires. Pour cette raison, nous avons simplifié/remplacé certaines modalités.")
with st.expander("Afficher/Cacher les variables recodées"):
    st.write("- Remplacement des modalités d’éco-innovation par un indicateur binaire de leur présence ou non sur le véhicule.")
    st.write("- Suppression des modalités petrol/electric et diesel/electric pour le type de carburant. Ces informations sont redondantes avec le mode de carburation qui indique si un véhicule est hybride ou non.")
    st.write("- Remplacement de la variable de largeur de voie d'essieu arrière par une variable binaire pour indiquer une largeur de voie d'essieu arrière plus grande.")
    st.write("- Les constructeurs sont regroupés en pool selon le rapport européen “Analyses of emission from new cars in 2020”.")

st.success("Le dataset “Cars” final contient **13 571 véhicules**, comme vous pourrez le constater en appliquant le pre-processing ci-dessus.")
