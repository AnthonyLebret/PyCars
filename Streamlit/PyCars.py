import streamlit as st

st.set_page_config(
    page_title="PyCars",
    page_icon="🚘"
)
st.sidebar.header("PyCars")
st.sidebar.info("Auteurs : \n\n - Isabelle EVRARD [Linkedin](https://www.linkedin.com/in/isabelle-evrard-82a6b2253/) \n\n - Anthony LEBRET [Linkedin](https://linkedin.com/in/anthony-lebret-a7aabb176)")
st.sidebar.info("Formation DataScientist Novembre 2022, [DataScientest](https://datascientest.com/)")
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
image1 = Image.open("Streamlit/CO2_wide.jpg")
st.image(image1)
    
# Titre
st.markdown("<h1 style='text-align: center;'>PyCars : Prédictions des émissions de CO₂</h1>", unsafe_allow_html=True)

# Project PyCars
st.write("Projet réalisé dans le cadre de la formation Data Scientist de [DataScientest](https://datascientest.com/), promotion novembre 2022.")
st.write("Auteurs :")
st.write("- Isabelle EVRARD", "[Linkedin](https://www.linkedin.com/in/isabelle-evrard-82a6b2253/)")
st.write("- Anthony LEBRET", "[Linkedin](https://linkedin.com/in/anthony-lebret-a7aabb176)")
st.write("Source de données :", "https://www.eea.europa.eu/data-and-maps/data/co2-cars-emission-22")
st.write()

# Contexte
st.header("Contexte")
st.write("Chaque année, **l'agence européenne de l’environnement (AEE)** construit et publie une base de données pour la surveillance des émissions de CO₂ des voitures particulières. A partir de ces données, l'objectif de cette étude sera **d'identifier les caractéristiques techniques responsables du taux d’émission de CO₂** et **d’établir un modèle prédictif de la pollution engendrée par de nouveaux types de véhicules.**")

st.info("Pour accéder à nos analyses détaillées et interactives (Data visualisation, Machine learning : Régression et Classification), cliquez dans le menu de gauche. Quant à notre synthèse, vous la trouverez ci-dessous.")

st.header("Résultats")

st.markdown("#### Quelles variables impactent le plus le taux d'émission de CO₂ ?")
st.write("L'importance des variables dans la prédiction des émissions de CO₂ a été vérifiée autour de différents axes : corrélation de Pearson, analyse des coefficients de régression, analyse de l'importance dans un arbre de décisions et valeurs de Shap notamment.")
st.write("Toutes ces méthodes corroborent l'impact de certaines caractéristiques techniques : **la masse d'un véhicule, la puissance moteur, la cylindrée, l'empattement, le type de carburant et la présence ou non d'assistance électrique** sont les variables les plus déterminantes dans la prédiction du taux d'émission de CO₂.")

st.markdown("#### Le Gradient Boosting se montre très efficace")
st.write("La modélisation du taux d’émission de CO₂ en fonction des caractéristiques du véhicule a été appréhendée selon deux approches : **la régression et la classification**. De nombreux modèles ont été testés et le modèle prédictif le plus performant a été exploité en calibrant au mieux ses hyperparamètres.")

st.write("Pour les deux approches, **le modèle XGBoost s'est montré le plus performant.**")

st.markdown("#### Régression ou classification ?")
st.write("Pour répondre à notre problématique, le modèle de régression était plus pertinent. D'autant plus que **les résultats obtenus sont très bons** : le coefficient de détermination obtenu sur les données de test est de **0.98**, avec une erreur absolue moyenne de **3.79** (en basant notre modèle sur les nouvelles normes européennes, après 2019).")
st.write("L'approche de classification n'est pas pour autant à jeter. Il est tout à fait possible qu’un système de classes pour les véhicules plus ou moins polluants soit mis en place à l’avenir, à l’instar des classes énergétiques que l’on peut trouver sur la plupart des produits électroniques aujourd’hui.")
st.write("On pourrait notamment penser au **dispositif Crit’air**, mis en place en 2015, qui est censé catégoriser les véhicules selon la pollution qu’ils engendrent, bien que celui-ci ne prenne en compte que le type de carburant et l’année de construction du véhicule.")

st.markdown("#### Une optimisation délicate")
st.write("Bien que le modèle XGBoost soit simple à implémenter, la difficulté réside dans le choix de ses nombreux paramètres.")
st.write("Deux stratégies d’optimisation automatique des hyperparamètres du modèle XGBoost ont été développées : **sélection séquentielle (GridSearch)** et **optimisation bayésienne (librairie Hyperopt)**.")
st.write("Une comparaison a été réalisée sur le modèle de régression. Résultat : **la méthode bayésienne est deux fois plus rapide.**") 
st.write("L'optimisation bayésienne ne permet pas d’améliorer les performances du modèle mais elle est cependant plus rapide à mettre en œuvre. Pour aller plus loin, il serait intéressant d’examiner les véhicules impactés par les changements de paramètres.")

st.header("Conclusion et perspectives")
st.write("Le modèle de régression obtenu par XGBoost est robuste et nous semble **parfaitement exploitable**. Cet exercice nous a permis de mettre en œuvre de nombreuses techniques et malgré des performances exceptionnelles, quelques pistes d’améliorations peuvent être proposées, notamment une **sélection de variable ou une réduction de dimension** pour les caractéristiques fortement corrélées entre elles.")
st.write("On pourrait également venir enrichir le modèle de données supplémentaires, comme une **mesure de la surface frontale des véhicules** (influence sur l'aérodynamisme) ou encore une information de la **présence ou non de boîte automatique** pour le passage des vitesses ; des caractéristiques dont l'effet est avéré sur la consommation de carburant et donc **l'émission de gaz à effet de serre.**")
