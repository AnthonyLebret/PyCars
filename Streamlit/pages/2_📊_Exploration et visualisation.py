import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly_express as px

st.set_page_config(page_title="Exploration et visualisation", page_icon="📊")
st.sidebar.header("Exploration et visualisation")
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
image = Image.open('Streamlit/CO2_wide.jpg')
st.image(image)

st.title('Data exploration et visualisation')
    
st.write("Les données utilisées pour la visualisation regroupent les véhicules neufs acquis par au moins une personne en France en 2019. Contrairement au dataset principal, les véhicules de tous types ont été conservés, c’est à dire les véhicules ne rejetant pas de CO₂ (électrique, hydrogène) ainsi que les véhicules aux carburants rares.")
    
# lecture du datasets "cars" à partir du jeu de données source en supprimant les doublons sur un identifiant véhicule.
cars = pd.read_csv('data/cars_FR2019.csv', dtype={'Cn': 'str', 'IT': 'str'},index_col=0)
    
# Création des DataFrames "cars_viz" et "parc_viz" et suppression des véhicules avec type de carburant inconnu
cars_viz = cars[(cars['Ft']!='unknown')]

# Création de la variable d'émission moyenne de CO2 (norme WLTP) par type de carburant
moy_CO2_wltp = cars_viz['Ewltp (g/km)'].groupby(cars_viz['Ft']).mean()
    
# Création des variables type_cars et type_parc : nombre de véhicules par type de carburant (en %)
type_cars = cars_viz['Ft'].value_counts(normalize=1)
type_cars *= 100
type_cars = type_cars.sort_index()
    
# Remplacement du nom des modalités de la variable 'Fm'
cars_viz['Fm'] = cars_viz['Fm'].replace(('M', 'H', 'P', 'B', 'F'), ('Mono-fuel', 'Hybrid not off-charged',
                                                                    'Hybrid off-charged', 'Bi-fuel', 'Flex-fuel'))

# Renommer les variables pour la lisibilité des graphiques
viz_rename = {'m (kg)': 'Masse',
              'ec (cm3)': 'Cylindrée',
              'ep (KW)': 'Puissance moteur',
              'Ewltp (g/km)': 'Emissions de CO2',
              'W (mm)': 'Empattement',
              'At1 (mm)': "Largeur de voie essieu dir.",
              'At2 (mm)': "Largeur de voie autre essieu"}
cars_viz = cars_viz.rename(viz_rename,axis=1)
    

### Graphique 1
st.subheader('Graphique 1 : Emissions de CO₂ en fonction du type de carburant')
st.write("Cette première visualisation met en évidence la disparité du taux d’émission de CO₂ entre les différents types de carburants.")
    
fig = px.bar(x=moy_CO2_wltp.index, y=moy_CO2_wltp.values, color=moy_CO2_wltp.index,
             color_discrete_sequence=px.colors.qualitative.Plotly,
             labels={'x':'Type de carburant', 'y':'Emission moyenne de CO2 (g/km)', 'color':'Type de carburant'},
             text_auto='.3s')
st.plotly_chart(fig)

with st.expander("Voir/Cacher la description"):
    st.write("- Les véhicules roulant au carburant E85, également appelé superéthanol (un mélange constitué de biocarburant, d’éthanol et d'essence SP95) rejette beaucoup plus de CO₂ que des véhicules roulant au gaz naturel -ng- par exemple.")
    st.write("- Les véhicules diesel et essence (“petrol”), qui constituent la majorité des véhicules du dataset, émettent une quantité similaire de CO₂.")
    st.write("- Les véhicules hybrides diesel/électrique sont ceux qui rejettent le moins de CO₂ suivis par les véhicules hybrides essence/électrique.")
    
### Graphique 2
st.subheader('Graphique 2 : Proportion des différents types de carburants')
st.write("Ce diagramme en barre présente les proportions des différents types de carburants, et expose un déséquilibre au sein du dataset.")
    
fig = px.bar(x=type_cars.values, y=type_cars.index, orientation='h', color=type_cars.index,
             color_discrete_sequence=px.colors.qualitative.Plotly,
             labels={'x':' ', 'y':'Type de carburant', 'color':'Type de carburant'})
fig.update_layout(xaxis = dict(tickvals = [0, 10, 20, 30, 40, 50, 60],
                               ticktext = ['0%', '10%', '20%', '30%', '40%', '50%', '60%']))
st.plotly_chart(fig)

with st.expander("Voir/Cacher la description"):
    st.write("- Les véhicules diesel et essence (“petrol”) sont largement représentés (plus de 97%).")
    st.write("- Certains types de carburant sont très rares : les véhicules alimentés à l’hydrogène, au superéthanol (E85), au gaz naturel (ng), au mélange gaz naturel-biométhane (ng-biomethane), au gaz de pétrole liquéfié (lpg), ainsi que les véhicules diesel/électrique représentent une très faible proportion du jeu de données (<0.4% tous types de carburant cumulés).")
    st.write("- Les véhicules électriques représentent moins de 1% du dataset, et seront retirés dans la partie pre-processing (ne servent pas à la modélisation puisqu’ils ne rejettent pas de CO₂).")

# Suppression des véhicules électriques et hydrogènes (ne rejettent pas de CO₂)
cars_viz = cars_viz[(cars_viz['Ft']!='electric') & (cars_viz['Ft']!='hydrogen') & (cars_viz['Ft']!='unknown')]

# Graphique 3
st.subheader("Graphique 3 : Boxplot des émissions de CO₂ en fonction du mode de carburation")
st.write("Cette visualisation en boîtes à moustache (boxplot) des émissions de CO₂ en fonction du mode de carburation apporte plusieurs informations complémentaires aux graphiques précédents.")
fig = px.box(x=cars_viz['Fm'], y=cars_viz['Emissions de CO2'], labels={'x':' ', 'y':'Emissions de CO₂ (g/km)', 'color':'Mode de carburation'}, color=cars_viz['Fm'], boxmode="overlay")
st.plotly_chart(fig)

with st.expander("Voir/Cacher la description"):
    st.write("- La catégorie de véhicules fonctionnant avec un unique carburant (“mono-fuel”) présente de nombreuses valeurs au-delà des moustaches : ces valeurs ne sont pas aberrantes et sont associées à des véhicules très puissants (type “supercar”).")
    st.write("- Le corps des boxplots (médianes et quartiles) met en lumière une émission de CO₂ plus élevée pour les véhicules hybrides non rechargeables (“hybrid not off-charged”), comparativement aux véhicules mono-fuel ou bi-fuel.")
    st.write("- Les véhicules de type flex-fuel ont la médiane la plus élevée et les véhicules hybrides rechargeables (“hybrid off-charged”) ont la médiane la plus faible.")

# Création de la variable moy_CO2_mk : Emission moyenne de CO2 par marque de véhicule
moy_CO2_mk = cars_viz['Emissions de CO2'].dropna().groupby(cars_viz['Mk']).mean()

# Graphique 4
st.subheader("Graphique 4 : Diagramme à barres des émissions de CO₂ moyennes selon les marques de véhicule")
st.write("Ce diagramme en barre représente l’émission moyenne de CO₂ en fonction des marques de véhicule du dataset. Les marques de véhicules électriques ont été retirées car ne rejettent pas de CO₂ (ex : Tesla).")
fig = px.bar(x=moy_CO2_mk.index, y=moy_CO2_mk.values, labels={'x':' ', 'y':'Emission de CO₂ moyenne (g/km)', 'color':'CO₂ (g/km)'}, color=moy_CO2_mk.values)
st.plotly_chart(fig)

with st.expander("Voir/Cacher la description"):
    st.write("- La plus grande partie des marques présente un taux d’émission de CO₂ moyen aux alentours de ~140 à ~180 g/km.")
    st.write("- Certains pics sont bien plus importants : les marques Aston Martin, Bentley, Ferrari, Lamborghini, Maserati, McLaren, Mercedes AMG ou encore Rolls-Royce présentent un taux d’émission moyen beaucoup plus important compris entre ~280 et ~380 g/km.")
    st.write("- Deux pics sont très faibles : la marque BMW i (ou BMW I), dispose du plus faible taux d’émission de CO₂ moyen. En effet, BMW i est une filiale du groupe BMW consacrée aux véhicules hybrides et électriques.")

# Graphique 5
st.subheader("Graphique 5 : Corrélation par paire entre les variables de masse, cylindrée, puissance moteur, largeur de voies d’essieux, d’empattement et d’émissions de CO₂")
st.write("Cette carte de chaleur (ou heatmap) établit une corrélation par paire des caractéristiques techniques (variables continues) des véhicules du dataset, par la méthode de Pearson.")

fig = px.imshow(cars_viz[['Emissions de CO2','Puissance moteur','Cylindrée','Masse',
                          'Largeur de voie essieu dir.','Largeur de voie autre essieu',
                          'Empattement']].corr(), text_auto='.2f', aspect='auto', labels={'color':' '})
st.plotly_chart(fig)

with st.expander("Voir/Cacher la description"):
    st.write("- En s’intéressant aux corrélations avec notre variable cible (émissions de CO₂), on s’aperçoit que la relation est modérée voire faible avec les variables d’empattement et de largeur de voies d’essieux (respectivement 0.46 et 0.58).")
    st.write("- La dépendance est plus forte avec la variable de masse des véhicules (0.68), et les variables de puissance moteur et de cylindrée montrent la plus forte dépendance avec la variable cible (respectivement 0.70 et 0.71).")
    st.write("- La dépendance est très importante (0.99) entre les deux variables de largeur de voies d’essieux. Cette relation s’explique notamment par le fait que 84,5% des valeurs de ces variables sont identiques.")

