# import des packages nécessaires à la modélisation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import f1_score, classification_report, accuracy_score
import numba
import streamlit as st

st.set_page_config(page_title="Classification", page_icon="🗂️")
st.sidebar.header("Classification")

# Cacher "Made with Streamlit"
hide_footer_style = """ 
    <style>
    footer {visibility: hidden; }
    </style>
    """
st.markdown(hide_footer_style, unsafe_allow_html=True)

# Cacher les warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# Image CO2
from PIL import Image
image = Image.open('Streamlit/CO2_wide.jpg')
st.image(image)

# Titre
st.title("Machine Learning")

# Partie Régression
st.header("Classification")
st.write("En aval du pre-processing, **les variables catégorielles ont été dichotomisées** pour la modélisation et les variables de masse et d'émission de CO₂ selon la norme NEDC supprimées (pour cette présentation Streamlit, on ne s'intéressera qu'à la norme WLTP afin d'accélérer l'exécution des scripts).")
st.write("La variable d'intérêt d’émission de CO₂ a été découpée en **quartiles** afin de produire des modèles de classification. De cette façon, il n’y a pas de déséquilibre de classes, bien qu’en raison de la **dissymétrie de distribution des valeurs**, les classes intermédiaires (2ème et 3ème quartiles) s’étendent sur une plage étroite de valeurs, ce qui rendra leur détection par le modèle probablement plus difficile.")
st.write("Ainsi, **la variable cible 'Classe_CO2'** est répartie de la façon suivante :")
st.write("- De **3,9 à 143 g/km, classe 1**.")
st.write("- De **143 à 160 g/km, classe 2**.")
st.write("- De **160 à 190 g/km, classe 3**.")
st.write("- De **190 à 499 g/km, classe 4**.")


# Import dataset
cars = pd.read_csv("data/cars_FR2019_clean.csv", dtype={'Cn': 'str', 'IT': 'uint8'},index_col=0)

### Preprocessing 

# Dichotomisation des variables qualitatives sur le jeu de données entier
list_quali  = ['At2supAt1','IT', 'Ft', 'Fm', 'Cr', 'Mp']
list_quanti = ['m', 'Mt', 'At1', 'W', 'ec', 'ep', 'Enedc', 'Ewltp']

# choix de la modalité de référence
cars_dummies = pd.get_dummies(cars[list_quali], columns = list_quali).drop(['At2supAt1_0',
                                                                            'IT_0',
                                                                            'Ft_petrol',
                                                                            'Fm_mono_fuel',
                                                                            'Cr_M1',
                                                                            'Mp_BMW GROUP'],axis=1)
cars_lm = cars[list_quanti]
cars_lm = pd.concat([cars[list_quanti], cars_dummies], axis=1)
cars_lm = cars_lm.drop(['m', 'Enedc'], axis=1)

# Discrétisation des variables d'émission de CO2 en quartile
cars_lm['Classe_CO2'] = pd.qcut(cars_lm['Ewltp'], q=4, labels=[1,2,3,4]).astype('int')
# Suppression de l'ancienne variable
cars_lm = cars_lm.drop(['Ewltp'], axis=1)
    
st.subheader("Affichage des features et target")
    
# Séparation des données
target_wltp = cars_lm['Classe_CO2']
features_wltp = cars_lm.drop(['Classe_CO2'], axis=1)

# Renommer les variables
rename = {'Mt': 'Masse test (kg)',
          'ec': 'Cylindrée en cm3',
          'ep': 'Puissance moteur en KW',
          'Ft': 'Type de carburant',
          'Fm': 'Mode de carburation',
          'Ft_diesel': 'Diesel',
          'Fm_NOVC-HEV': 'Hybride non-rechargeable',
          'Fm_OVC-HEV': 'Hybride rechargeable',
          'IT_1': 'Eco-innovation',
          'W': 'Empattement en mm',
          'At1': "Largeur de voie essieu directeur (mm)",
          'At2supAt1_1': 'Essieu arrière supérieur essieu avant',
          'Cr_M1G': 'Motricité (4x4)'}
cars_lm_rename = cars_lm.rename(rename,axis=1)

# Fonction pour surligner des valeurs dans le dataframe
def color_df(val):
    if val > 0:
        color = '#97FF9A'
    if val > 1:
        color = '#FBFF61'
    if val > 2:
        color = '#FFB24D'
    if val > 3:
        color = '#FF4D4D'
    return f'background-color: {color}'

### Affichage du dataframe
line_to_plot = st.slider("Sélectionnez le nombre de lignes à afficher :", min_value=3, max_value=1000)

if st.sidebar.checkbox("Renommer les variables"):
    st.dataframe(cars_lm_rename.head(line_to_plot).style.applymap(color_df, subset=["Classe_CO2"]))
    features_wltp = features_wltp.rename(rename, axis=1)
else:
    st.dataframe(cars_lm.head(line_to_plot).style.applymap(color_df, subset=["Classe_CO2"]))
    



# Sélection des features
options = st.multiselect('Choisissez les features :', options=features_wltp.columns, default=features_wltp.columns.values)

col1, col2 = st.columns(2)
with col1:
    # Choix du modèle
    model = st.radio(
        "Choisissez un modèle :",
        ('Régression logistique', 'KNN', 'Decision tree', 'Random Forest', 'XGBoost'))
with col2:
    # select how to split the data
    train_size_wltp = st.slider(label = "Choix de la taille de l'échantillon d'entrainement",
                                        min_value = 0.70, max_value = 0.90, step = 0.01)

# Séparation des données en jeu d'entrainement et de test pour la norme WLTP
X_train_wltp, X_test_wltp, y_train_wltp, y_test_wltp = train_test_split(features_wltp[options], target_wltp,
                                                                        train_size=train_size_wltp,
                                                                        random_state=33)
# Normalisation des donnéees norme WLTP
scaler = StandardScaler()
X_train_wltp_scaled = scaler.fit_transform(X_train_wltp)
X_test_wltp_scaled = scaler.transform(X_test_wltp)

if model == 'Régression logistique':
    col1, col2 = st.columns(2)
    with col1:
        penalty = st.selectbox(
    'Choisissez un paramètre de régularisation :',
    ('l2', 'none'))
    with col2:
        solver = st.selectbox(
    'Choisissez un solver :',
    ('lbfgs', 'sag', 'saga', 'newton-cg'))
    st.subheader("Performances du modèle de régression logistique")
    lr = LogisticRegression(max_iter=500, penalty=penalty, solver=solver)
    lr.fit(X_train_wltp_scaled, y_train_wltp)
    # Affichage des performances du modèle
    st.write("  - Score du modèle sur le jeu d'entraînement :", lr.score(X_train_wltp_scaled, y_train_wltp).round(3))
    st.write("  - Score du modèle sur le jeu de test :", lr.score(X_test_wltp_scaled, y_test_wltp).round(3))
    y_pred_train = lr.predict(X_train_wltp_scaled)
    y_pred_test = lr.predict(X_test_wltp_scaled)
    st.markdown("##### Matrice de confusion")
    st.write(pd.crosstab(y_test_wltp, y_pred_test, colnames=['Predictions']))

if model == 'KNN':
    col1, col2 = st.columns(2)
    with col1:
        k = st.number_input('Sélectionnez un nombre de voisins (max = 8) :', min_value=1, max_value=8)
    with col2:
        metric = st.selectbox(
    'Choisissez une métrique de distance :',
    ('minkowski', 'manhattan', 'chebyshev'))
    st.subheader("Performances du modèle des K plus proches voisins")
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    knn.fit(X_train_wltp_scaled, y_train_wltp)
    # Affichage des performances du modèle
    st.write("  - Score du modèle sur le jeu d'entraînement :", knn.score(X_train_wltp_scaled, y_train_wltp).round(3))
    st.write("  - Score du modèle sur le jeu de test :", knn.score(X_test_wltp_scaled, y_test_wltp).round(3))
    y_pred_train = knn.predict(X_train_wltp_scaled)
    y_pred_test = knn.predict(X_test_wltp_scaled)
    st.markdown("##### Matrice de confusion")
    st.write(pd.crosstab(y_test_wltp, y_pred_test, colnames=['Predictions']))
    
    # Affichage du score en fonction du nombre de voisins
    score = []

    for k in range(1, 9):
        knn = KNeighborsClassifier(n_neighbors = k, metric = metric)
        knn.fit(X_train_wltp_scaled, y_train_wltp)
        score.append(knn.score(X_test_wltp_scaled, y_test_wltp))
    st.markdown("##### Affichage du score en fonction de la métrique choisie et du nombre de voisins")
    sns.set_style('darkgrid')
    plt.figure(figsize=(9,4))
    plt.plot(range(1, 9), score, 'r--', label = metric)
    plt.xlabel('Nombre de voisins')
    plt.ylabel('Score')
    plt.legend()
    st.pyplot()

if model == 'Decision tree':
    col1, col2, col3 = st.columns(3)
    with col1:
        max_depth = st.selectbox(
    "Choisissez la profondeur maximale de l'arbre :",
    (3,5,10,15,20,30,None))
    with col2:
        min_samples_split = st.selectbox(
    "Choisissez le nombre d'observations requis pour diviser un nœud :",
    (2,5,7,10))
    with col3:
        min_samples_leaf = st.selectbox(
    "Choisissez le nombre d'observations qu'un nœud enfant doit avoir :",
    (1,2,5))
    st.subheader("Performances du modèle d'arbre de décisions")
    dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    dt.fit(X_train_wltp, y_train_wltp)
    # Affichage des performances du modèle
    st.write("  - Score du modèle sur le jeu d'entraînement :", dt.score(X_train_wltp, y_train_wltp).round(3))
    st.write("  - Score du modèle sur le jeu de test :", dt.score(X_test_wltp, y_test_wltp).round(3))
    y_pred_train = dt.predict(X_train_wltp)
    y_pred_test = dt.predict(X_test_wltp)
    st.markdown("##### Matrice de confusion")
    st.write(pd.crosstab(y_test_wltp, y_pred_test, colnames=['Predictions']))
    # Visualisation de l'arbre de décision avec critère gini
    st.markdown("##### Affichage de l'arbre")
    plt.figure(figsize=(22,22))
    plot_tree(dt, max_depth=2, feature_names=features_wltp.columns, filled=True, fontsize=16)
    st.pyplot()
    
if model == 'Random Forest':
    col1, col2, col3 = st.columns(3)
    with col1:
        max_depth = st.selectbox(
    "Choisissez la profondeur maximale des arbres :",
    (3,5,10,15,20,30,None))
    with col2:
        min_samples_split = st.selectbox(
    "Choisissez le nombre d'observations requis pour diviser un nœud :",
    (2,5,7,10))
    with col3:
        min_samples_leaf = st.selectbox(
    "Choisissez le nombre d'observations qu'un nœud enfant doit avoir :",
    (1,2,5))
    st.subheader("Performances du modèle de forêts aléatoires")
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train_wltp, y_train_wltp)
    # Affichage des performances du modèle
    st.write("  - Score du modèle sur le jeu d'entraînement :", rf.score(X_train_wltp, y_train_wltp).round(3))
    st.write("  - Score du modèle sur le jeu de test :", rf.score(X_test_wltp, y_test_wltp).round(3))
    y_pred_train = rf.predict(X_train_wltp)
    y_pred_test = rf.predict(X_test_wltp)
    st.markdown("##### Matrice de confusion")
    st.write(pd.crosstab(y_test_wltp, y_pred_test, colnames=['Predictions']))
    
if model == 'XGBoost':
    st.subheader("Performances du modèle XGBoost")
    st.write("En raison de la complexité d'optimisation des hyperparamètres du modèle XGBoost et du temps d'exécution de celui-ci, le modèle est pré-configuré avec les meilleurs paramètres sélectionnés via optimisation bayésienne (librairie Hyperopt).")
    # Préparation des données pour XGBoost (WLTP)
    train_wltp = xgb.DMatrix(X_train_wltp, y_train_wltp)
    test_wltp = xgb.DMatrix(X_test_wltp, y_test_wltp)
    params = {'objective': 'multi:softmax', 'eval_metric' : 'mlogloss', 'num_class': 5, 'colsample_bytree': 1.0,
              'gamma': 0.19,'learning_rate': 0.2, 'max_depth': 13, 'min_child_weight': 5.0, 'reg_alpha': 1.0,
              'reg_lambda': 2.3000000000000003, 'subsample': 0.9}
    # Entrainement du modèle XGBoost optimisé (WLTP)
    xgb4 = xgb.train(params = params, dtrain = train_wltp, num_boost_round = 600, early_stopping_rounds= 15,
                     evals= [(train_wltp, 'train'), (test_wltp, 'eval')])

    # Affichage des performances du modèle optimisé (WLTP)
    y_pred_train_wltp = xgb4.predict(train_wltp)
    y_pred_test_wltp = xgb4.predict(test_wltp)
    st.markdown("##### Rapport de classification sur le jeu d'entraînement")
    st.text(classification_report(y_train_wltp, y_pred_train_wltp))
    st.markdown("##### Rapport de classification sur le jeu de test")
    st.text(classification_report(y_test_wltp, y_pred_test_wltp))
    st.markdown("##### Matrice de confusion")
    st.write(pd.crosstab(y_test_wltp, y_pred_test_wltp, colnames=['Predictions']))
    
    # Valeurs de Shapley
    st.markdown("##### Affichage des valeurs de Shap")
    explainer = shap.TreeExplainer(xgb4)
    shap_values = explainer.shap_values(X_test_wltp)

    plt.figure(figsize=(8,8))
    plt.subplot(1,1,1)
    shap.summary_plot(shap_values, X_test_wltp, plot_type="bar", plot_size=None, show=True)
    st.pyplot()
