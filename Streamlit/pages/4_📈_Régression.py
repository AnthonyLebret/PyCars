import pandas as pd 
import seaborn as sns 
import streamlit as st 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import statsmodels.api
import statsmodels.api as sm
import scipy.stats as stats
import xgboost as xgb
import shap
from scipy.stats import norm, ttest_ind
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor, plot_importance

st.set_page_config(page_title="Régression", page_icon="📈")
st.sidebar.header("Régression")

# Cacher "Made with Streamlit"
hide_footer_style = """ 
    <style>
    footer {visibility: hidden; }
    </style>
    """
st.markdown(hide_footer_style, unsafe_allow_html=True)

# Image CO2
from PIL import Image
image = Image.open('CO2_wide.jpg')
st.image(image)

# Titre
st.title("Machine Learning")
    
# Partie Régression
st.header("Régression")
st.write("En aval du pre-processing, **les variables catégorielles ont été dichotomisées** pour la modélisation et les variables de masse et d'émission de CO₂ selon la norme NEDC supprimées (pour cette présentation Streamlit, on ne s'intéressera qu'à la norme WLTP afin d'accélérer l'exécution des scripts).")

# Import dataset
cars = pd.read_csv("C:/Users/lebre/Documents/Jupyter Notebook/Projet CO2/Data/cars_FR2019_clean.csv", index_col=0)


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
    
st.subheader("Affichage des features et target")
    
# Séparation des données
target_wltp = cars['Ewltp']
feats_wltp = cars_lm.drop(['Ewltp'], axis=1)

# Renommer les variables
rename = {'Mt': 'Masse test (kg)',
          'ec': 'Cylindrée en cm3',
          'ep': 'Puissance moteur en KW',
          'Ewltp': 'Emissions de CO2 en g/km',
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
    if val > 143:
        color = '#FBFF61'
    if val > 160:
        color = '#FFB24D'
    if val > 190:
        color = '#FF4D4D'
    return f'background-color: {color}'

### Affichage du dataframe
line_to_plot = st.slider("Sélectionnez le nombre de lignes à afficher", min_value=3, max_value=1000)

if st.sidebar.checkbox("Renommer les variables"):
    st.dataframe(cars_lm_rename.head(line_to_plot).style.applymap(color_df, subset=['Emissions de CO2 en g/km']))
    feats_wltp = feats_wltp.rename(rename, axis=1)
else:
    st.dataframe(cars_lm.head(line_to_plot).style.applymap(color_df, subset=['Ewltp']))

# Sélection des features
options = st.multiselect('Choisissez les features :', options=feats_wltp.columns, default=feats_wltp.columns.values)
    

col1, col2 = st.columns(2)
with col1:
    # Choix du modèle
    model = st.radio(
        "Choisissez un modèle :",
        ('Régression linéaire', 'Régression linéaire régularisée (ElasticNet)', 'XGBoost'))
with col2:
    # select how to split the data
    train_size_wltp = st.slider(label = "Choix de la taille de l'échantillon d'entrainement",
                                        min_value = 0.70, max_value = 0.90, step = 0.01)

# Séparation des données
X_wltp_train, X_wltp_test, y_wltp_train, y_wltp_test = train_test_split(feats_wltp[options], target_wltp,
                                                                        train_size=train_size_wltp,
                                                                        random_state=33)
# normalisation des données
sc_wltp = preprocessing.StandardScaler()
X_wltp_scaled_train = pd.DataFrame(sc_wltp.fit_transform(X_wltp_train),
                                   columns = X_wltp_train.columns,
                                   index = X_wltp_train.index)
X_wltp_scaled_test = pd.DataFrame(sc_wltp.transform(X_wltp_test),
                                  columns = X_wltp_test.columns,
                                  index = X_wltp_test.index)

# Régression linéaire
if model == 'Régression linéaire':
    st.subheader("Régression linéaire")
    st.write("Un modèle de base a été produit à partir d’une régression linéaire multiple **sans sélection de variables et sans paramétrage particulier**. Les hypothèses de normalité et d’homoscédasticité des résidus ont été testées sur ce modèle de base.")

    lr_wltp = LinearRegression()
    lr_wltp.fit(X_wltp_scaled_train, y_wltp_train)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write('  - R² train :', lr_wltp.score(X_wltp_scaled_train, y_wltp_train).round(3))
        st.write('  - R² test :', lr_wltp.score(X_wltp_scaled_test,y_wltp_test).round(3))
    lr_wltp_pred_train = lr_wltp.predict(X_wltp_scaled_train)
    lr_wltp_pred_test = lr_wltp.predict(X_wltp_scaled_test)
    with col2:
        st.write('  - RMSE train :', np.sqrt(mean_squared_error(lr_wltp_pred_train, y_wltp_train)).round(3))
        st.write('  - RMSE test :', np.sqrt(mean_squared_error(lr_wltp_pred_test, y_wltp_test)).round(3))
    with col3:
        st.write('  - MAE train :', mean_absolute_error(lr_wltp_pred_train, y_wltp_train).round(3))
        st.write('  - MAE test :', mean_absolute_error(lr_wltp_pred_test, y_wltp_test).round(3))
    
    st.write("Les performances globales du modèle sont **élevées** : le coefficient de détermination est aux alentours de **0,89** sur les données d'entraînement et de test (faire varier la taille du jeu d'entraînement pour obtenir la meilleure performance possible). **Le modèle ne semble pas souffrir de sur-apprentissage** avec des écarts quadratiques moyens (RMSE) relativement faibles entre les jeux d'entraînement et de test.")
    
    
    
    ## points influents et significativité des coefficients avec statsmodel 
    y = y_wltp_train
    x = X_wltp_scaled_train

    # entrainement du modèle de régression linéaire
    model_wltp = sm.OLS(y, sm.add_constant(x, prepend=False)).fit()
    print(model_wltp.summary())

    
    # Analyse des résidus
    st.write('\n')
    st.markdown("#### Analyse des résidus : Normalité et homoscédasticité")
    st.write("Si l’hypothèse de normalité peut être retenue, la variabilité des résidus est clairement dépendante des valeurs d’émission de CO2 avec une variabilité plus élevée pour les valeurs hautes et basses.")
    
    ## histogramme
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig = plt.figure(figsize=(7,5.2))
    plt.hist(model_wltp.resid, bins=50)
    plt.title('normalité - histogramme\n')

    ## Normalité
    # les tests de Jarques Bera et Omnibus produits dans les résultats indiquent que l'hypothèse de normalité est rejettée (p<0.5)
    # le QQ plot permet une représentation visuelle qui montre toutefois que les résultats sont corrects

    residus_std = model_wltp.resid_pearson # résidus standardisés
    fig2 = sm.qqplot(residus_std, stats.t, fit=True,line='45')
    
    plt.xlabel("Quantiles théoriques")
    plt.ylabel("Quantiles observés")

    plt.title("normalité - QQ plot\n")
    
    col1, col2 = st.columns(2)
    col1.pyplot(fig, use_container_width=True)
    col2.pyplot(fig2, use_container_width=True)
    
        
    ## Homoscédasticité
    st.write("Le test de Breush-Pagan confirme que l’hypothèse d’homoscédasticité ne peut être retenue. L’allure du nuage de points indique que le modèle prédit moins bien les taux d’émission des véhicules avec des faibles ou des fortes valeurs.")
    
    lagrande, pval, f, fpval = statsmodels.stats.diagnostic.het_breuschpagan(model_wltp.resid, model_wltp.model.exog)
    print("Test d'homoscédasticité des résidus (Breush-Pagan)")
    print('f-statistic', f.round(6), '\np-value for the f-statistic', fpval)
    plt.figure(figsize=(7,3))
    plt.scatter(y,residus_std, s=2)

    plt.title("homoscédasticité\n")
    plt.xlabel("Emission de CO2")
    plt.ylabel("Résidus")
    st.pyplot()


    # analyse graphique de l'influence à partir de la distance de Cook
    st.markdown("#### Analyse de l'influence grâce à la distance de Cook")
    st.write("La représentation des distances de Cook permet d’identifier trois véhicules particulièrement atypiques : il s’agit de trois différentes versions de la Lamborghini Avantador, un véhicule aux caractéristiques techniques effectivement hors norme. La régression linéaire étant très sensible aux valeurs extrêmes, ces véhicules ont été exclus du dataset d'entraînement. Néanmoins, on note que l’influence de certains véhicules dans l’estimation des coefficients de régression reste très marquée.")
    
    influence = model_wltp.get_influence()
    (c, p) = influence.cooks_distance # c : distance et p : p-value
    plt.figure(figsize=(7,2))
    plt.stem(np.arange(len(c)), c, markerfmt=",")
    plt.ylim(0,0.12)
    # plt.axhline(y = 3*c.mean(),color="r",linewidth=1)

    plt.title('Points influents : Dataset complet')
    plt.ylabel('Distance de Cook')
    st.pyplot()

    # visualisation des 3 points très influents
    df_cooks = pd.DataFrame(c, columns = ['cook'], index = X_wltp_scaled_train.index)
    ind = df_cooks.loc[df_cooks['cook']>0.08].index
    cars[['Mh','Cn', 'Ewltp', 'ec', 'ep', 'Mt']].loc[ind]
    
    # la Lamborgini Aventador est effectivement une voiture avec des caratéristiques atypiques

    # modèle sans les 3 points influents
    x = X_wltp_scaled_train.drop(X_wltp_scaled_train.loc[ind].index, axis=0)
    y = y_wltp_train.drop(y_wltp_train.loc[ind].index, axis=0)

    # entrainement du modèle de régression linéaire&é
    model_wltp = sm.OLS(y, sm.add_constant(x, prepend=False)).fit()
    print(model_wltp.summary())

    # analyse graphique de l'influence à partir de la distance de Cook
    influence = model_wltp.get_influence()
    (c, p) = influence.cooks_distance # c : distance et p : p-value

    plt.stem(np.arange(len(c)), c, markerfmt=",")
    plt.ylim(0,0.03)
    # plt.axhline(y = 3*c.mean(),color="r",linewidth=1)

    plt.title('Points influents : exclusion des Lamborghini Aventador')
    plt.ylabel('Distance de Cook')
    st.pyplot()

if model == 'Régression linéaire régularisée (ElasticNet)':
    ## points influents et significativité des coefficients avec statsmodel 
    y = y_wltp_train
    x = X_wltp_scaled_train

    # entrainement du modèle de régression linéaire
    model_wltp = sm.OLS(y, sm.add_constant(x, prepend=False)).fit()
    print(model_wltp.summary())
    
    influence = model_wltp.get_influence()
    (c, p) = influence.cooks_distance # c : distance et p : p-value
    
    # visualisation des 3 points très influents
    df_cooks = pd.DataFrame(c, columns = ['cook'], index = X_wltp_scaled_train.index)
    ind = df_cooks.loc[df_cooks['cook']>0.08].index


    # modèle sans les 3 points influents
    x = X_wltp_scaled_train.drop(X_wltp_scaled_train.loc[ind].index, axis=0)
    y = y_wltp_train.drop(y_wltp_train.loc[ind].index, axis=0)
    

    st.subheader('Régression linéaire régularisée - ElasticNet')
    st.write("Compte tenu des corrélations fortes observées entre certaines caractéristiques techniques des véhicules, une régularisation est appliquée. Un modèle Elastic Net est implémenté avec une optimisation des paramètres de régularisation à partir d’une méthode grid search.")
    
    ## Recherche des paramètres les plus explicatifs avec une régression ElasticNet
    ## reprise des données sans les 3 points influents
    x = X_wltp_scaled_train.drop(X_wltp_scaled_train.loc[ind].index, axis=0)
    y = y_wltp_train.drop(y_wltp_train.loc[ind].index, axis=0)

    model_en = ElasticNetCV(cv=10,
                            l1_ratio = (0.1, 0.25, 0.5, 0.75, 0.99),
                            n_alphas = 200,
                            random_state=33)

    model_en.fit(x, y)
    st.write('Alpha sélectionné par validation croisée :', model_en.alpha_.round(4), 'l1_ratio sélectionné par validation croisée :', model_en.l1_ratio_)

    pred_train = model_en.predict(x)
    pred_test = model_en.predict(X_wltp_scaled_test)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write('  - R² train :', model_en.score(x, y).round(4))
        st.write('  - R² test :', model_en.score(X_wltp_scaled_test, y_wltp_test).round(4))
    with col2:
        st.write('  - RMSE train :', np.sqrt(mean_squared_error(y, pred_train)).round(3))
        st.write('  - RMSE test :', np.sqrt(mean_squared_error(y_wltp_test, pred_test)).round(3))
    with col3:
        st.write('  - MAE train :', mean_absolute_error(pred_train, y).round(3))
        st.write('  - MAE test :', mean_absolute_error(pred_test, y_wltp_test).round(3))
    
    # Coefficients
    lr_wltp = LinearRegression()
    lr_wltp.fit(X_wltp_scaled_train, y_wltp_train)
    st.markdown("#### Comparaison des coefficients de régression entre le modèle simple et régularisé")
    coeffs = pd.DataFrame({'modèle simple': lr_wltp.coef_}, index=feats_wltp[options].columns) # modèle linéaire
    coeffs = coeffs.join(pd.DataFrame({'modèle régularisé': model_en.coef_}, index=feats_wltp[options].columns)) # modèle régularisé
    coeffs.sort_values(by='modèle simple', inplace=True)

    coeffs.plot.barh(figsize=(14, 5))
    plt.subplots_adjust(left=0.3)
    plt.title("Coefficents de régression linéaire\n")
    st.pyplot()
    
    st.write("La représentation graphique des coefficients obtenus pour le modèle simple et le modèle régularisé montre à la fois qu’aucun paramètre n’est exclu du modèle, et que les valeurs des coefficients restent très proches. Néanmoins, le modèle pénalisé permet bien de diminuer la différence des RMSE entre données d'entraînement et données test.")

if model == 'XGBoost':
    st.subheader("XGBoost")
    st.write("XGBoost est un modèle d’ensemble très utilisé pour ses qualités prédictives mais comprend un nombre très important d’hyperparamètres à optimiser.")
    st.write("Une méthode d'optimisation séquentielle des paramètres a été adoptée : les paramètres sont optimisés de façon isolée ou par paire, les modèles suivants intégrant les paramètres sélectionnés dans les étapes précédentes.")

    with st.expander("Afficher les paramètres sélectionnés après optimisation"):
        st.success("objective = 'regsquarederror',  eval_metric = 'rmse',  learning_rate = 0.005,  n_estimators = 6000,  max_depth = 11, min_child_weight = 3,  gamma = 20,  subsample = 0.8,  colsample_bytree = 1,  reg_alpha = 0.35,  reg_lambda = 1.8, random_state = 1")
        
    st.write('  - Coefficient de détermination du modèle :', 0.994)
    st.write('  - Coefficient de détermination des données test :', 0.98)
    st.write('  - RMSE train :', 3.07)
    st.write('  - RMSE test :', 5.5)
    st.write('  - MAE train :', 2.24)
    st.write('  - MAE test :', 3.79)
    
    st.warning("En raison d'un temps d'exécution très long, le modèle a été exécuté en amont et les scores présentés sont donc figés.")
    
    st.markdown("#### Importance des variables selon les valeurs de Shap")
    
    from PIL import Image
    shap = Image.open('shap_regression.png')
    st.image(shap)