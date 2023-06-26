# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 11:09:49 2023

@author: PatriciaWintrebert
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from functions import axes_figure, title_filtering, reduction, train_model, display_crosstab, variance_graph, grid_search_model, get_param_grid, train_best_model,grid_search_params
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import warnings

# Ignorer les avertissements
warnings.filterwarnings("ignore")

#import plotly_express as px

#st.markdown("# Exploration des données")
#st.sidebar.markdown("# Exploration des données")

st.title('Projet DataJob')

df = pd.read_csv('kaggle_survey_2020_responses.csv')

pages = ['Exploration des données', 
         'Data Visualisation',
         'Modélisation : Méthode supervisée', 
         'Modélisation : Méthode non supervisée',
         'Conclusion']

st.markdown("[Lien vers le rapport écrit ](https://docs.google.com/document/d/1DLS5DsbR-z5cnq5FYZIlrufrJUUiUxgFgHqk9vBGz2c/edit?usp=sharing')")
st.markdown("[Lien vers les données sur le site de Kaggle](https://www.kaggle.com/c/kaggle-survey-2020/overview')")

st.sidebar.title("Sommaire")
page = st.sidebar.radio('Aller vers', pages)

 #On vire la première ligne 
df = df.drop(df.index[0])

#On renomme la colonne Duration
df = df.rename(columns={"Time from Start to Finish (seconds)": "Duration"})
#Conversion duration en int
df['Duration'].astype(int)
#Ajout colonne DurationMin convertie en minutes
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
df['DurationMin'] = df['Duration'] / 60

#nombre de lignes en dessous de 2 
duration_threshold = 2  # Threshold in minutes
df_minvalues = df[df['DurationMin'] < duration_threshold]

#Création du nouveau df dépourvu de ces valeurs aberrantes
df_new = df.drop(df_minvalues.index)
df_new.head()

#Création du nouveau df dépourvu des gens qui ne programment pas
df_new = df_new[df_new['Q6'] != "I have never written code"]

#On vire toutes les lignes dont la Q5 (titre) est vide
df_new = df_new.dropna(subset=['Q5'])

#On vire les colonnes correspondant aux question "dans deux ans" 
#On les met dans un dataframe pour plus tard 
df_future = df_new.iloc[:, 256:356]

#On les efface de df_new
df_new = df_new.drop(columns=df_future.columns)

#On vire la colonne duration devenue inutile
df_new.drop('Duration', axis = 1, inplace = True)

#df_new.shape

#FIN NETTOYAGE

if page == pages[0]:
    #st.image('titanic.jpg')
    st.subheader('Exploration des données')    
    st.dataframe(df.head())
    
    if st.checkbox("Afficher les valeurs manquantes"):
        st.dataframe(df.isna().sum())

elif page == pages[1]:
    
    st.subheader('Data Vizualisation')    
    
    fig = plt.figure()
    sns.countplot(x = "Q1", data = df)
    st.pyplot(fig)
    
    fig = plt.figure()
    sns.countplot(x = "Q6", data = df)
    st.pyplot(fig)
    
elif page == pages[2]:
    
    st.subheader('Modélisation : Méthode supervisée')

    #Conversion des valeurs vides des colonnes contenant les sous questions (identifiées par "_" en 0 et des valeurs existantes en 1)
    for column in df_new.columns:
        if "_" in column:
            df_new[column] = np.where(df_new[column].fillna('') != '', 1, 0)

    #On exclue les lignes contenant les vaariables cibles qui ne nous intéressent pas 
    #On réserve le dataframe de côté pour une éventuelle application plus tard

    # On garde de côté un df contenant ces valeurs
    excluded_values = ["Student", "Other", "Currently not employed"]
    df_backup = df_new[df_new['Q5'].isin(excluded_values)]
    
    #On vire les lignes qui ne nous concernent pas du dataframe
    df_new = df_new.drop(df_new[df_new['Q5'].isin(excluded_values)].index)  
    
    #st.dataframe(df_new.head(20))
    #st.dataframe(df_backup.head(20))
    
    #Input radio de filtrage de la variable cible
    filter_title = st.radio(label = "Filtrage des Variables cibles", 
                                options = ['Oui', 'Non'])  

    df_new['Q5'] = title_filtering(filter_title, df_new['Q5'])
    
    #Check des valeurs de Q5 filtrées
    st.write(df_new['Q5'].value_counts())
    
    
    #Séparation et Encodage variable cible
    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(df_new['Q5'])

    #Séparation des Variables explicatives
    feats = df_new.drop('Q5', axis = 1)

    X_train, X_test, y_train, y_test = train_test_split(feats, target , test_size=0.25, random_state=42)

    #On isole les variables cat des deux jeux
    df_cat_train = X_train.select_dtypes(include='object')
    df_cat_test = X_test.select_dtypes(include='object')

    #Je vire les variables catégorielles des jeux pour les traiter
    df_new_train = X_train.drop(df_cat_train.columns, axis=1)
    df_new_test = X_test.drop(df_cat_train.columns, axis=1)

    #Encodage des variables catégorielles
    df_encoded_train = pd.get_dummies(df_cat_train)
    df_encoded_test = pd.get_dummies(df_cat_test)

    #On fusionne les df après reset des index
    df_new_train = df_new_train.reset_index(drop=True)
    df_encoded_train = df_encoded_train.reset_index(drop=True)
    X_train = pd.concat([df_encoded_train, df_new_train], axis=1)

    df_new_test = df_new_test.reset_index(drop=True)
    df_encoded_test = df_encoded_test.reset_index(drop=True)
    X_test = pd.concat([df_encoded_test, df_new_test], axis=1)
    
    #Checkpoint
    #st.write("Format de X_test: ", X_test.shape)
    #st.write("Format de X_train: ", X_train.shape)
    #st.write("On constate que le train_test_split a laissé une colonne surnuméraire dans un des jeux")
    
    #Détection de la colonne en trop
    #extra_column = None
    #for column in X_train.columns:
        #if column not in X_test.columns:
            #extra_column = column
            #break
    #if extra_column is not None:
        #st.write("La colonne surnuméraire dans X_train est :", extra_column, "Nous choissons de la supprimer.")
    #else:
        #st.write("Il n'y a pas de colonne surnuméraire dans X_train.")
    
    #Elimination de la colonne en trop
    X_train = X_train.drop("Q32_Domo", axis=1)
    
    #Normalisation des données
    scaler = StandardScaler() # Création de l'instance StandardScaler
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    
    #Selectbox Choix de la méthode de réduction
    reduction_choice = st.selectbox(label = "Choix de la méthode de réduction", 
                                options = ['PCA', 'LDA'])    
    
    #Appel fonction réduction de données
    X_train_reduced, X_test_reduced, reduction = reduction(reduction_choice, X_train_scaled, y_train, X_test_scaled)
    
    #Selectbox avec Choix du modèle
    model_choisi = st.selectbox(label = "Choix du modèle", 
                                options = ['Régression logistique', 'Arbre de décision', 'KNN', 'Forêt aléatoire'])    
    
    #Appel fonction d'entrainement du modèle
    score, model = train_model(model_choisi, X_train_reduced, y_train, X_test_reduced, y_test)
    #model = train_model(model_choisi)
    
    #Affichage du score
    st.write("Score :", score)
       

    #Affichage tableau prédiction vs réalité
    #st.write("Comparaison prédictions vs réalité :")
    #st.write(display_crosstab(model, X_test_reduced, y_test)[0])
    #st.dataframe(display_crosstab(model, X_test_reduced, y_test)[0])
    #Affichage rapport de classification
    #st.write("Rapport de classification :")
    #st.text(display_crosstab(model, X_test_reduced, y_test)[1])
        
    #Affichage graphique axes
    #if model_choisi != "K-means Clustering": 
        #axes_figure(X_train_reduced, y_train, reduction_choice) 

    #Affichage graphique variance expliquée
    #if reduction_choice == 'PCA':
        #variance_graph(reduction) 
        
    st.subheader("Recherche des meilleurs hyperparamètres :")
    
    search_param = st.button("Lancer l'optimisation")
    reset_button = st.button("Réinitialiser")
    
    if reset_button:
        search_param = False
        reset_button = False
        
    if search_param:
       
       #Récupération des paramètres possible pour chsque modèle
       #param_grid, model = get_param_grid(model_choisi)
       
       # Utilisation de la fonction grid_search_model 
       #best_model, y_test, y_pred, best_params = grid_search_model(model, param_grid, X_train_reduced, X_test_reduced, y_train, y_test)
       
       #score_best_model = train_best_model(model, param_grid, X_train_reduced, X_test_reduced, y_train, y_test)
       
       param_grid = {
           'C': [0.1, 1.0, 10.0],
           'penalty': ['l1', 'l2']
           }


       # Créer le modèle de régression logistique
       model = LogisticRegression()
       grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
       grid_search.fit(X_train_reduced, y_train)
       best_model = grid_search.best_estimator_
       y_pred = best_model.predict(X_test_reduced)
       st.write(classification_report(y_test, y_pred))
       score = best_model.score(X_test_reduced, y_test)
       st.write("Score : ", score)


elif page == pages[3]:
    
    st.subheader('Modélisation : Méthode non supervisée')
    
    #Selectbox avec Choix du modèle
    methode_choisie = st.selectbox(label = "Choix du modèle", 
                                options = ['t-SNE', 'K-means'])    
    
    #Appel fonction d'entrainement du modèle
    score, model = train_model(methode_choisie, X_train_reduced, y_train, X_test_reduced, y_test)
    #model = train_model(model_choisi)
    
    #Affichage du score
    
elif page == pages[4]:  
    st.subheader('Conclusion')