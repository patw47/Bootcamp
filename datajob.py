# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 11:09:49 2023

@author: PatriciaWintrebert
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from main_functions import nettoyage
from modeling_functions import reduction, variance_graph, train_supervised_model, display_crosstab, select_best_model, train_non_supervised_model, search_clusters, display_clusters
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import warnings

# Ignorer les avertissements
warnings.filterwarnings("ignore")

st.title('Projet DataJob')

# Initialisation des données dans st.session_state
#if 'df' not in st.session_state:
   # st.session_state.df = None

#if 'df_new' not in st.session_state:
    #st.session_state.df_new = None

# Chargement des données si elles n'ont pas déjà été chargées
#if st.session_state.df is None:
#st.session_state.df = pd.read_csv('kaggle_survey_2020_responses.csv')
df = pd.read_csv('kaggle_survey_2020_responses.csv')
st.session_state.df = df

pages = ['Exploration des données', 
         'Data Visualisation',
         'Modélisation Supervisée',
         'Modélisation non supervisée'
         ]

st.markdown("[Lien vers le rapport écrit ](https://docs.google.com/document/d/1DLS5DsbR-z5cnq5FYZIlrufrJUUiUxgFgHqk9vBGz2c/edit?usp=sharing')")
st.markdown("[Lien vers les données sur le site de Kaggle](https://www.kaggle.com/c/kaggle-survey-2020/overview')")
st.markdown("[Lien GitHub](https://github.com/patw47/Bootcamp)")
st.sidebar.title("Sommaire")
page = st.sidebar.radio('Aller vers', pages)

if page == pages[0]:
    #st.image('titanic.jpg')
    st.subheader('Nettoyage et Préparation des données')   
    
    st.write("Dataframe original")
    st.dataframe(df.head())

    if st.checkbox("Afficher les valeurs manquantes"):
        st.dataframe(df.isna().sum())
        st.write("On constate énormément de valeurs manquantes sur les dernières questions")
        st.write("Total valeurs manquantes :", df.isna().sum().sum())
        
        st.markdown("---")

    st.subheader("Préparation des données en 9 étapes : ")
    st.write("1.Suppression première ligne")
    st.write("2.Supression lignes dont la valeur duration est inf à 2 min")
    st.write("3.Supression des lignes où la réponse à Q6 est I have never written Code")
    st.write("4.Supression des colonnes avec questions Projection dans deux ans")
    st.write("5.Regroupement par famille de métiers")
    st.write("6.Supression de colonnes peu pertinentes avec bcp de valeurs manquantes (Q11, Q13, Q15, Q20, Q21, Q22, Q24, Q25")
    st.write("7.Supression des colonnes avec taux de réponse inférieur à 50%")
    st.write("8.Remplacement des valeurs par 0 ou 1 pour les colonnes contenant des réponses binaires")
    st.write("9.Remplacement des dernières valeurs vides par leur moyenne")
    
    #Récupération var df_new dans la session
    df_new = nettoyage(df)
    st.session_state.df_new = df_new
    
    st.markdown("---")
    #Check des valeurs de Q5 filtrées
    
    col1, col2 = st.columns(2)

    col1.write("Variables cibles avant regroupement et nettoyage")
    col1.write(df['Q5'].value_counts())

    col2.write("Variables cibles après regroupement")
    col2.write(df_new['Q5'].value_counts())
    
    st.markdown("---")
    
    st.write("Dataframe nettoyé")
    st.dataframe(df_new.head())

    if st.checkbox("Afficher les valeurs manquantes après nettoyage"):
        st.dataframe(df_new.isna().sum())
        st.write("Total valeurs manquantes :", df_new.isna().sum().sum())
      
elif page == pages[1]:
    
    st.subheader('Data Vizualisation')  
    
    #Récupération var df_new dans la session
    df_new = st.session_state.df_new
    
    fig = plt.figure()
    sns.countplot(x = "Q1", data = df_new)
    st.pyplot(fig)
    
    fig = plt.figure()
    sns.countplot(x = "Q6", data = df_new)
    st.pyplot(fig)
    

elif page == pages[2]:
     
    st.subheader("Encodage des données en 4 étapes : ")
    st.write("1.Séparation variable cible et encodage avec LabelEncoder")
    st.write("2.Séparation jeu de test et jeu d'entrainement")
    st.write("3.Encodage des variables catégorielles avec Getdummies")
    st.write("4.Réduction au choix avec PCA ou LDA")
    
    #Récupération var df_new dans la session
    df_new = st.session_state.df_new
    
    #Séparation et Encodage variable cible
    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(df_new['Q5'])

    #Séparation des Variables explicatives
    feats = df_new.drop('Q5', axis = 1)
    
    X_train, X_test, y_train, y_test = train_test_split(feats, target , test_size=0.21, random_state=42)

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
    st.write("Format de X_test après processing: ", X_test.shape)
    st.write("Format de X_train après processing: ", X_train.shape)
    #st.write("On constate que le train_test_split a laissé une colonne surnuméraire dans un des jeux")
    
    #Normalisation des données
    #scaler = StandardScaler() # Création de l'instance StandardScaler
    #X_train_scaled = scaler.fit_transform(X_train)
    #X_test_scaled = scaler.transform(X_test)
    X_train_scaled = X_train
    X_test_scaled = X_test
    
    st.session_state.X_train_scaled = X_train_scaled
    st.session_state.X_test_scaled = X_test_scaled
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    
    st.subheader('Modélisation : Méthode supervisée')
    
    #Selectbox Choix de la méthode de réduction
    reduction_choice = st.selectbox(label = "Choix de la méthode de réduction", 
                                options = ['PCA', 'LDA'])    
    
    #Appel fonction réduction de données
    X_train_reduced, X_test_reduced, reduction = reduction(reduction_choice, X_train_scaled, y_train, X_test_scaled)
    
    
    #Selectbox avec Choix du modèle
    model_choisi = st.selectbox(label = "Choix du modèle", 
                                options = ['Régression logistique', 'Arbre de décision', 'KNN', 'Forêt aléatoire', 'Boosting'])    
    
    #Appel fonction d'entrainement du modèle
    score, model = train_supervised_model(model_choisi, X_train_reduced, y_train, X_test_reduced, y_test)
    #Affichage du score
    st.write("Score :", score)

    #Affichage graphique variance expliquée
    #if reduction_choice == 'PCA':
        #variance_graph(reduction)
       
    st.markdown("**Recherche des meilleurs hyperparamètres :**")
    col1, col2 = st.columns(2)
    # Bouton "Lancer l'optimisation" dans la première colonne
    search_param = col1.button("Lancer l'optimisation")
    # Bouton "Réinitialiser" dans la deuxième colonne
    reset_button = col2.button("Réinitialiser")
    
    if reset_button:
        search_param = False
        reset_button = False
        
    if search_param:
       
       best_model, best_params = select_best_model(model_choisi, X_train_reduced, y_train)
       
       st.write("Les meilleurs hyperparamètres sont :", best_params)
       score = best_model.score(X_test_reduced, y_test)
       st.write("Score après optimisation des hyperparamètres : ", score)
       
       st.write("Comparaison prédictions vs réalité :")
       st.write(display_crosstab(best_model, X_test_reduced, y_test)[0])

       st.write("Rapport de classification :")
       st.text(display_crosstab(best_model, X_test_reduced, y_test)[1])
       
'''elif page == pages[3]:
    
    st.subheader('Modélisation : Méthode non supervisée')
    
    X_train_scaled = st.session_state['X_train_scaled']
    X_test_scaled = st.session_state['X_test_scaled']
    y_train = st.session_state['y_train']
    y_test = st.session_state['y_test']
    
    X_train_reduced, X_test_reduced, reduction = reduction("PCA", X_train_scaled, y_train, X_test_scaled)
    
    #Selectbox avec Choix du modèle
    methode_choisie = st.selectbox(label = "Choix du modèle", 
                                options = ['K-means', 'Clustering Hiérarchique'])    
    
    #Appel fonction d'entrainement du modèle non supersivé
    model, labels = train_non_supervised_model(methode_choisie, X_train_reduced, y_train, X_test_reduced, y_test)   
    
    #Recherche des clusters optimaux      
    axes, distorsions = search_clusters(methode_choisie, X_train_reduced)

    # Graphique d'affichage des Clusters
    fig = plt.figure()
    plt.plot(axes, distorsions, 'gx-')
    plt.xlabel('Nombre de Clusters K')
    plt.ylabel('Distorsion SSW/(SSW+SSB)')
    plt.title('Méthode du coude affichant le nombre de clusters optimal pour ' + methode_choisie)
    plt.grid(True)
    st.pyplot(fig)
    
    #Boutons d'affichage des clusters
    col1, col2 = st.columns(2)
    search_clusters = col1.button("Afficher les clusters", key="searchcluster")
    reset_button = col2.button("Reset", key="resetclusters")
        
    if reset_button:
        search_clusters = False
        reset_button = False
        
    if search_clusters: 
        
        X_train_reduced, silhouette_avg, labels = display_clusters(methode_choisie, X_train_reduced) 
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(X_train_reduced[:, 0], X_train_reduced[:, 1], c=labels, cmap=plt.cm.Spectral)
        ax.set_xlabel('Axe 1')
        ax.set_ylabel('Axe 2')
        ax.set_title('Visualisation des clusters ')
        st.pyplot(fig)  
        
        st.write("Score Silhouette :", silhouette_avg)'''
    