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
from sklearn.metrics import classification_report
from main_functions import nettoyage, processing
from modeling_functions import reduction, train_supervised_model, select_best_model, train_non_supervised_model, search_clusters, display_clusters
import warnings

# Ignorer les avertissements
warnings.filterwarnings("ignore")

st.title('Projet DataJob')
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

    st.subheader("Préparation des données en 8 étapes : ")
    st.write("1.Suppression première ligne inutile")
    st.write("2.Supression lignes dont la valeur duration est inf à 2 min")
    st.write("3.Supression des lignes où la réponse à Q6 est I have never written Code")
    st.write("4.Supression des colonnes avec questions sur le thème Projection dans deux ans")
    st.write("5.Regroupement par famille de métiers")
    st.write("6.Supression de quelques colonnes peu pertinentes")
    st.write("7.Remplacement des valeurs par 0 ou 1 pour les colonnes contenant des réponses binaires")
    st.write("8.Remplacement des dernières valeurs catégorielles vides par leur mode")
    
    #Appel fonction nettoyage des données
    df_new = nettoyage(df, remove = False)
    
    #Récupération var df_new dans la session
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
    
    df_new = nettoyage(df, remove = False)
    
    #Appel fonction processing et séparation des données
    X_test, X_train, y_test, y_train = processing(df_new)
    
    #Checkpoint
    st.write("Format de X_test après processing: ", X_test.shape)
    st.write("Format de X_train après processing: ", X_train.shape)
  
    #Stockage des variables pour récupération
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    
    st.subheader('Modélisation : Méthode supervisée')
    
    #Selectbox Choix de la méthode de réduction
    reduction_choice = st.selectbox(label = "Choix de la méthode de réduction", 
                                options = ['PCA', 'LDA'])    
    
    #Appel fonction réduction de données
    X_train_reduced, X_test_reduced, reduction = reduction(reduction_choice, X_train, y_train, X_test)
    
    if reduction_choice =="PCA":
        #Cercle des corrélations, qui nous permet d'évaluer
        #l'influence de chaque variable pour chaque axe de représentation.
        sqrt_eigval = np.sqrt(reduction.explained_variance_)
        corvar = np.zeros((356, 356))
        for k in range(356):
            corvar[:, k] = reduction.components_[k, :] * sqrt_eigval[k]
                # Delimitation de la figure
            fig, axes = plt.subplots(figsize=(20, 20))
            axes.set_xlim(-1, 1)
            axes.set_ylim(-1, 1)
                # Affichage des étiquettes (noms des variables)
        for j in range(356):
                plt.annotate(pd.DataFrame(X_train_reduced).columns[j], (corvar[j, 0], corvar[j, 1]), color='#091158')
                plt.arrow(0, 0, corvar[j, 0]*0.9, corvar[j, 1]*0.9, alpha=0.5, head_width=0.03, color='b')
            # Ajouter les axes
        plt.plot([-1, 1], [0, 0], color='silver', linestyle='-', linewidth=1)
        plt.plot([0, 0], [-1, 1], color='silver', linestyle='-', linewidth=1)
    # Cercle et légendes
        cercle = plt.Circle((0, 0), 1, color='#16E4CA', fill=False)
        axes.add_artist(cercle)
        plt.xlabel('AXE 1')
        plt.ylabel('AXE 2')
        st.pyplot(plt)
    
    #Selectbox avec Choix du modèle
    model_choisi = st.selectbox(label = "Choix du modèle", 
                                options = ['Régression logistique', 'Arbre de décision', 'KNN', 'Forêt aléatoire', 'Boosting'])    
    
    #Appel fonction d'entrainement du modèle
    score, model = train_supervised_model(model_choisi, X_train_reduced, y_train, X_test_reduced, y_test)
    #Affichage du score
    st.write("Score :", score)

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
       
       best_model, best_params, model, score = select_best_model(model_choisi, X_train_reduced, y_train, X_test_reduced, y_test)
       
       st.write("Les meilleurs hyperparamètres sont :", best_params)

       st.write("Score après optimisation des hyperparamètres : ", score)
       
       st.write("La meilleure combinaison pour ce modèle est :")
       st.write(best_model)
       
       st.write("Comparaison prédictions vs réalité :")
       #st.write(display_crosstab(model, X_test_reduced, y_test)[0])
       y_pred = best_model.predict(X_test_reduced) 
       st.write(pd.crosstab(y_test, y_pred, rownames=['Réel'], colnames=['Prédiction']))

       st.write("Rapport de classification :")
       st.write(classification_report(y_test, y_pred))
       
elif page == pages[3]:
    
    st.subheader('Modélisation : Méthode non supervisée')
    
    #Récupération var df_new dans la session
    df_new = st.session_state.df_new
    
    df_new = nettoyage(df, remove = True)
    
    #Appel fonction processing et séparation des données
    X_test, X_train, y_test, y_train = processing(df_new)
    
    #Checkpoint
    st.write("Format de X_test après processing: ", X_test.shape)
    st.write("Format de X_train après processing: ", X_train.shape)
    
    #Pour la modélisation non supervisée non choisissons après différents tests d'opter d'office pour PCA pour des 
    #raisons de performances du code
    X_train_reduced, X_test_reduced, reduction = reduction("PCA", X_train, y_train, X_test)
    
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
        #Appel fonction d'affichage des clusters et score silhouette
        X_train_reduced, silhouette_avg, labels = display_clusters(methode_choisie, X_train_reduced) 
        
        fig = plt.figure()
        #ax = fig.add_subplot(111)
        ax = fig.add_subplot(111)
        ax.scatter(X_train_reduced[:, 0], X_train_reduced[:, 1], c=labels, cmap=plt.cm.Spectral)
        ax.set_xlabel('Axe 1')
        ax.set_ylabel('Axe 2')
        ax.set_title('Visualisation des clusters ')
        st.pyplot(fig)  
        
        st.write("Score Silhouette :", silhouette_avg)
    