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
import webbrowser
from sklearn.metrics import classification_report
from main_functions import nettoyage, processing
from sklearn.linear_model import LogisticRegression
from modeling_functions import  reduction, train_supervised_model, select_best_model, train_non_supervised_model, search_clusters, display_clusters
import warnings

# Ignorer les avertissements
warnings.filterwarnings("ignore")

st.title('Les métiers de la Data')
st.subheader("Quelles qualifications pour quel métier?")
st.write("Etude de faisabilité à partir de [données Kaggle](https://www.kaggle.com/c/kaggle-survey-2020/overview) d'un modèle à destination des recruteurs, pour définir avec justesse le titre du métier recherché en fonction des qualifications et expérience.")
df = pd.read_csv('kaggle_survey_2020_responses.csv')
st.session_state.df = df

pages = ['Exploration des données', 
         'Data Visualisation',
         'Modélisation Supervisée',
         'Modélisation non supervisée',
         'Application'
         ]

col1, col2 = st.columns(2)

with col1:
    if st.button("Lien vers le rapport", key="rapport"):
        webbrowser.open("https://docs.google.com/document/d/1DLS5DsbR-z5cnq5FYZIlrufrJUUiUxgFgHqk9vBGz2c/edit?usp=sharing")

# Bouton pour le lien GitHub
with col2:
    if st.button("GitHub", key="github"):
        webbrowser.open("https://github.com/patw47/Bootcamp")
        

st.sidebar.title("Sommaire")
st.sidebar.image('datajob.jpg')
page = st.sidebar.radio('Aller vers', pages)

if page == pages[0]:
    st.subheader(':broom: Nettoyage et Préparation des données')   
    
    st.write(":mag_right: Aperçu du Dataframe original")
    st.dataframe(df.head(10))

    if st.checkbox("Afficher les valeurs manquantes"):
        st.dataframe(df.isna().sum())
        st.write("On constate énormément de valeurs manquantes sur les dernières questions")
        st.write("Total valeurs manquantes :", df.isna().sum().sum())
        
        st.markdown("---")

    st.subheader(":cooking: Préparation des données en 8 étapes : ")
    st.write(":white_check_mark: Suppression première ligne inutile")
    st.write(":white_check_mark: Suppression lignes dont la valeur duration est inf à 2 min")
    st.write(":white_check_mark: Suppression des lignes où la réponse à Q6 est **_I have never written Code_**")
    st.write(":white_check_mark: Suppression des colonnes avec questions sur le thème **_Projection dans deux ans_**")
    st.write(":white_check_mark: Regroupement par famille de métiers")
    st.write(":white_check_mark: Suppression de quelques colonnes peu pertinentes")
    st.write(":white_check_mark: Remplacement des valeurs par 0 ou 1 pour les colonnes contenant des réponses binaires")
    st.write(":white_check_mark: Remplacement des dernières valeurs catégorielles vides par leur mode")
    
    #Appel fonction nettoyage des données
    df_new = nettoyage(df, remove = False)
    
    #Récupération var df_new dans la session
    st.session_state.df_new = df_new
    st.session_state.df = df
  
    st.markdown("---")
    
    #Check des valeurs de Q5 filtrées    
    col1, col2 = st.columns(2)
    col1.write("Variables cibles avant regroupement et nettoyage")
    col1.write(df['Q5'].value_counts())
    col2.write("Variables cibles après regroupement")
    col2.write(df_new['Q5'].value_counts())
    
    st.markdown("---")
    
    st.write(":mag_right: Aperçu du Dataframe nettoyé")
    st.dataframe(df_new.head())

    if st.checkbox("Afficher les valeurs manquantes après nettoyage"):
        st.dataframe(df_new.isna().sum())
        st.write("Total valeurs manquantes :", df_new.isna().sum().sum())
      
elif page == pages[1]:
    
    st.subheader(':bar_chart: Data Vizualisation')  
    
    #Récupération var df_new dans la session
    df_new = st.session_state.df_new
    df = st.session_state.df
    
    #Bar chart âge des répondants
    fig = plt.figure()
    ax = sns.countplot(x="Q1", data=df_new, order=df_new["Q1"].value_counts().index.sort_values())
    ax.set_title("Age des participants")
    st.pyplot(fig)
    
    #Bar chart expérience des répondants
    fig = plt.figure()
    ax = sns.countplot(x = "Q6", data = df_new, order=df_new["Q6"].value_counts().index.sort_values())
    ax.set_title("Expérience des participants")
    st.pyplot(fig)
    
    
    #Heatmap corrélation titre et salaire
    st.subheader("Relation entre poste et salaire annuel en fonction du pays")
    selected_country = st.selectbox("Sélectionner ou taper le nom d'un pays", [''] + list(df_new['Q3'].unique()))

    if selected_country:
        filtered_data = df_new[df_new['Q3'] == selected_country]
    else:
        filtered_data = df_new
    contingency_table = pd.crosstab(filtered_data['Q24'], filtered_data['Q5'])
    fig = plt.figure(figsize=(10, 20))
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu')
    plt.xlabel('Titre')
    plt.ylabel('Salaire annuel')
    st.pyplot(fig)


elif page == pages[2]:
     
    st.subheader(":chains: Encodage des données en 4 étapes : ")
    st.write(":white_check_mark: Séparation variable cible et encodage avec LabelEncoder")
    st.write(":white_check_mark: Séparation jeu de test et jeu d'entrainement")
    st.write(":white_check_mark: Encodage des variables catégorielles avec Getdummies")
    st.write(":white_check_mark: Réduction au choix avec PCA ou LDA")
    
    df_new = nettoyage(df, remove = False)
    
    #Appel fonction processing et séparation des données
    X_test, X_train, y_test, y_train, target_df = processing(df_new)
    
    #Checkpoint
    st.write("Format de X_test après processing: ", X_test.shape)
    st.write("Format de X_train après processing: ", X_train.shape)
  
    #Stockage des variables pour récupération
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.target_df = target_df
    
    st.subheader(':gear: Modélisation : Méthode supervisée')
    
    #Selectbox Choix de la méthode de réduction
    reduction_choice = st.selectbox(label = "Choix de la méthode de réduction", 
                                options = ['PCA', 'LDA'])    
    
    #Appel fonction réduction de données
    X_train_reduced, X_test_reduced, reduction = reduction(reduction_choice, X_train, y_train, X_test)
    
    if reduction_choice =="PCA":
        #Cercle des corrélations, qui nous permet d'évaluer
        #l'influence de chaque variable pour chaque axe de représentation.
        sqrt_eigval = np.sqrt(reduction.explained_variance_)
        corvar = np.zeros((382, 382))
        for k in range(382):
            corvar[:, k] = reduction.components_[k, :] * sqrt_eigval[k]
                # Delimitation de la figure
            fig, axes = plt.subplots(figsize=(10, 10))
            axes.set_xlim(-1, 1)
            axes.set_ylim(-1, 1)
                # Affichage des étiquettes (noms des variables)
        for j in range(382):
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
        y_pred = best_model.predict(X_test_reduced) 
        st.write(pd.crosstab(y_test, y_pred, rownames=['Réel'], colnames=['Prédiction']))

        st.write("Rapport de classification :")
        st.write(classification_report(y_test, y_pred))
            
        st.session_state.search_param = search_param
        
    #Pour plus de facilité on va directement stocker les bonnes variables entrainées pour la suite
    best_params = {
        "C":0.01,
        "max_iter":100,
        "penalty":"l2",
        "solver":"liblinear"
        }
    model_app = LogisticRegression()
    model_app.set_params(**best_params)
    model_app.fit(X_train_reduced, y_train)
    
    st.session_state.trained_model = model_app

elif page == pages[3]:
    
    st.subheader(':gear: Modélisation : Méthode non supervisée')
    
    #Récupération var df_new dans la session
    df_new = st.session_state.df_new
    
    #On refait un nettoyage mais plus avancé pour accélérer le chargement
    df_new = nettoyage(df, remove = True)   
    #Appel fonction processing et séparation des données
    X_test, X_train, y_test, y_train, target_df = processing(df_new)
    
    #Checkpoint
    st.write("Pour la modélisation non supervisée nous avons choisi de réduire encore plus les données pour un chargement plus rapide.")
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
        
elif page == pages[4]:
    
    #Stockage des variables pour récupération
    y_train = st.session_state.y_train
    X_train = st.session_state.X_train
    df_new = st.session_state.df_new


    st.subheader(":pencil: Application du modèle")
    
    #Formulaire contenant les questions pour les recruteurs
    q1_values = df_new["Q1"].unique()
    age = st.selectbox("Âge du candidat", q1_values, key = "Q1") 
    q3_values = df_new["Q3"].unique()
    country = st.selectbox("Pays d'origine", q3_values, key = "Q3") 
    q4_values = df_new["Q4"].unique()
    education_level = st.selectbox("Niveau d'études", q4_values, key = "Q4") 
    
    st.write("Languages de programmation maitrisés:")   
    selected_q7_1 = st.checkbox("Python", key="Q7_Part_1")
    programming_language_1 = 1 if selected_q7_1 else 0
    selected_q7_2 = st.checkbox("R", key="Q7_Part_2")
    programming_language_2 = 1 if selected_q7_2 else 0
    selected_q7_3 = st.checkbox("SQL", key="Q7_Part_3")
    programming_language_3 = 1 if selected_q7_3 else 0
    selected_q7_4 = st.checkbox("C", key="Q7_Part_4")
    programming_language_4 = 1 if selected_q7_4 else 0      
    selected_q7_5 = st.checkbox("C++", key="Q7_Part_5")
    programming_language_5= 1 if selected_q7_5 else 0    
    selected_q7_6 = st.checkbox("Java", key="Q7_Part_6")
    programming_language_6= 1 if selected_q7_6 else 0     
    selected_q7_7 = st.checkbox("Javascript", key="Q7_Part_7")
    programming_language_7= 1 if selected_q7_7 else 0     
    q6_values = df_new["Q6"].unique()
    experience = st.selectbox("Expérience", q6_values, key = "Q6")
      
    st.write("IDE le plus souvent utilisé:")  
    selected_q9_1 = st.checkbox("Jupyter (JupyterLab, Jupyter Notebooks, etc) ", key="Q9_Part_1")
    ide_1 = 1 if selected_q9_1 else 0
    selected_q9_2 = st.checkbox("RStudio", key="Q9_Part_2")
    ide_2 = 1 if selected_q9_2 else 0
    selected_q9_3 = st.checkbox("Visual Studio", key="Q9_Part_3")
    ide_3 = 1 if selected_q9_3 else 0
    selected_q9_4 = st.checkbox("Visual Studio Code (VSCode)", key="Q9_Part_4")
    ide_4 = 1 if selected_q9_4 else 0  
    selected_q9_5 = st.checkbox( "PyCharm", key="Q9_Part_5")
    ide_5 = 1 if selected_q9_5 else 0   
    selected_q9_6 = st.checkbox( "Spyder ", key="Q9_Part_6")
    ide_6 = 1 if selected_q9_6 else 0 
    selected_q9_7 = st.checkbox( "Notepad++", key="Q9_Part_7")
    ide_7 = 1 if selected_q9_7 else 0     
    q24_values = df_new["Q24"].unique()
    salary = st.selectbox("Salary", q24_values, key = "Q24")
      
    # Bouton "Identifier le poste recherché"
    if st.button("Identifier le poste recherché"):
        features = pd.DataFrame(data=[[age, country, education_level, programming_language_1, programming_language_2, programming_language_3, 
                                         programming_language_4, programming_language_5, programming_language_6, programming_language_7, experience,
                                         ide_1, ide_2, ide_3, ide_4, ide_5, ide_6, ide_7, salary]], 
                                  columns=['age', 'country', 'education_level', 
                                           'programming_language_1', 'programming_language_2' ,'programming_language_3', 'programming_language_4', 'programming_language_5', 'programming_language_6', 'programming_language_7',
                                           'experience', 'ide_1', 'ide_2', 'ide_3', 'ide_4', 'ide_5', 'ide_6', 'ide_7', 'salary'])
      
   

    # Encoder les données catégorielles avec pd.get_dummies()
        features_encoded = pd.get_dummies(features)
    
        #Comme on a pas pu poser toutes les question du df de base, nous préférons remplacer les colonnes manquantes par 0
        missing_cols = set(X_train.columns) - set(features_encoded.columns)
        for col in missing_cols:
            features_encoded[col] = 0
        features_encoded = features_encoded[X_train.columns]
  
        # Charger le modèle pré-entraîné
        model = st.session_state.trained_model

        prediction = model.predict(features_encoded)
      
        
      #On récupère la variable target sous forme d'un df pour extraire le label correspondant
        target_df = st.session_state.target_df

        # Afficher la prédiction
        st.write("Le poste prédit est :", target_df.loc[prediction[0], 'Label original'])
    


   