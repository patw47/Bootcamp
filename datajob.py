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
from sklearn.decomposition import PCA
from main_functions import processing, big_cleaning
from sklearn.linear_model import LogisticRegression
from modeling_functions import ModelContainer, reduction, train_supervised_model, select_best_model, train_non_supervised_model, search_clusters, display_clusters
import warnings
# Ignorer les avertissements
warnings.filterwarnings("ignore")

st.title('Les métiers de la Data')
st.subheader("Quelles qualifications pour quel métier?")
st.write("Etude de faisabilité à partir de [données Kaggle](https://www.kaggle.com/c/kaggle-survey-2020/overview) d'un modèle à destination des recruteurs, pour définir avec justesse le titre du métier recherché en fonction des qualifications et expérience.")
df = pd.read_csv('kaggle_survey_2020_responses.csv')
st.session_state.df = df

pages = ['Introduction',
         'Exploration des données', 
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
    
    st.image('Diapositive1.JPG')   
    st.image('Diapositive2.JPG')
    st.image('Diapositive3.JPG')
    st.image('Diapositive4.JPG')
    st.image('Diapositive5.JPG')
    
if page == pages[1]:
    
    st.subheader(':broom: Nettoyage et Préparation des données')   
    
    st.write(":mag_right: Aperçu du Dataframe original")
    st.dataframe(df.head(10))

    if st.checkbox("Afficher les valeurs manquantes"):
        st.dataframe(df.isna().sum())
        st.write("On constate énormément de valeurs manquantes sur les dernières questions")
        st.write("Total valeurs manquantes :", df.isna().sum().sum())
        
        st.markdown("---")

    st.subheader(":cooking: Préparation des données en 9 étapes : ")
    st.write(":white_check_mark: Suppression première ligne inutile")
    st.write(":white_check_mark: Suppression lignes dont la valeur duration est inf à 2 min")
    st.write(":white_check_mark: Suppression des lignes où la réponse à Q6 est **_I have never written Code_**")
    st.write(":white_check_mark: Suppression des colonnes avec questions sur le thème **_Projection dans deux ans_**")
    st.write(":white_check_mark: Regroupement par famille de métiers, par classe de salaires, région réographique")
    st.write(":white_check_mark: Suppression de valeurs aberrantes (Salaires trop bas ou au contraire trop hauts)")
    st.write(":white_check_mark: Sélection de colonnes pertinentes pour la logique business")
    st.write(":white_check_mark: Remplacement des valeurs par 0 ou 1 pour les colonnes contenant des réponses binaires")
    st.write(":white_check_mark: Remplacement des dernières valeurs catégorielles vides par leur mode")
    
    #Appel fonction nettoyage des données
    df_new = big_cleaning(df)
    
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
        
    st.write(df_new.shape)
      
elif page == pages[2]:
    
    st.subheader(':bar_chart: Data Vizualisation')  
    
    #Récupération var df_new dans la session
    df_new = st.session_state.df_new
    df = st.session_state.df

    
    values_count = df_new["Q5"].value_counts()    
    fig = plt.figure(figsize=(8, 6))
    plt.pie(values_count, labels=values_count.index, autopct='%1.1f%%')
    plt.title("Répartition de la variable cible")
    plt.axis('equal')
    st.pyplot(fig)
    
    st.markdown("---")
    # Filtrer par pays
    selected_country = st.selectbox("Filtrer par pays", sorted(df_new["Q3"].unique()))
    # Filtrer par poste
    selected_category = st.selectbox("Filtrer par catégorie", df_new["Q5"].unique())
    filtered_df = df_new[(df_new["Q3"] == selected_country) & (df_new["Q5"] == selected_category)]
    values_count = filtered_df["Q24"].value_counts()
   
    fig = plt.figure(figsize=(8, 6))
    plt.pie(values_count, labels=values_count.index, autopct='%1.1f%%')
    plt.title("Répartition des salaires par zone géographique et poste")
    plt.axis('equal')
    st.pyplot(fig)
    
    st.markdown("---")
    
    count_by_country = df_new["Q3"].value_counts()
    q4_counts = df_new['Q4'].value_counts()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Premier sous-graphique : Répartition des participants par zone géographique
    ax1.pie(count_by_country.values, labels=count_by_country.index, autopct='%1.1f%%')
    ax1.set_title("Répartition des participants par zone géographique")

    # Deuxième sous-graphique : Répartition des niveaux d'études
    ax2.pie(q4_counts.values, labels=q4_counts.index, autopct='%1.1f%%')
    ax2.set_title("Répartition des niveaux d'études")

    # Ajuster les espacements entre les sous-graphiques
    plt.tight_layout()

    # Afficher la figure
    st.pyplot(fig) 
    st.markdown("---")



    #Bar chart expérience des répondants
    fig = plt.figure(figsize=(8, 6))
    ax = sns.countplot(x = "Q6", data = df_new, order=df_new["Q6"].value_counts().index.sort_values())
    ax.set_title("Expérience des participants")
    st.pyplot(fig)

elif page == pages[3]:
     
    st.subheader(":chains: Encodage des données en 5 étapes : ")
    st.write(":white_check_mark: Séparation variable cible et encodage avec LabelEncoder")
    st.write(":white_check_mark: Séparation jeu de test et jeu d'entrainement")
    st.write(":white_check_mark: Encodage des variables catégorielles avec Getdummies")
    st.write(":white_check_mark: Resampling de la variable cible")
    st.write(":white_check_mark: Réduction au choix avec PCA ou LDA")
    
    df_new = big_cleaning(df)
    
    #Appel fonction processing et séparation des données
    X_test, X_train, y_test, y_train, target_df = processing(df_new)
    
    st.write("Occurence de la variable cible après underresampling de la catégorie majoritaite")
    
    st.write(df_new['Q5'].value_counts())
    
    #Checkpoint
    st.write("Format de X_test après processing: ", X_test.shape)
    st.write("Format de X_train après processing: ", X_train.shape)

  
    #Stockage des variables pour récupération
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.target_df = target_df
    st.session_state.df_new = df_new
    st.session_state.df_new_app = df_new
    
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
        corvar = np.zeros((66, 66))
        for k in range(66):
            corvar[:, k] = reduction.components_[k, :] * sqrt_eigval[k]
                # Delimitation de la figure
            fig, axes = plt.subplots(figsize=(10,10))
            axes.set_xlim(-1, 1)
            axes.set_ylim(-1, 1)
                # Affichage des étiquettes (noms des variables)
        for j in range(66):
                plt.annotate(pd.DataFrame(X_train).columns[j], (corvar[j, 0], corvar[j, 1]), color='#091158')
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
        
        # Obtention des poids des variables dans les composantes principales
        variable_weights = pd.DataFrame(reduction.components_, columns=X_train.columns)
        top_variables = variable_weights.iloc[0].abs().sort_values(ascending=False)[:10]  # Limitez ici le nombre de variables à afficher
        top_variable_weights = variable_weights[top_variables.index]
        fig, ax = plt.subplots()
        ax.bar(top_variable_weights.columns, top_variable_weights.iloc[0].abs().sort_values(ascending=False)[:10])  # Afficher les poids de la première composante principale pour les variables les plus importantes
        ax.set_xlabel("Variables")
        ax.set_ylabel("Poids")
        ax.set_title("Poids des variables les plus importantes dans la première composante principale")
        plt.xticks(rotation=90)
        st.pyplot(fig)
    
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
    
    container = ModelContainer()
    
    if reset_button:
        search_param = False
        reset_button = False
        
    if search_param:
       
        best_model, best_params, model, score_train, score = select_best_model(model_choisi, X_train_reduced, y_train, X_test_reduced, y_test)
        st.write("Les meilleurs hyperparamètres sont :", best_params)
        st.write("Score données train après optimisation des hyperparamètres : ", score_train)
        st.write("Score données test après optimisation des hyperparamètres : ", score)
        st.write("La meilleure combinaison pour ce modèle est :")
        st.write(best_model)
        
        st.write("Comparaison prédictions vs réalité :") 
        y_pred = best_model.predict(X_test_reduced) 
        st.write(pd.crosstab(y_test, y_pred, rownames=['Réel'], colnames=['Prédiction']))

        st.write("Rapport de classification :")
        st.write(classification_report(y_test, y_pred))
            
        st.session_state.search_param = search_param
               
        container.store_model(best_model)
        
    else:    
        container.store_model(model)

    st.session_state.container = container
    
elif page == pages[4]:
    
    st.subheader(':gear: Modélisation : Méthode non supervisée')
    
    #Récupération var df_new dans la session
    df_new = st.session_state.df_new
    X_test = st.session_state.X_test
    X_train = st.session_state.X_train
    y_test = st.session_state.y_test
    y_train = st.session_state.y_train
    
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
        
elif page == pages[5]:
    
    df = pd.read_csv('kaggle_survey_2020_responses.csv')  
    df_new = big_cleaning(df)

    #Appel fonction processing et séparation des données
    X_test, X_train, y_test, y_train, target_df = processing(df_new)    
   # X_train_reduced, X_test_reduced, reduction = reduction("PCA", X_train, y_train, X_test)
    X_train_reduced = X_train
    X_test_reduced = X_test
    
    st.write(X_train_reduced)
    
    best_params = {
        "C":0.1,
        "max_iter":100,
        "penalty":"l2",
        "solver":"liblinear"
        }   
   
    model_app = LogisticRegression()
    model_app.set_params(**best_params)
    model_app.fit(X_train_reduced, y_train)
    
    # Calculer le score sur les données d'entraînement
    train_score = model_app.score(X_train_reduced, y_train)
    st.write("Score sur les données d'entraînement :", train_score)

    # Calculer le score sur les données de test
    test_score = model_app.score(X_test_reduced, y_test)
    st.write("Score sur les données de test :", test_score)

    st.subheader(":pencil: Application du modèle")
    
    #Formulaire contenant les questions pour les recruteurs
    q3_values = df_new["Q3"].unique()
    country = st.selectbox("Pays d'origine", sorted(q3_values), key = "Q3") 
    q4_values = df_new["Q4"].unique()
    education_level = st.selectbox("Niveau d'études", q4_values, key = "Q4") 
    
    st.subheader("Languages de programmation maitrisés:")   
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
      
    st.subheader("IDE le plus souvent utilisé:")  
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
    
    st.subheader("ML framework plus souvent utilisé:")  
    selected_q16_1 = st.checkbox("Scikit-learn", key="Q16_Part_1")
    ml_1 = 1 if selected_q16_1 else 0
    selected_q16_2 = st.checkbox("TensorFlow", key="Q16_Part_2")
    ml_2 = 1 if selected_q16_2 else 0
    selected_q16_3 = st.checkbox("Keras", key="Q16_Part_3")
    ml_3 = 1 if selected_q16_3 else 0
    selected_q16_4 = st.checkbox("PyTorch", key="Q16_Part_4")
    ml_4 = 1 if selected_q16_4 else 0  
    selected_q16_5 = st.checkbox( "Fast.ai", key="Q16_Part_5")
    ml_5 = 1 if selected_q16_5 else 0   
    selected_q16_6 = st.checkbox( "MXNet", key="Q16_Part_6")
    ml_6 = 1 if selected_q16_6 else 0 
    selected_q16_7 = st.checkbox( "Xgboost", key="Q16_Part_7")
    ml_7 = 1 if selected_q16_7 else 0     
    
    st.subheader("Tâches principales:")  
    selected_q23_1 = st.checkbox("Analyze and understand data to influence product or business decisions", key="Q23_Part_1")
    tk_1 = 1 if selected_q23_1 else 0
    selected_q23_2 = st.checkbox("Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data", key="Q23_Part_2")
    tk_2 = 1 if selected_q23_2 else 0
    selected_q23_3 = st.checkbox("Build prototypes to explore applying machine learning to new areas", key="Q23_Part_3")
    tk_3 = 1 if selected_q23_3 else 0
    selected_q23_4 = st.checkbox("Build and/or run a machine learning service that operationally improves my product or workflows", key="Q23_Part_4")
    tk_4 = 1 if selected_q23_4 else 0  
    selected_q23_5 = st.checkbox( "Experimentation and iteration to improve existing ML models", key="Q23_Part_5")
    tk_5 = 1 if selected_q23_5 else 0   
    selected_q23_6 = st.checkbox( "Do research that advances the state of the art of machine learning", key="Q23_Part_6")
    tk_6 = 1 if selected_q23_6 else 0 
    selected_q23_7 = st.checkbox( "None of these activities are an important part of my role at work", key="Q23_Part_7")
    tk_7 = 1 if selected_q23_7 else 0     
    selected_q23_other = st.checkbox( "Other", key="Q23_OTHER")
    tk_8 = 1 if selected_q23_other else 0     
    
    
    st.subheader("Outil de Business Intelligence utilisé:")  
    q32_values = df_new["Q32"].unique()
    bi = st.selectbox("BI", sorted(q32_values), key = "Q32") 
    
    st.subheader("Salaire proposé") 
    q24_values = df_new["Q24"].unique()
    salary = st.selectbox("Salary", sorted(q24_values), key = "Q24")
    
    
      
    # Bouton "Identifier le poste recherché"
    if st.button("Identifier le poste recherché"):
        features = pd.DataFrame(
        data=[
            [
                country,
                education_level,
                programming_language_1,
                programming_language_2,
                programming_language_3,
                programming_language_4,
                programming_language_5,
                programming_language_6,
                programming_language_7,
                experience,
                ide_1,
                ide_2,
                ide_3,
                ide_4,
                ide_5,
                ide_6,
                ide_7,
                bi,
                salary,
                ml_1,
                ml_2,
                ml_3,
                ml_4,
                ml_5,
                ml_6,
                ml_7,
                tk_1,
                tk_2,
                tk_3,
                tk_4,
                tk_5,
                tk_6,
                tk_7,
                tk_8,
            ]
        ],
        columns=[
            "country",
            "education_level",
            "programming_language_1",
            "programming_language_2",
            "programming_language_3",
            "programming_language_4",
            "programming_language_5",
            "programming_language_6",
            "programming_language_7",
            "experience",
            "ide_1",
            "ide_2",
            "ide_3",
            "ide_4",
            "ide_5",
            "ide_6",
            "ide_7",
            "BI",
            "salary",
            "ml_1",
            "ml_2",
            "ml_3",
            "ml_4",
            "ml_5",
            "ml_6",
            "ml_7",
            "tk_1",
            "tk_2",
            "tk_3",
            "tk_4",
            "tk_5",
            "tk_6",
            "tk_7",
            "tk_8",
        ],
    )

        
        # Encoder les données catégorielles avec pd.get_dummies()
        features_encoded = pd.get_dummies(features)
        
        
        #Comme on a pas pu poser toutes les question du df de base, nous préférons remplacer les colonnes manquantes par 0
        missing_cols = set(X_train.columns) - set(features_encoded.columns)
        for col in missing_cols:
            features_encoded[col] = 0
        features_encoded = features_encoded[X_train.columns]
        
        st.write(features_encoded)
        
        
        #pca = PCA()
        #features_reduced = pca.fit_transform(features_encoded)
        prediction = model_app.predict(features_encoded)
      
        st.write(prediction)
        
      #On récupère la variable target sous forme d'un df pour extraire le label correspondant
        target_df = st.session_state.target_df

        # Afficher la prédiction
        #st.write("Le poste prédit est :", target_df.loc[prediction[0], 'Label original'])
        
        predicted_encoding = prediction[0]  # Résultat de la prédiction
        matching_row = target_df[target_df['Encodage'] == predicted_encoding]
        predicted_label = matching_row['Label original'].values[0]  # Valeur de la colonne 'Label original'
        st.write("Le poste prédit est :", predicted_label)  
    

        st.subheader(":pencil: Conclusion")
        st.write("En quelques mots :")
        st.write("Beaucoup d'autres tests à faire, enrichir les données avec plus de data.")
        st.write("Notre hypothèse : À l'époque du questionnaire la majorité des postes data étaient référencés sous l'appelation datascientist.")

   