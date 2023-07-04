# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 10:17:41 2023

@author: PatriciaWintrebert

"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def nettoyage(df, remove):
    '''
    Effectue le nettoyage des données dans un DataFrame avant processing

    Parameters
    ----------
    df (pandas.DataFrame): Le DataFrame contenant les données à nettoyer 
    remove (bool): Indique si les colonnes incomplètes doivent être supprimées pour alléger le calcul

    Returns
    -------
    pandas.DataFrame: Le DataFrame nettoyé..

    '''
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
    
    #On regroupe les variables cibles par familles de métiers
    filter_title = "Oui"
    df_new['Q5'] = title_filtering(filter_title, df_new['Q5'])

    #On vire les colonnes correspondant aux question "dans deux ans" 
    #On les met dans un dataframe pour plus tard 
    df_future = df_new.iloc[:, 256:356]

    #On les efface de df_new
    df_new = df_new.drop(columns=df_future.columns)

    #On vire la colonne duration devenue inutile
    df_new.drop('Duration', axis = 1, inplace = True)

    #On exclue les lignes contenant les vaariables cibles qui ne nous intéressent pas 
    #On réserve le dataframe de côté pour une éventuelle application plus tard
    # On garde de côté un df contenant ces valeurs
    excluded_values = ["Student", "Other", "Currently not employed"]
    #df_backup = df_new[df_new['Q5'].isin(excluded_values)]

    #On vire les lignes qui ne nous concernent pas du dataframe
    df_new = df_new.drop(df_new[df_new['Q5'].isin(excluded_values)].index)  
    
    #On vire les colonnes peux pertinentes avec beaucoup de NA
    cols_to_drop = ['Q11', 'Q13', 'Q15', 'Q20', 'Q21', 'Q22', 'Q24', 'Q25']
    df_new = df_new.drop(cols_to_drop, axis=1)

    if remove == True:
        #On vire les colonnes avec taux de réponse inf à 90%
        cols = colonnes_incompletes(df_new, 0.9)
        df_new = df_new.drop(cols, axis=1)
    
    #Conversion des valeurs vides des colonnes contenant les sous questions (identifiées par "_" en 0 et des valeurs existantes en 1)
    for column in df_new.columns:
        if "_" in column:
            df_new[column] = np.where(df_new[column].fillna('') != '', 1, 0)
      
    #Remplacement des dernières valeurs vides catégorielles par leur mode
    df_new['Q6'].fillna(df_new['Q6'].mode().iloc[0], inplace=True)
    df_new['Q8'].fillna(df_new['Q8'].mode().iloc[0], inplace=True)
    
   #Colonnes à traiter en plus si on garde toutes les données
    if remove == False:
        df_new['Q38'].fillna(df_new['Q38'].mode().iloc[0], inplace=True) 
        df_new['Q30'].fillna(df_new['Q30'].mode().iloc[0], inplace=True)
        #On supprime la colonne Q32 peu pertinente qui rend les jeu de test impair
        df_new = df_new.drop("Q32", axis=1)

    #Stockage df_new dans la var de session
    st.session_state.df_new = df_new
    
    return df_new    


def processing(df):
    '''
    Effectue le prétraitement des données en vue de l'entraînement d'un modèle.

    Parameters
    ----------
    df (pandas.DataFrame): Le DataFrame contenant les données.


    Returns
    -------
    tuple: Un tuple contenant les jeux de données d'entraînement et de test, ainsi que les variables cibles.
           - X_train (pandas.DataFrame): Le jeu de données d'entraînement prétraité.
           - X_test (pandas.DataFrame): Le jeu de données de test prétraité.
           - y_train (numpy.array): Les valeurs cibles correspondant au jeu de données d'entraînement.
           - y_test (numpy.array): Les valeurs cibles correspondant au jeu de données de test.

    '''
    #Récupération var df_new dans la session
    df_new = st.session_state.df_new
    
    #Séparation et Encodage variable cible
    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(df_new['Q5'])

    #Séparation des Variables explicatives
    feats = df_new.drop('Q5', axis = 1)
    #Séparation jeu de test et entrainement
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
    
    return X_train, X_test, y_train, y_test

def title_filtering(filter_title, column):
    '''
    Cette fonction permet de réduire le nombre de variables cibles en remplaçant certaines valeurs par celles qui s'
    en rapprochent le plus.
    
    Params
    -------
    filter_title : str
        L'option de filtrage choisie ('Oui' or 'Non').
    dataframe : pandas.DataFrame
        L'input DataFrame.
    
    Returns
    -------
   pandas.Series
        La colonne modifiée 

    '''        
    if filter_title == 'Oui':
        replacement_dict = {
            "DBA/Database Engineer": "Data Engineer",
            "Statistician" : "Data Analyst",
            "Research Scientist" : "Data Scientist",
            "Business Analyst" : "Data Analyst",
            "Software Engineer": "Machine Learning Engineer"
            }
        column = column.replace(replacement_dict)
    return column
    

def colonnes_incompletes(dataframe, seuil):
    '''
    Identifie les colonnes incomplètes dans un DataFrame en fonction d'un seuil donné.
    
    Parameters
    ----------
    dataframe (pandas.DataFrame): Le DataFrame à analyser.
    seuil (float): Le seuil de complétude des colonnes. 

    Returns
    -------
    list: Une liste contenant les noms des colonnes incomplètes

    '''
    colonnes_incompletes = []
    total_lignes = len(dataframe)
    for colonne in dataframe.columns:
        nb_reponses = dataframe[colonne].count()
        ratio_reponses = nb_reponses / total_lignes

        if ratio_reponses < seuil:
            colonnes_incompletes.append(colonne)

    return colonnes_incompletes
