# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 10:17:41 2023

@author: PatriciaWintrebert

"""
import numpy as np
import pandas as pd
import streamlit as st
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

def big_cleaning(df):
    '''
    Effectue le nettoyage des données dans un DataFrame avant processing. On ne garde que quelques colonnes

    Parameters
    ----------
    df (pandas.DataFrame): Le DataFrame contenant les données à nettoyer 

    Returns
    -------
    pandas.DataFrame: Le DataFrame nettoyé.

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

    #Création du nouveau df dépourvu des gens qui ne programment pas
    df_new = df_new[(df_new['Q6'] != "I have never written code") & (df_new['Q6'].notna())]

    #On vire les colonnes correspondant aux question "dans deux ans" 
    #On les met dans un dataframe pour plus tard 
    df_future = df_new.iloc[:, 256:356]

    #On les efface de df_new
    df_new = df_new.drop(columns=df_future.columns)

    #On vire la colonne duration devenue inutile
    df_new.drop('Duration', axis = 1, inplace = True)

    #Traitement variable cible : 
    #On exclue les lignes contenant les vaariables cibles qui ne nous intéressent pas 
    #On réserve le dataframe de côté pour une éventuelle application plus tard
    # On garde de côté un df contenant ces valeurs
    
    #On vire toutes les lignes dont la Q5 (titre) est vide
    df_new = df_new.dropna(subset=['Q5'])
    
    excluded_values = ["Student", "Other", "Currently not employed", "Product/Project Manager"]
    #df_backup = df_new[df_new['Q5'].isin(excluded_values)]

    #On vire les lignes qui ne nous concernent pas du dataframe
    df_new = df_new.drop(df_new[df_new['Q5'].isin(excluded_values)].index)  
    
    #On regroupe les variables cibles par familles de métiersd
    df_new['Q5'] = title_filtering(df_new['Q5'])
    
    #On vire les colonnes peu pertinentes et/ou avec beaucoup de NA
    #On supprime age et genre qui induisent des biais
   # cols_to_drop = ['Q1','Q2','Q11', 'Q13', 'Q15', 'Q20', 'Q21', 'Q22', 'Q25']
   # cols_to_drop = ['Q1','Q2','Q20', 'Q21', 'Q22', 'Q25', "Q12_Part_1", "Q12_Part_2", "Q12_OTHER", 'Q13']
   
   # cols_to_drop = ['Q1','Q2','Q20', 'Q21', 'Q22', 'Q25', "Q12_Part_1", "Q12_Part_2", "Q12_Part_3", "Q12_OTHER", "Q9_Part_11"]
   # df_new = df_new.drop(cols_to_drop, axis=1)
  
    columns_to_keep = ['Q3', 'Q4', 'Q5', 'Q6', 'Q24','Q25', 'Q32','Q7_Part_1', 'Q7_Part_2', 'Q7_Part_3', 'Q7_Part_4', 'Q7_Part_5', 'Q7_Part_6',
                   'Q7_Part_7', 'Q9_Part_1', 'Q9_Part_2', 'Q9_Part_3', 'Q9_Part_4', 'Q9_Part_5', 'Q9_Part_6', 'Q9_Part_7',
                   'Q16_Part_1', 'Q16_Part_2', 'Q16_Part_3', 'Q16_Part_4', 'Q16_Part_5', 'Q16_Part_6', 'Q16_Part_7',
                   'Q23_Part_1', 'Q23_Part_2', 'Q23_Part_3', 'Q23_Part_4', 'Q23_Part_5', 'Q23_Part_6', 'Q23_Part_7', 'Q23_OTHER']

    df_new = df_new[columns_to_keep]
    
    #On suprrime les pays émergents ou sans données de salaire
    drop_countries = ["Ukraine", "Taiwan", "Nigeria", "India", "Colombia", "Other", "Chile", "Mexico", 
                      "Brazil", "Malaysia", "Philippines", "Argentina", "Bangladesh", "China", 
                      "Romania", "Nepal", "Egypt", "Thailand", "Belarus", "Tunisia", "Pakistan",
                      "Morocco", "Vietnam", "Sri Lanka", "Indonesia", "Peru", "Kenya", 
                      "Iran, Islamic Republic of...", "Republic of Korea",
                      "Poland","Saudi Arabia", "Russia", "Turkey", "Viet Nam", "Viet Nam", 
                      "South Africa","Israel","United Arab Emirates" ]
    
    df_new = df_new[~df_new['Q3'].isin(drop_countries)]
    
    df_new["Q3"] = df_new["Q3"].apply(group_countries_by_region)
    df_new["Q4"] = df_new["Q4"].apply(group_education)
    df_new["Q24"] = df_new["Q24"].apply(group_salaries)
    
    
   # cols = colonnes_incompletes(df_new, 0.2)
    #df_new = df_new.drop(cols, axis=1)
    
    #Conversion des valeurs vides des colonnes contenant les sous questions (identifiées par "_" en 0 et des valeurs existantes en 1)
    for column in df_new.columns:
        if "_" in column:
            df_new[column] = np.where(df_new[column].fillna('') != '', 1, 0)
     
    #Remplacement des dernières valeurs vides catégorielles par None
  #  df_new['Q11'].fillna("A personal computer or laptop", inplace=True)
  #  df_new['Q13'].fillna("Never", inplace=True)
  #  df_new['Q15'].fillna("I do not use machine learning methods", inplace=True)
    df_new['Q32'].fillna("None", inplace=True)
  #  df_new['Q30'].fillna("None", inplace=True)
  #  df_new['Q38'].fillna("None", inplace=True)
    df_new['Q25'].fillna("$0 ($USD)", inplace=True)
    
      
    df_new = remove_outliers_by_category(df_new)
    
    #Enleve des espaces entre les caractères
    df_new = df_new.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    df_new['Q24'].fillna(df_new['Q24'].mode().iloc[0], inplace=True)

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
    
    target_df = pd.DataFrame({'Label original': df_new['Q5'], 'Encodage': target})

    #Séparation des Variables explicatives
    feats = df_new.drop('Q5', axis = 1)
    #Séparation jeu de test et entrainement
    X_train, X_test, y_train, y_test = train_test_split(feats, target , test_size=0.3, random_state=42)

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
    
    train_columns = set(X_train.columns)
    test_columns = set(X_test.columns)
   
    columns_missing_in_train = test_columns - train_columns
    columns_missing_in_test = train_columns - test_columns
   
    # Supprimer les colonnes manquantes dans X_train
    X_train = X_train.drop(columns=columns_missing_in_test)
   
   # Supprimer les colonnes manquantes dans X_test
    X_test = X_test.drop(columns=columns_missing_in_train)
    
    return X_train, X_test, y_train, y_test, target_df

def title_filtering(column):
    '''
    Cette fonction permet de réduire le nombre de variables cibles en remplaçant certaines valeurs par celles qui s'
    en rapprochent le plus.
    
    Params
    -------
    dataframe : pandas.DataFrame
        L'input DataFrame.
    
    Returns
    -------
   pandas.Series
        La colonne modifiée 

    '''        
    replacement_dict = {
        "DBA/Database Engineer": "Machine Learning Engineer",
        "Data Engineer": "Machine Learning Engineer",
        "Statistician" : "Data Analyst",
        "Research Scientist" : "Data Scientist",
        "Business Analyst" : "Data Analyst"
        }
    column = column.replace(replacement_dict)
    return column

def group_countries_by_region(country):
    north_america = ["United States of America", "Canada"]
    europe = ["Italy", "Germany", "France", "Sweden", "United Kingdom of Great Britain and Northern Ireland",
              "Spain", "Portugal", "Netherlands", "Switzerland", "Greece", "Ireland", "Belgium"]
    asia = ["South Korea", "Japan", "Singapore", "Australia"]
    emergents = ["Ukraine", "Taiwan", "Nigeria", "India", "Colombia", "Other", "Chile", "Mexico", 
                      "Brazil", "Malaysia", "Philippines", "Argentina", "Bangladesh", "China", 
                      "Romania", "Nepal", "Egypt", "Thailand", "Belarus", "Tunisia", "Pakistan",
                      "Morocco", "Vietnam", "Sri Lanka", "Indonesia", "Peru", "Kenya", 
                      "Iran, Islamic Republic of...", "Republic of Korea",
                      "Poland","Saudi Arabia", "Russia", "Turkey", "Viet Nam", "Viet Nam", 
                      "South Africa","Israel","United Arab Emirates" ]

    if country in north_america:
        return "Amérique du Nord"
    elif country in europe:
        return "Europe"
    elif country in asia:
        return "Asie et Australie"
    elif country in emergents:
        return "Pays émergents"

    else:
        return country
    
def group_education(school):
       
    no_preference = ["Professional degree", "Some college/university study without earning a bachelor's degree", 
                     'No formal education past high school', 'I prefer not to answer', "Some college/university study without earning a bachelorâ€™s degree"]
    
    pattern = r"study without earning"
    
    if re.search(pattern, school):
        no_preference.append(school)

    if school in no_preference:
        return "No University degree"
    else:
        return school
    
def group_salaries(salary):
  
    very_low = ['20,000-24,999', '25,000-29,999']
    
    low = ['50,000-59,999', '30,000-39,999', '40,000-49,999']
    
    medium = ['60,000-69,999', '70,000-79,999', '80,000-89,999']
    
    expert = ['90,000-99,999', '100,000-124,999' ]
    
    high = ['125,000-149,999','150,000-199,999']
    
    very_high = ['200,000-249,999']
    
    if salary in very_low:
        return "20'000 - 30'000 par an"
    elif salary in low:
        return "30'000 - 60'000 par an"
    elif salary in medium:
        return "60'000 - 90'000 par an"
    elif salary in expert:
        return "90'000 - 125'000 par an"
    elif salary in high:
        return "125'000 - 200'000 par an"
    elif salary in very_high:
        return "plus de 200'000 par an"

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

def remove_outliers_by_category(df):
    '''
    Supprime les lignes correspondant à des valeurs spécifiques dans la colonne "Q24".

    Parameters
    ----------
    df : pandas.DataFrame
        Le DataFrame contenant les données.

    Returns
    -------
    pandas.DataFrame
        Le DataFrame après suppression des lignes.

    '''
    values_to_remove = ["$0 ($USD)", "$0-999","$1000-$9,999","$1000-$1,999","1,000-1,999", "2,000-2,999", "$2000-$2,999", "$3000-$3,999", "$4000-$4,999", "$1-$99", "$100-$999", "15,000-19,999",
                       "10,000-14,999", "5,000-7,499", "3,000-3,999","4,000-4,999", "7,500-9,999", "7,500-7,999", "300,000-500,000", "> $500,000"]
    
    # Supprimer les lignes où "Q24" est vide
    df = df.dropna(subset=['Q24'])
    
    return df[~df['Q24'].isin(values_to_remove)]
    
def resample_df(df_new):
    
   # target_counts = df_new.loc[df_new['Q5'] != "Data Scientist", 'Q5'].value_counts()

   # Trouver le nombre d'occurrences de la catégorie la moins fréquente
   # min_count = int(target_counts.mean())

   # Rééchantillonner la valeur cible 'Data Scientist' pour atteindre le nombre d'occurrences minimum
   # df_data_scientist = df_new[df_new['Q5'] == 'Data Scientist'].sample(n=min_count, replace=False, random_state=42)

   # Concaténer les observations rééchantillonnées avec les autres catégories de la variable cible
   # df_other_categories = df_new[df_new['Q5'] != 'Data Scientist']
   # df_new = pd.concat([df_data_scientist, df_other_categories])
    
    
    # Séparer les données avec la valeur "Data Scientist"
    data_scientist = df_new[df_new['Q5'] == "Data Scientist"]

    # Séparer les données sans la valeur "Data Scientist"
    other_categories = df_new[df_new['Q5'] != "Data Scientist"]


    # Rééchantillonner chaque groupe en conservant le nombre d'occurrences initial
    undersampled_data = other_categories.groupby('Q5').apply(lambda x: x.sample(len(data_scientist), replace=True, random_state=42))

    # Combiner les données rééchantillonnées avec la valeur "Data Scientist"
    df_new = pd.concat([data_scientist, undersampled_data])

    return df_new

def undersampling(df_new):
    # Compter le nombre d'exemples dans chaque classe
    class_counts = df_new['Q5'].value_counts()

   # Trouver la classe majoritaire
    major_class = class_counts.idxmax()

   # Calculer le nombre d'exemples de la classe majoritaire
    major_class_count = class_counts[major_class]

   # Sélectionner les exemples de la classe majoritaire
    major_class_samples = df_new[df_new['Q5'] == major_class]

   # Effectuer l'undersampling de la classe majoritaire en utilisant resample
    undersampled_major_class = resample(major_class_samples,
                                       replace=False,  # Ne pas remplacer les exemples
                                       n_samples=major_class_count,  # Nombre d'exemples à sélectionner
                                       random_state=42)  # Graine aléatoire pour la reproductibilité

   # Sélectionner les exemples des autres classes
    other_classes = df_new[df_new['Q5'] != major_class]

   # Concaténer les exemples de la classe majoritaire undersamplée avec les exemples des autres classes
    df_new = pd.concat([undersampled_major_class, other_classes])

   # Réorganiser l'index du DataFrame résultant
    df_new.reset_index(drop=True, inplace=True)

    return df_new