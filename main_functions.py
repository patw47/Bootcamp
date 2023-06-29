# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 10:17:41 2023

@author: PatriciaWintrebert

"""
import numpy as np
import pandas as pd
import streamlit as st


def nettoyage(df):
    
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

    #On vire les colonnes avec taux de réponse inf à 90%
    cols = colonnes_incompletes(df_new, 0.50)
    df_new = df_new.drop(cols, axis=1)
    
    #Conversion des valeurs vides des colonnes contenant les sous questions (identifiées par "_" en 0 et des valeurs existantes en 1)
    for column in df_new.columns:
        if "_" in column:
            df_new[column] = np.where(df_new[column].fillna('') != '', 1, 0)
    
    #Remplacement des dernières valeurs vides par leur mode
    df_new['Q6'].fillna(df_new['Q6'].mode().iloc[0], inplace=True)
    df_new['Q8'].fillna(df_new['Q8'].mode().iloc[0], inplace=True)
    df_new['Q38'].fillna(df_new['Q38'].mode().iloc[0], inplace=True)
    
    #Stockage df_new dans la var de session
    st.session_state.df_new = df_new
    
    return df_new

    #FIN NETTOYAGE
        
    st.dataframe(df_new.head())

    if st.checkbox("Afficher les valeurs manquantes après nettoyage"):
        st.dataframe(df_new.isna().sum())


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
    colonnes_incompletes = []
    total_lignes = len(dataframe)
    for colonne in dataframe.columns:
        nb_reponses = dataframe[colonne].count()
        ratio_reponses = nb_reponses / total_lignes

        if ratio_reponses < seuil:
            colonnes_incompletes.append(colonne)

    return colonnes_incompletes
