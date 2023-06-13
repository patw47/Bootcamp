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
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
#import plotly_express as px


st.title('Rapport Projet DataJob')

df = pd.read_csv('kaggle_survey_2020_responses.csv')

pages = ['Exploration des données',
         'Nettoyage des données', 
         'Data Visualisation',
         'Pré-Proccessing des données', 
         'Modélisation', 
         'Conclusion']

st.write("Introduction et Cadre de la recherche")
st.markdown("[Lien vers l'introduction](https://docs.google.com/document/d/1Jq3kX3E5a1M_fzxiSG2rvLELMUrImyOqEbJ0CEYMHEo/edit?usp=sharing')")

st.markdown("[Lien vers les données sur le site de Kaggle](https://www.kaggle.com/c/kaggle-survey-2020/overview')")

st.sidebar.title("Sommaire")

page = st.sidebar.radio('Aller vers', pages)


if page == pages[0]:
    #st.image('titanic.jpg')
    st.write('###Exploration des données')    
    st.dataframe(df.head())
    
    if st.checkbox("Afficher les valeurs manquantes"):
        st.dataframe(df.isna().sum())
        
elif page == pages[1]:
    st.write("###Nettoyage des données")
  

elif page == pages[2]:
    
    st.write('###Data Vizualisation')    
    
    fig = plt.figure()
    sns.countplot(x = "Q1", data = df)
    st.pyplot(fig)
    
    fig = plt.figure()
    sns.countplot(x = "Q6", data = df)
    st.pyplot(fig)
    
elif page == pages[3]:
    st.write('###Pre-Processing des données')

elif page == pages[4]:
    st.write('###Modélisation')
    
    df = df.dropna()
   # df = df.drop(["name", "sex", "ticket", "Cabin", "Embarked"], axis = 1)
              
    X = df.drop('Q6', axis = 1)
    y = df['Q6']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    
    model_choisi = st.selectbox(label = "Choix du modèle", 
                                options = ['Régression logistique', 'Arbre de décision', 'KNN'])
    
    def train_model(model_choisi):
        if model_choisi == 'Régression logistique' :
            model = LogisticRegression()
        elif model_choisi == "Arbre de décision":
            model = DecisionTreeClassifier()
        elif model_choisi == "KNN":
            model = KNeighborsClassifier()        
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        return score
    
    st.write("Score test" , train_model(model_choisi) )
    
elif page == pages[5]:
    st.write('###Conclusion')
    
    
    