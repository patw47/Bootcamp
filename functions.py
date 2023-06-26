# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 17:44:35 2023

@author: PatriciaWintrebert
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import prince
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import warnings

# Ignorer les avertissements
warnings.filterwarnings("ignore")


def axes_figure(X_train_reduced, y_train, reduction_choice):
    '''
    Parameters
    ----------
    X_train_reduced : array-like
        Données réduites.
    y_train : array-like
        Étiquettes des données.
    reduction_choice : str
        Méthode de réduction utilisée.

    Returns
    -------
    None
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X_train_reduced[:, 0], X_train_reduced[:, 1], c=y_train, cmap=plt.cm.Spectral)
    ax.set_xlabel('Axe 1')
    ax.set_ylabel('Axe 2')
    ax.set_title("Données projetées sur les 2 axes de " + reduction_choice)
    st.pyplot(fig)
    
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

def reduction(reduction_choice, X_train_scaled, y_train, X_test_scaled):
    '''
    Choix de la méthode de réduction pour le modèle

    Parameters
    ----------
    reduction_choice : str
        The choice of reduction method ('PCA', 'LDA', 'MCA').
    X_train : array-like or sparse matrix, shape (n_samples, n_features)
        The training input samples.
    y_train : array-like, shape (n_samples,)
        The target values.
    X_test : array-like or sparse matrix, shape (n_samples, n_features)
        The testing input samples.

    Returns
    -------
    X_train_reduced : array-like or sparse matrix, shape (n_samples, n_components)
        The reduced feature set for the training data.
        X_test_reduced : array-like or sparse matrix, shape (n_samples, n_components)
        The reduced feature set for the testing data.
    '''
    if reduction_choice == 'PCA':
        reduction = PCA()
    elif reduction_choice == 'LDA':
        reduction = LDA()
    elif reduction_choice == 't-SNE':
        reduction = TSNE(n_components=2)

    X_train_reduced = reduction.fit_transform(X_train_scaled, y_train)
    #t-SNE n'a pas besoin de données test réduites
    if reduction_choice != 't-SNE':
        X_test_reduced = reduction.transform(X_test_scaled)
    if reduction_choice == 't-SNE':    
       X_test_reduced =  X_test_scaled
    return X_train_reduced, X_test_reduced, reduction

def train_model(model_choisi, X_train_reduced, y_train, X_test_reduced, y_test):
    if model_choisi == 'Régression logistique' :
        model = LogisticRegression()
    elif model_choisi == "Arbre de décision":
        model = DecisionTreeClassifier()
    elif model_choisi == "KNN":
        model = KNeighborsClassifier()
    elif model_choisi == "Forêt aléatoire":
        model = RandomForestClassifier()
    elif model_choisi == "K-means Clustering":
        model = KMeans(n_clusters=3)
    
    if model_choisi != "K-means Clustering":
        model.fit(X_train_reduced, y_train)
        score = model.score(X_test_reduced, y_test)
        return score, model
    else:
        model.fit(X_train_reduced)
        labels = model.predict(X_train_reduced)
        score = 0
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(X_train_reduced[:, 0], X_train_reduced[:, 1], c=labels, cmap=plt.cm.Spectral)
        ax.set_xlabel('Axe 1')
        ax.set_ylabel('Axe 2')
        ax.set_title("K-means Clustering")
        st.pyplot(fig)
        return model
    
 # Affichage de la table de contingence
def display_crosstab(model, X_test_reduced, y_test):
    y_pred = model.predict(X_test_reduced)
    #crosstab = pd.crosstab(y_test, y_pred, colnames=['Prédiction'], rownames=['Realité'])
    crosstab = pd.crosstab(y_test, y_pred, rownames=['Réel'], colnames=['Prédiction'])
    # Ajouter un rapport de classification
    report = classification_report(y_test, y_pred)

    #Features importance
    return crosstab, report

def variance_graph(reduction):
    # Afficher la variance expliquée pour chaque composante grâce à l'attribut explained_variance_ de PCA.
    #st.write('Les valeurs propres sont :', reduction.explained_variance_)
    fig = plt.figure()
    # Tracer le graphe représentant la variance expliquée en fonction du nombre de composantes.
    plt.plot(np.arange(1, 439), reduction.explained_variance_)
    plt.xlabel('Nombre de facteurs')
    plt.ylabel('Valeurs propres')
    # Afficher le graphe dans Streamlit
    st.pyplot(fig)