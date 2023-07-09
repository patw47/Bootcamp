# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 17:44:35 2023

@author: PatriciaWintrebert
"""
import numpy as np
import streamlit as st
import pickle
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
#import matplotlib.pyplot as plt
#from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import cdist
import warnings

# Ignorer les avertissements
warnings.filterwarnings("ignore")

    
@st.cache_data
def reduction(reduction_choice, X_train_scaled, y_train, X_test_scaled):
    '''
    Réduit la dimensionnalité des données en utilisant une méthode de réduction spécifiée.

    Parameters
    ----------
    reduction_choice (str): Le choix de la méthode de réduction. 'PCA' pour l'analyse en composantes principales (PCA) ou 'LDA' pour l'analyse discriminante linéaire (LDA).
    X_train_scaled (numpy.array): Les données d'entraînement mises à l'échelle.
    y_train (numpy.array): Les valeurs cibles d'entraînement.
    X_test_scaled (numpy.array): Les données de test mises à l'échelle.

    Returns
    -------
    tuple: Un tuple contenant les données réduites d'entraînement et de test, ainsi que l'objet de réduction.
            - X_train_reduced (numpy.array): Les données d'entraînement réduites.
            - X_test_reduced (numpy.array): Les données de test réduites.
            - reduction (object): L'objet de réduction (PCA ou LDA) qui a été ajusté sur les données d'entraînement.
    '''
    if reduction_choice == 'PCA':
        reduction = PCA()
    elif reduction_choice == 'LDA':
        reduction = LDA()

    X_train_reduced = reduction.fit_transform(X_train_scaled, y_train)
    X_test_reduced = reduction.transform(X_test_scaled)
    return X_train_reduced, X_test_reduced, reduction

@st.cache_data
def grid_search_params(best_model, param_grid, X_train_reduced, X_test_reduced, y_train, y_test):
    '''
    Effectue une recherche de grille pour trouver les meilleurs hyperparamètres d'un modèle donné.

    Parameters
    ----------
    best_model (object): L'objet du meilleur modèle à utiliser.
    param_grid (dict): Le dictionnaire des paramètres à tester dans la recherche de grille.
    X_train_reduced (numpy.array): Les données d'entraînement réduites.
    X_test_reduced (numpy.arraye): Les données de test réduites.
    y_train (numpy.array): Les valeurs cibles d'entraînement.
    y_test (numpy.array): Les valeurs cibles de test.

    Returns
    -------
    tuple: Un tuple contenant les meilleurs paramètres trouvés, les valeurs cibles réelles et les prédictions.
            - best_params (dict): Les meilleurs paramètres trouvés par la recherche de grille.
            - y_test (numpy.array or pandas.Series): Les valeurs cibles réelles correspondant aux données de test.
            - y_pred (numpy.array or pandas.Series): Les prédictions faites par le meilleur modèle.
    '''
    grid_search = GridSearchCV(estimator=best_model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train_reduced, y_train)
    y_pred = best_model.predict(X_test_reduced)
    return grid_search.best_params_, y_test, y_pred

@st.cache_resource
def train_supervised_model(model_choisi, X_train_reduced, y_train, X_test_reduced, y_test):
    '''
    Entraîne un modèle supervisé choisi sur les données d'entraînement réduites et évalue sa performance sur les données de test.

    Parameters
    ----------
    model_choisi (str): Le choix du modèle à entraîner.
    X_train_reduced (numpy.array): Les données d'entraînement réduites.
    y_train (numpy.array): Les valeurs cibles d'entraînement.
    X_test_reduced (numpy.array): Les données de test réduites.
    y_test (numpy.array): Les valeurs cibles de test.

    Returns
    -------
    tuple: Un tuple contenant le score de performance du modèle sur les données de test et le modèle entraîné.
            - score (float): Le score de performance du modèle sur les données de test.
            - model (object): L'objet du modèle supervisé entraîné.

    '''
    model = 0
    
    if model_choisi == 'Régression logistique':
        model = LogisticRegression()
    elif model_choisi == "Arbre de décision":
        model = DecisionTreeClassifier()
    elif model_choisi == "KNN":
        model = KNeighborsClassifier()
    elif model_choisi == "Forêt aléatoire":
        model = RandomForestClassifier()
    elif model_choisi == "Boosting":
        model = AdaBoostClassifier()

    model.fit(X_train_reduced, y_train)
    score = model.score(X_test_reduced, y_test)
    return score, model

@st.cache_resource
def train_non_supervised_model(model_choisi, X_train_reduced, y_train, X_test_reduced, y_test):
    '''
    Entraîne un modèle non supervisé choisi sur les données d'entraînement réduites et retourne le modèle entraîné ainsi que les labels prédits.

    Parameters
    ----------
    model_choisi (str): Le choix du modèle non supervisé à entraîner.
    X_train_reduced (numpy.array): Les données d'entraînement réduites.
    y_train (numpy.array): Les valeurs cibles d'entraînement.
    X_test_reduced (numpy.array): Les données de test réduites.
    y_test (numpy.array): Les valeurs cibles de test.

    Raises
    ------
    ValueError: Si la méthode choisie n'est pas prise en charge.

    Returns
    -------
    tuple: Un tuple contenant le modèle entraîné et les labels prédits sur les données d'entraînement.
            - model (object): L'objet du modèle non supervisé entraîné.
            - labels (numpy.array): Les labels prédits par le modèle sur les données d'entraînement.
    '''
    if model_choisi == 'K-means':
      model = KMeans(n_clusters=3)  # Modifier le nombre de clusters selon vos besoins
      labels = model.fit_predict(X_train_reduced)
      
    elif model_choisi == 'Clustering Hiérarchique':
      model = AgglomerativeClustering(n_clusters=3)  # Modifier le nombre de clusters selon vos besoins
      labels = model.fit_predict(X_train_reduced)

    else:
      raise ValueError("Méthode non prise en charge")

    return model, labels

@st.cache_resource
def select_best_model(model_choisi, X_train_reduced, y_train, X_test_reduced, y_test):
    '''
    Sélectionne le meilleur modèle pour un choix donné

    Parameters
    ----------
    model_choisi (str): Choix du modèle ('Régression logistique', 'KNN', 'Forêt aléatoire', 'Arbre de décision' ou 'Boosting').
    X_train_reduced (array-like): Données d'entraînement réduites.
    y_train (array-like): Cible des données d'entraînement.
    X_test_reduced (array-like): Données de test réduites.
    y_test (array-like): Cible des données de test.

    Raises
    ------
    ValueError: Si le modèle choisi n'est pas pris en charge.

    Returns
    -------
    tuple: Tuple contenant le meilleur modèle sélectionné, les meilleurs paramètres, le modèle initial et le score.

    '''
    if model_choisi == 'Régression logistique':
        model = LogisticRegression()
        param_grid = {
            'penalty': ['l1', 'l2'],
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag'],
            'max_iter': [100]
        }
    elif model_choisi == 'KNN':
        model = KNeighborsClassifier()
        param_grid = {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'algorithm': ['ball_tree', 'kd_tree', 'brute']
        }
    elif model_choisi == 'Forêt aléatoire':
        model = RandomForestClassifier()
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10]
        }
    elif model_choisi == 'Arbre de décision':
        model = DecisionTreeClassifier()
        param_grid = {
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_choisi == 'Boosting':
        model = AdaBoostClassifier()
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.1, 1.0, 10.0]
        }
    else:
        raise ValueError("Modèle non pris en charge")

    # Effectuer une recherche de grille avec validation croisée
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train_reduced, y_train)

    # Obtenir le meilleur modèle avec les meilleurs paramètres
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    score_train = best_model.score(X_train_reduced, y_train)
    score = best_model.score(X_test_reduced, y_test)

    return best_model, best_params, model, score_train, score

@st.cache_resource
def search_clusters(methode_choisie, X_train_reduced):
    '''
    methode_choisie (str): Le choix de la méthode de clustering. "K-means" pour la méthode des K-moyennes ou "Clustering Hiérarchique" pour le clustering hiérarchique.

    Parameters
    ----------
    methode_choisie (str): Le choix de la méthode de clustering. 
    X_train_reduced (numpy.array): Les données d'entraînement réduites..

    Returns
    -------
    tuple: Un tuple contenant les axes (nombre de clusters) et les distorsions correspondantes.
            - axes (list): La liste des nombres de clusters testés.
            - distorsions (list): La liste des distorsions pour chaque nombre de clusters.
    '''
    range_n_clusters = range(1, 11)
    distorsions = []
    axes = []

    for n_clusters in range_n_clusters:
        if methode_choisie == "K-means":
            cluster = KMeans(n_clusters=n_clusters)
            cluster.fit(X_train_reduced)
            distorsions.append(
                sum(np.min(cdist(X_train_reduced, cluster.cluster_centers_, 'euclidean'), axis=1)) / np.size(
                    X_train_reduced, axis=0))
            axes.append(n_clusters)

        elif methode_choisie == "Clustering Hiérarchique":
            cluster = AgglomerativeClustering(n_clusters=n_clusters)
            labels = cluster.fit_predict(X_train_reduced)
            centroids = []
            for label in range(n_clusters):
                centroid = np.mean(X_train_reduced[labels == label], axis=0)
                centroids.append(centroid)
            distorsion = sum(np.min(cdist(X_train_reduced, centroids, 'euclidean'), axis=1)) / np.size(X_train_reduced,
                                                                                                         axis=0)
            distorsions.append(distorsion)
            axes.append(n_clusters)

    return axes, distorsions

@st.cache_resource       
def display_clusters(methode_choisie, X_train_reduced):
    '''
    Affiche les clusters obtenus avec la méthode choisie et renvoie les données réduites, la silhouette moyenne et les labels des clusters.
    Parameters
    ----------
    methode_choisie : str
        Méthode de clustering choisie ('K-means' ou 'Clustering Hiérarchique').
    X_train_reduced : array-like
        Données réduites d'entraînement.

    Returns
    -------
    tuple
        Tuple contenant les données réduites, la silhouette moyenne et les labels des clusters.
    '''    

    if methode_choisie == "K-means":
        model = KMeans(n_clusters=2)
        labels = model.fit_predict(X_train_reduced)
    elif methode_choisie == "Clustering Hiérarchique":
        model = AgglomerativeClustering(n_clusters=4)
        labels = model.fit_predict(X_train_reduced)
    
    silhouette_avg = silhouette_score(X_train_reduced, labels)
    
    return X_train_reduced, silhouette_avg, labels

class ModelContainer:
    def __init__(self):
        self.model = None

    def store_model(self, model):
        """
        Stocke le modèle entraîné.

        Args:
            model (object): Modèle entraîné à stocker.

        Returns:
            None
        """
        self.model = model

    def load_model(self):
        """
        Charge le modèle stocké.

        Returns:
            object: Modèle chargé.
        """
        return self.model