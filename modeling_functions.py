# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 17:44:35 2023

@author: PatriciaWintrebert
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import GridSearchCV
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import cdist
import warnings

# Ignorer les avertissements
warnings.filterwarnings("ignore")

@st.cache
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

@st.cache
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

    X_train_reduced = reduction.fit_transform(X_train_scaled, y_train)
    X_test_reduced = reduction.transform(X_test_scaled)
    return X_train_reduced, X_test_reduced, reduction
    
 # Affichage de la table de contingence
@st.cache
def display_crosstab(model, X_test_reduced, y_test):
    y_pred = model.predict(X_test_reduced)
    #crosstab = pd.crosstab(y_test, y_pred, colnames=['Prédiction'], rownames=['Realité'])
    crosstab = pd.crosstab(y_test, y_pred, rownames=['Réel'], colnames=['Prédiction'])
    # Ajouter un rapport de classification
    report = classification_report(y_test, y_pred)

    #Features importance
    return crosstab, report

@st.cache
def variance_graph(reduction):
    # Afficher la variance expliquée pour chaque composante grâce à l'attribut explained_variance_ de PCA.
    #st.write('Les valeurs propres sont :', reduction.explained_variance_)
    fig = plt.figure()
    # Tracer le graphe représentant la variance expliquée en fonction du nombre de composantes.
    plt.plot(np.arange(1, 112), reduction.explained_variance_)
    plt.xlabel('Nombre de facteurs')
    plt.ylabel('Valeurs propres')
    # Afficher le graphe dans Streamlit
    st.pyplot(fig)
    
@st.cache
def grid_search_params(best_model, param_grid, X_train_reduced, X_test_reduced, y_train, y_test):
    grid_search = GridSearchCV(estimator=best_model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train_reduced, y_train)
    # Évaluation du  modèle sur les données de test
    y_pred = best_model.predict(X_test_reduced)
    # Renvoie les prédictions et les meilleurs hyperparamètres
    return grid_search.best_params_, y_test, y_pred

@st.cache
def train_supervised_model(model_choisi, X_train_reduced, y_train, X_test_reduced, y_test):
    
    #X_train_reduced = reduce_sample(X_train_reduced)
    #y_train = reduce_y_train(y_train)
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

@st.cache
def train_non_supervised_model(model_choisi, X_train_reduced, y_train, X_test_reduced, y_test):
    #On réduit les données pour alléger le calcul
   #X_train_reduced = reduce_sample(X_train_reduced)
   #y_train = reduce_y_train(y_train)

   if model_choisi == 'K-means':
      model = KMeans(n_clusters=3)  # Modifier le nombre de clusters selon vos besoins
      labels = model.fit_predict(X_train_reduced)
      
   elif model_choisi == 'Clustering Hiérarchique':
      model = AgglomerativeClustering(n_clusters=3)  # Modifier le nombre de clusters selon vos besoins
      labels = model.fit_predict(X_train_reduced)
      # Calcul de la matrice de dissimilarité
      #linkage_matrix = linkage(X_train_reduced, method='complete', metric='euclidean')

       # Construction du dendrogramme
      #fig = plt.figure()
      #ax = fig.add_subplot(111)
      #dendrogram(linkage_matrix)
      #ax.set_title('Dendrogramme')
      #ax.set_xlabel('Échantillons')
      #ax.set_ylabel('Distance')
      #st.pyplot(fig)

   elif model_choisi == 'Mean Shift':
      model = MeanShift(bandwidth=0.5)
      labels = model.fit_predict(X_train_reduced)

   else:
      raise ValueError("Méthode non prise en charge")

   return model, labels

@st.cache
def select_best_model(model_choisi, X_train_reduced, y_train):
    if model_choisi == 'Régression logistique':
        model = LogisticRegression()
        param_grid = {
            'penalty': ['l1', 'l2'],
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag'],
            'max_iter': [100, 200, 300]
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

    return best_model, best_params

@st.cache
def find_optimal_clusters(model, data, max_clusters):
    scores = []
    
    for n_clusters in range(2, max_clusters+1, 2):
        model.n_clusters = n_clusters
        labels = model.fit_predict(data)
        
        # Calculer le score de silhouette
        silhouette = silhouette_score(data, labels)   
        scores.append((n_clusters, silhouette))
    
    # Afficher les scores
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot([score[0] for score in scores], [score[1] for score in scores], marker='o')
    ax.set_xlabel('Nombre de clusters')
    ax.set_ylabel('Coefficient de silhouette')
    ax.set_title('Coefficient de silhouette en fonction du nombre de clusters')
    
    plt.tight_layout()
    plt.show()
    
    return scores

@st.cache
def plot_clusters(axes, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(axes[:, 0], axes[:, 1], c=labels, cmap=plt.cm.Spectral)
    ax.set_xlabel('Axe 1')
    ax.set_ylabel('Axe 2')
    ax.set_title('Projection t-SNE avec couleurs de cluster')
    return fig

@st.cache
def search_clusters(methode_choisie, X_train_reduced):
    # On échantillonne les données pour limiter le calcul
    #X_train_reduced = reduce_sample(X_train_reduced)

    # Liste des nombres de clusters
    range_n_clusters = range(1, 21)

    # Initialisation des listes de distorsions et d'axes
    distorsions = []
    axes = []
    bandwidths = [0.1, 0.5, 1.0, 2.0]

    # Calcul des distorsions pour les différents modèles
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

        elif methode_choisie == "Mean Shift":
            for bandwidth in bandwidths:
                cluster = MeanShift(bandwidth=bandwidth)
                labels = cluster.fit_predict(X_train_reduced)
                unique_labels = np.unique(labels)
                centroids = []
                for label in unique_labels:
                    centroid = np.mean(X_train_reduced[labels == label], axis=0)
                    centroids.append(centroid)
                distorsion = sum(np.min(cdist(X_train_reduced, centroids, 'euclidean'), axis=1)) / np.size(
                    X_train_reduced, axis=0)
                distorsions.append(distorsion)
                axes.append(bandwidth)

    return axes, distorsions


@st.cache
def reduce_sample(X_train_reduced):
    # On échantillonne les données pour limiter le calcul
    sample_size = 500  # Taille de l'échantillon
    indices = np.random.choice(len(X_train_reduced), size=sample_size, replace=False)
    X_train_reduced_sample = X_train_reduced[indices]
    return X_train_reduced_sample

@st.cache
def reduce_y_train(y_train, sample_size=50):
    indices = np.random.choice(len(y_train), size=sample_size, replace=False)
    y_train_sample = y_train[indices]
    return y_train_sample

@st.cache        
def display_clusters(methode_choisie, X_train_reduced):
    X_train_reduced = reduce_sample(X_train_reduced)
    if methode_choisie == "K-means":
        model = KMeans(n_clusters=8)
        model.fit_predict(X_train_reduced)
    elif methode_choisie == "Clustering Hiérarchique":
        model = AgglomerativeClustering(n_clusters=6)
        model.fit_predict(X_train_reduced)
    elif methode_choisie == "Mean Shift":
        model = MeanShift(bandwidth=0.5)
        model.fit_predict(X_train_reduced)
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X_train_reduced[:, 0], X_train_reduced[:, 1], c=model.labels_, cmap=plt.cm.Spectral)
    ax.set_xlabel('Axe 1')
    ax.set_ylabel('Axe 2')
    ax.set_title('Visualisation des clusters ')
    st.pyplot(fig)  