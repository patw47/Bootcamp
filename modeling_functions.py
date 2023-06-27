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
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
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
    
def grid_search_model(model, param_grid, X_train_reduced, X_test_reduced, y_train, y_test):
    """
    Effectue une recherche sur la grille des hyperparamètres pour un modèle donné.

    Arguments :
    - model : Objet modèle à entraîner et à évaluer.
    - param_grid : Dictionnaire des hyperparamètres à tester.
    - X_train : Matrice des caractéristiques d'entraînement.
    - X_test : Matrice des caractéristiques de test.
    - y_train : Vecteur des étiquettes d'entraînement.
    - y_test : Vecteur des étiquettes de test.

    Retourne :
    - best_model : Meilleur modèle trouvé lors de la recherche sur la grille.
    - y_pred : Prédictions du meilleur modèle sur l'ensemble de test.
    - best_params : Meilleurs hyperparamètres trouvés lors de la recherche sur la grille.
    """

    # Recherche des meilleures combinaisons d'hyperparamètres
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train_reduced, y_train)

    # Meilleur modèle trouvé
    best_model = grid_search.best_estimator_

    # Évaluation du meilleur modèle sur les données de test
    y_pred = best_model.predict(X_test_reduced)

    # Renvoie le meilleur modèle, les prédictions et les meilleurs hyperparamètres
    return best_model, y_test, y_pred, grid_search.best_params_

def grid_search_params(best_model, param_grid, X_train_reduced, X_test_reduced, y_train, y_test):
    grid_search = GridSearchCV(estimator=best_model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train_reduced, y_train)
    # Évaluation du  modèle sur les données de test
    y_pred = best_model.predict(X_test_reduced)
    # Renvoie les prédictions et les meilleurs hyperparamètres
    return grid_search.best_params_, y_test, y_pred

def find_optimal_clusters_tsne(X, max_clusters):
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)
    
    inertias = []
    for n_clusters in range(2, max_clusters+1):
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(X_tsne)
        inertias.append(kmeans.inertia_)
    
    plt.plot(range(2, max_clusters+1), inertias)
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Inertie')
    plt.title('Méthode du coude (Elbow method)')
    st.pyplot(plt.gcf())
    
    optimal_clusters = st.number_input("Entrez le nombre optimal de clusters selon la méthode du coude :", min_value=2, max_value=max_clusters, step=1)
    
    return optimal_clusters


def find_optimal_clusters_kmeans(X, max_clusters):
    silhouette_scores = []
    for n_clusters in range(2, max_clusters+1):
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, labels))
    
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # +2 pour compenser le range de 2 à max_clusters
    
    return optimal_clusters


def train_model(model_choisi, X_train_reduced, y_train, X_test_reduced, y_test):
    model = None
    axes = None
    
    if model_choisi == 'Régression logistique':
        model = LogisticRegression()
    elif model_choisi == "Arbre de décision":
        model = DecisionTreeClassifier()
    elif model_choisi == "KNN":
        model = KNeighborsClassifier()
    elif model_choisi == "Forêt aléatoire":
        model = RandomForestClassifier()
    elif model_choisi == "K-means":
        optimal_clusters = find_optimal_clusters_kmeans(X_train_reduced, max_clusters=10)
        model = KMeans(n_clusters=optimal_clusters)
        axes = model.fit_transform(X_train_reduced)
        labels = model.predict(X_train_reduced)
    elif model_choisi == "t_SNE":
        tsne = TSNE(n_components=2)
        optimal_clusters = find_optimal_clusters_kmeans(X_train_reduced, max_clusters=10)
        axes = tsne.fit_transform(X_train_reduced)
        labels = KMeans(optimal_clusters).fit_predict(X_train_reduced)  # Ou utilisez le nombre de clusters souhaité
        plot_clusters(axes, labels)
  
    if model_choisi != "K-means" and model_choisi != "t_SNE":
        model.fit(X_train_reduced, y_train)
        score = model.score(X_test_reduced, y_test)
        axes = 0
        return score, model, axes
    
    else:
        score = 0
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(axes[:, 0], axes[:, 1], c=labels, cmap=plt.cm.Spectral)
        ax.set_xlabel('Axe 1')
        ax.set_ylabel('Axe 2')
        ax.set_title(model_choisi)
        st.pyplot(fig)
        
        return score, model, axes
    
def plot_clusters(axes, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(axes[:, 0], axes[:, 1], c=labels, cmap=plt.cm.Spectral)
    ax.set_xlabel('Axe 1')
    ax.set_ylabel('Axe 2')
    ax.set_title('Projection t-SNE avec couleurs de cluster')
    return fig

def plot_elbow_curve(X, max_clusters):
    inertias = []
    for n_clusters in range(2, max_clusters+1):
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    plt.plot(range(2, max_clusters+1), inertias)
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Inertie')
    plt.title('Méthode du coude (Elbow method)')
    st.pyplot(plt.gcf())
        
        
        
    
def get_param_grid(model_choisi):
    """
    Retourne le dictionnaire param_grid approprié pour le modèle donné.

    Arguments :
    - model_name : Nom du modèle (LinearRegression, KNN, DecisionTree, RandomForest).

    Retourne :
    - param_grid : Dictionnaire des hyperparamètres à tester pour le modèle donné.
    """
    if model_choisi == 'Régression logistique':
        model = LogisticRegression()
        param_grid = {
            'C': [0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
            
        }
    elif model_choisi == 'KNN':
        model = KNeighborsClassifier()
        param_grid = {
            'n_neighbors': range(3, 8, 2),
            'weights': ['uniform', 'distance']
        }
    elif model_choisi == 'Arbre de décision':
        model = DecisionTreeClassifier()
        param_grid = {
            'max_depth': [None, 5, 10],
            'min_samples_split': range(2, 12, 3)
        }
    elif model_choisi == 'Forêt aléatoire':
        model = RandomForestClassifier()
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10],
            'min_samples_split': range(2, 12, 3)
        }
    elif model_choisi == 'K-means Clustering':
        model = KMeans()
        param_grid = {
            'n_clusters': range(2, 8),
            'init': ['k-means++', 'random'],
            'max_iter': range(100, 501, 100)
        }
    else:
        raise ValueError('Modèle non pris en charge.')

    return param_grid, model

def train_best_model(best_model, best_params, X_train_reduced, X_test_reduced, y_train, y_test):
    best_model.set_params(**best_model.best_params_)
    best_model.fit(X_train_reduced, y_train)
    score_best_model = best_model.score(X_test_reduced, y_test)
    return score_best_model
    