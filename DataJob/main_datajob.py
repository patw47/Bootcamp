# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 11:59:41 2023

@author: PatriciaWintrebert
"""

import streamlit as st

st.set_page_config(
    page_title="Projet Datajobp",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("# 💻 Projet Datajob")
st.sidebar.markdown("# 💻 Projet Datajob")


st.header('Contexte')

st.subheader("Contexte du projet d'un point de vue'métier et technique")

st.write('le contenu de ce rapport aura comme objectif de mettre en avant les différents profils métiers data et machine learning sur la base de de la colonne  Q5 “Select the title most similar to your current role (or most recent title if retired)” renommé “TITLE”. Ce rapport sera à destination principalement  des professionnels du recrutement et permettra d’identifier en fonction du métier, entre autres,  les différents outils et programmes utilisés et depuis combien de temps, l’ancienneté, la taille des entreprises employés')

st.write('Ce rapport sera à destination principalement  des professionnels du recrutement et permettra d’identifier en fonction du métier, entre autres,  les différents outils et programmes utilisés et depuis combien de temps, l’ancienneté, la taille des entreprises employés')

st.write('Le projet "Data Job” consistera à traiter les données du dataset “kaggle_survey_2020_responses” au format csv (disponible sur le site internet kaggle)')

st.markdown("[Lien vers les données sur le site de Kaggle](https://www.kaggle.com/c/kaggle-survey-2020/overview')")

st.write('Ce fichier a été constitué des 20 036  réponses d’une enquête menée en ligne par le site internet kaggle auprès de ses membres')

st.subheader('Méthodologie de Kaggle pour récolter les données')

st.write('L’enquête Kaggle DS & ML 2020 a reçu 20 036 réponses utilisables de participants dans 171 différents pays et territoires. Si un pays ou territoire a reçu moins de 50 répondants, nous les avons regroupés dans un groupe nommé « Autre » pour l’anonymat')

st.write('Une invitation à participer à l’enquête a été envoyée à l’ensemble de la communauté Kaggle (n’importe qui s’est inscrit à la liste de diffusion Kaggle). L’enquête a également été promue sur le site Web de Kaggle et sur la chaîne Twitter de Kaggle')

st.write('L’enquête a été réalisée en direct du 07/10/2020 au 30/10/2020. Nous avons permis aux répondants de remplir le questionnaire à tout moment au cours de cette période')

st.write("Les réponses aux questions à choix multiples (un seul choix peut être sélectionné) ont été consignées dans des colonnes individuelles. Les réponses à plusieurs questions de l'enquête (où plusieurs choix peuvent être sélectionnés) ont été divisés en plusieurs colonnes (avec une colonne par choix de réponse).")

st.write("Pour protéger la vie privée des répondants, les réponses textuelles en format libre n’ont pas été incluses dans le jeu de données de l’enquête et l’ordre des lignes a été mélangé (les réponses ne sont pas affichées dans ordre chronologique)")