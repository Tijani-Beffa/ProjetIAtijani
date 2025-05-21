import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Titre de l'application
st.title("Projet IA - Prédiction")

# Charger le modèle sauvegardé
@st.cache_resource
def load_model():
    return joblib.load('modele_final.pkl')

model = load_model()

# Upload du fichier CSV
uploaded_file = st.file_uploader("Choisissez un fichier CSV avec les données d'entrée", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Données chargées :")
    st.dataframe(data.head())

    # Prétraitement simple (adapter selon ton projet)
    # Par exemple, on suppose que data est prêt à la prédiction
    try:
        predictions = model.predict(data)
        st.write("Prédictions :")
        st.write(predictions)
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
else:
    st.info("Veuillez uploader un fichier CSV pour faire la prédiction.")
