import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Charger le modèle et le tokenizer
model = load_model("sms_lstm_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Fonction de prédiction
def predict_sms(sms):
    sms_seq = tokenizer.texts_to_sequences([sms])
    sms_pad = pad_sequences(sms_seq, maxlen=50)
    prediction = model.predict(sms_pad)
    return "AGRESSIF" if prediction[0][0] > 0.5 else "NON AGRESSIF"

# Interface Streamlit
st.title("Détection d'Agressivité des SMS 📱")
st.write("Entrez un SMS et obtenez une prédiction !")

# Champ d'entrée utilisateur
sms_input = st.text_area("Tapez votre SMS ici")

# Bouton de prédiction
if st.button("Prédire"):
    if sms_input.strip():
        resultat = predict_sms(sms_input)
        st.subheader(f"Résultat : {resultat}")
    else:
        st.warning("Veuillez entrer un SMS avant de prédire !")
