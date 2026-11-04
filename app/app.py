import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Risque de départ client", page_icon="📉")

@st.cache_resource
def charger():
    return joblib.load("models/modele_churn.joblib")

bundle = charger()
modele = bundle["modele"]
colonnes = bundle["colonnes"]
seuil = bundle["seuil"]
defauts = bundle["defauts"]

st.title("📉 Risque de départ d'un client")
st.write("Renseignez le profil d'un client pour estimer sa probabilité de résilier.")

col1, col2 = st.columns(2)
with col1:
    anciennete = st.slider("Ancienneté (mois)", 0, 72, 12)
    mensuel = st.slider("Facture mensuelle (USD)", 18.0, 120.0, 70.0)
    contrat = st.selectbox("Type de contrat", ["Month-to-month", "One year", "Two year"])
with col2:
    internet = st.selectbox("Type d'internet", ["Fiber optic", "DSL", "No"])
    paiement = st.selectbox("Mode de paiement", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])
    senior = st.checkbox("Client senior")

if st.button("Calculer le risque"):
    ligne = dict(defauts)
    ligne.update({
        "tenure": anciennete,
        "MonthlyCharges": mensuel,
        "TotalCharges": anciennete * mensuel,
        "Contract": contrat,
        "InternetService": internet,
        "PaymentMethod": paiement,
        "SeniorCitizen": int(senior),
    })

    entree = pd.get_dummies(pd.DataFrame([ligne]), dtype=int)
    entree = entree.reindex(columns=colonnes, fill_value=0)

    proba = modele.predict_proba(entree)[0, 1]
    st.metric("Probabilité de départ", f"{proba:.0%}")

    if proba >= seuil:
        st.error(f"Client à risque (au-dessus du seuil de {seuil:.0%}) : proposer une offre de fidélisation.")
    else:
        st.success(f"Client peu à risque (sous le seuil de {seuil:.0%}).")

st.caption("Modèle : régression logistique entraînée sur le jeu Telco Customer Churn. "
           "Les champs non affichés sont fixés à la valeur la plus fréquente. "
           "Estimation indicative : lien statistique, pas causalité.")
