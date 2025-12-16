import streamlit as st
import numpy as np
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Disease Prediction System")
st.markdown("Predict diseases based on symptoms using **Machine Learning**")

# ---------------- LOAD PICKLES ----------------
@st.cache_resource
def load_files():
    model = pickle.load(open("model.pkl", "rb"))
    le = pickle.load(open("label_encoder.pkl", "rb"))
    symptoms = pickle.load(open("symptoms.pkl", "rb"))
    return model, le, symptoms

model, label_encoder, symptoms = load_files()

# ---------------- SIDEBAR ----------------
st.sidebar.title("📂 Navigation")
menu = st.sidebar.radio("Go to", ["Disease Prediction", "Project Details", "About"])

# ---------------- PREDICTION PAGE ----------------
if menu == "Disease Prediction":
    st.subheader("🔍 Select Symptoms")

    selected_symptoms = st.multiselect(
        "Choose the symptoms you are experiencing:",
        symptoms
    )

    if st.button("🔮 Predict Disease"):
        if not selected_symptoms:
            st.warning("Please select at least one symptom")
        else:
            input_vector = [1 if s in selected_symptoms else 0 for s in symptoms]
            input_array = np.array(input_vector).reshape(1, -1)

            prediction = model.predict(input_array)
            disease = label_encoder.inverse_transform(prediction)[0]

            probs = model.predict_proba(input_array)[0]
            top = np.argsort(probs)[-3:][::-1]

            st.success(f"🧾 **Predicted Disease:** {disease}")
            st.subheader("📊 Top 3 Predictions")

            for i in top:
                st.write(f"• **{label_encoder.inverse_transform([i])[0]}** : {probs[i]*100:.2f}%")

# ---------------- DETAILS ----------------
elif menu == "Project Details":
    st.subheader("📌 Project Details")
    st.write("""
    - Algorithm: Random Forest
    - Domain: Healthcare
    - Input: Symptoms
    - Output: Disease prediction
    - Model loaded from pickle file
    """)

# ---------------- ABOUT ----------------
elif menu == "About":
    st.subheader("ℹ️ About")
    st.write("""
    Academic ML mini-project using Streamlit.
    Predicts diseases using a trained ML model.
    """)

st.divider()
st.caption("Machine Learning Based Disease Prediction System")
