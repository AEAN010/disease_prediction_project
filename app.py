import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Disease Prediction System")
st.markdown("Predict diseases based on symptoms using **Machine Learning (Supervised Learning)**")


@st.cache_data
def create_dataset(n_samples=2000):

    diseases = [
        'Common Cold','Influenza','Migraine','Allergic Rhinitis',
        'Gastroenteritis','Bronchitis','Sinusitis',
        'Urinary Tract Infection','Pneumonia',
        'Hypertension','Diabetes','Asthma','Arthritis'
    ]

    symptoms = [
        'fever','cough','headache','fatigue','nausea',
        'sore_throat','runny_nose','body_aches','chills',
        'shortness_of_breath','chest_pain','dizziness',
        'abdominal_pain','vomiting','diarrhea','sneezing',
        'itchy_eyes','joint_pain','frequent_urination',
        'excessive_thirst','blurred_vision'
    ]

    disease_symptoms = {
        'Common Cold':['runny_nose','sneezing','sore_throat','cough'],
        'Influenza':['fever','cough','headache','body_aches','fatigue'],
        'Migraine':['headache','nausea','dizziness'],
        'Allergic Rhinitis':['sneezing','itchy_eyes','runny_nose'],
        'Gastroenteritis':['nausea','vomiting','diarrhea','abdominal_pain'],
        'Bronchitis':['cough','fatigue','shortness_of_breath'],
        'Sinusitis':['headache','runny_nose','fever'],
        'Urinary Tract Infection':['frequent_urination','abdominal_pain'],
        'Pneumonia':['fever','cough','shortness_of_breath','chest_pain'],
        'Hypertension':['headache','dizziness'],
        'Diabetes':['excessive_thirst','frequent_urination','fatigue'],
        'Asthma':['cough','shortness_of_breath','chest_pain'],
        'Arthritis':['joint_pain','fatigue']
    }

    data = []

    for _ in range(n_samples):
        disease = np.random.choice(diseases)
        row = [0] * len(symptoms)

        for s in disease_symptoms[disease]:
            if np.random.rand() < 0.8:
                row[symptoms.index(s)] = 1

        for i in range(len(symptoms)):
            if row[i] == 0 and np.random.rand() < 0.1:
                row[i] = 1

        row.append(disease)
        data.append(row)

    df = pd.DataFrame(data, columns=symptoms + ['disease'])
    return df, symptoms


@st.cache_resource
def train_model():
    df, symptoms = create_dataset()

    le = LabelEncoder()
    df['disease_encoded'] = le.fit_transform(df['disease'])

    X = df[symptoms]
    y = df['disease_encoded']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, le, symptoms, accuracy

model, label_encoder, symptoms, accuracy = train_model()


st.sidebar.title("📂 Navigation")

menu = st.sidebar.radio(
    "Go to",
    ["Disease Prediction", "Project Details", "About"]
)


if menu == "Disease Prediction":

    st.subheader("🔍 Select Symptoms")

    selected_symptoms = st.multiselect(
        "Choose the symptoms you are experiencing:",
        symptoms
    )

    if st.button("🔮 Predict Disease"):

        if len(selected_symptoms) == 0:
            st.warning("⚠️ Please select at least one symptom.")
        else:
            input_vector = [1 if s in selected_symptoms else 0 for s in symptoms]
            input_array = np.array(input_vector).reshape(1, -1)

            prediction = model.predict(input_array)
            predicted_disease = label_encoder.inverse_transform(prediction)[0]

            probabilities = model.predict_proba(input_array)[0]
            top_indices = np.argsort(probabilities)[-3:][::-1]

            st.success(f"🧾 **Predicted Disease:** {predicted_disease}")

            st.subheader("📊 Top 3 Possible Diseases")
            for idx in top_indices:
                st.write(
                    f"• **{label_encoder.inverse_transform([idx])[0]}** "
                    f": {probabilities[idx]*100:.2f}%"
                )


elif menu == "Project Details":

    st.subheader("📌 Project Details")

    st.write(f"""
    **Project Name:** Disease Prediction System  
    **Domain:** Healthcare + Machine Learning  
    **Algorithm Used:** Random Forest Classifier  
    **Learning Type:** Supervised Learning  
    **Dataset:** Synthetic symptom-based dataset  
    **Model Accuracy:** {accuracy:.2f}
    """)

    st.write("""
    ### 🎯 Objective
    To predict possible diseases based on user-selected symptoms
    using machine learning techniques.

    ### 🧠 Why Random Forest?
    - High accuracy
    - Handles multiple features
    - Reduces overfitting
    """)


elif menu == "About":

    st.subheader("ℹ️ About This Application")

    st.write("""
    This **Disease Prediction System** is an academic mini-project
    developed using **Python, Streamlit, and Machine Learning**.

    ### 🚀 Features
    - Interactive UI
    - Multiple disease prediction
    - Probability-based output
    - Single-file Streamlit application
    """)

    st.info("Machine Learning–Based Disease Prediction for Early Diagnosis")


st.divider()
st.caption("A Data-Driven Approach to Disease Prediction Using ML Models.")

