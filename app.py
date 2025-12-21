import streamlit as st
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="AI Disease Prediction Platform",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


SYMPTOMS = [
    "fever","cough","headache","fatigue","nausea","sore_throat",
    "runny_nose","body_aches","chills","shortness_of_breath",
    "chest_pain","dizziness","abdominal_pain","vomiting",
    "diarrhea","sneezing","itchy_eyes","joint_pain",
    "frequent_urination","excessive_thirst","blurred_vision"
]

DISEASE_SYMPTOMS = {
    "Common Cold": ["runny_nose","sneezing","sore_throat","cough"],
    "Influenza": ["fever","cough","headache","body_aches","fatigue"],
    "Migraine": ["headache","nausea","dizziness"],
    "Allergic Rhinitis": ["sneezing","itchy_eyes","runny_nose"],
    "Gastroenteritis": ["nausea","vomiting","diarrhea","abdominal_pain"],
    "Bronchitis": ["cough","fatigue","shortness_of_breath"],
    "Sinusitis": ["headache","runny_nose","fever"],
    "Urinary Tract Infection": ["frequent_urination","abdominal_pain"],
    "Pneumonia": ["fever","cough","shortness_of_breath","chest_pain"],
    "Hypertension": ["headache","dizziness"],
    "Diabetes": ["excessive_thirst","frequent_urination","fatigue"],
    "Asthma": ["cough","shortness_of_breath","chest_pain"],
    "Arthritis": ["joint_pain","fatigue"]
}


@st.cache_data
def generate_dataset(n_samples=3500):
    data = []
    for _ in range(n_samples):
        disease = np.random.choice(list(DISEASE_SYMPTOMS.keys()))
        row = [0] * len(SYMPTOMS)
        # Primary symptoms (high probability)
        for s in DISEASE_SYMPTOMS[disease]:
            if np.random.rand() < 0.85:
                row[SYMPTOMS.index(s)] = 1
        # Noise symptoms (low probability)
        for i in range(len(row)):
            if row[i] == 0 and np.random.rand() < 0.1:
                row[i] = 1
        row.append(disease)
        data.append(row)
    df = pd.DataFrame(data, columns=SYMPTOMS + ["disease"])
    return df


@st.cache_resource
def train_model():
    df = generate_dataset()
    encoder = LabelEncoder()
    df["disease_encoded"] = encoder.fit_transform(df["disease"])
    X = df[SYMPTOMS]
    y = df["disease_encoded"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=350, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, encoder, accuracy

MODEL, LABEL_ENCODER, MODEL_ACCURACY = train_model()


def calculate_risk(symptom_count):
    if symptom_count <= 3:
        return "Low Risk", 30 + symptom_count * 5
    elif symptom_count <= 7:
        return "Medium Risk", 50 + symptom_count * 4
    else:
        return "High Risk", 75 + symptom_count * 3


st.sidebar.title("🧠 AI Healthcare Platform")
PAGE = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Dashboard",
        "🩺 Disease Prediction",
        "📊 Health Insights",
        "🧠 Model & Training",
        "🏗 System Architecture",
        "ℹ️ About"
    ]
)


if PAGE == "🏠 Dashboard":
    st.title("🩺 AI Disease Prediction Platform")
    st.markdown("""
    **An AI-powered healthcare decision support system**  
    leveraging **machine learning** to predict diseases from symptoms.
    """)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ML Algorithm", "Random Forest 🌲")
    col2.metric("Diseases Modeled", len(DISEASE_SYMPTOMS))
    col3.metric("Model Accuracy", f"{MODEL_ACCURACY*100:.2f}% 📊")
    col4.metric("Deployment Type", "Interactive Streamlit App 🚀")
    st.info(
        "This platform demonstrates how **data-driven models** can assist "
        "in healthcare decision-making by analyzing symptoms and predicting possible diseases."
    )

elif PAGE == "🩺 Disease Prediction":
    st.title("🔍 Disease Prediction")
    st.markdown("""
    Select your symptoms below and let the AI system predict the most likely disease.  
    The platform also provides a **health risk score** to indicate severity.
    """)
    selected_symptoms = st.multiselect("Choose Symptoms", SYMPTOMS)
    if st.button("Predict Disease"):
        if not selected_symptoms:
            st.warning("Please select at least one symptom.")
        else:
            input_vector = [1 if s in selected_symptoms else 0 for s in SYMPTOMS]
            prediction = MODEL.predict([input_vector])[0]
            disease = LABEL_ENCODER.inverse_transform([prediction])[0]
            risk_level, risk_score = calculate_risk(len(selected_symptoms))

            # Display prediction
            st.success(f"🧾 Predicted Disease: **{disease}**")
            st.info(f"⚠️ Risk Level: **{risk_level}** | Risk Score: **{risk_score}/100**")
            st.caption("💡 Prediction is based on symptom patterns analyzed by a Random Forest ML model.")

            # Explain risk levels
            st.markdown("### ⚠️ Risk Level Explanation")
            st.markdown("""
            - **Low Risk:** 1–3 symptoms selected – minimal likelihood of severe disease  
            - **Medium Risk:** 4–7 symptoms selected – moderate likelihood, monitor symptoms  
            - **High Risk:** 8 or more symptoms selected – high likelihood, consider consulting a healthcare professional  

            The **Risk Score** provides a numeric estimate of severity based on the number of symptoms.
            """)

elif PAGE == "📊 Health Insights":
    st.title("📊 Health Risk Insights")
    st.markdown("""
    This page explains the **ML reasoning and risk assessment methodology** behind predictions:
    - Number of reported symptoms
    - Symptom co-occurrence patterns
    - Likelihood of disease based on historical data
    """)
    st.info("💡 Provides insight into how AI models analyze symptoms to make healthcare predictions.")

elif PAGE == "🧠 Model & Training":
    st.title("🧠 Model Training & Deployment Details")
    st.markdown("""
    ### Machine Learning Model
    **Random Forest Classifier** – robust and interpretable

    ### Training Process
    - Generated **synthetic symptom-based dataset**
    - Features encoded as binary vectors (0/1)
    - Target labels encoded using LabelEncoder
    - Train-test split: 80% / 20%

    ### Deployment
    - Model trained at runtime and cached
    - Instant predictions via Streamlit
    - Designed for real-time decision support
    """)
    st.caption("💡 Shows a complete ML workflow: dataset → training → deployment → inference")


elif PAGE == "🏗 System Architecture":
    st.title("🏗 System Architecture")
    st.code("""
User (Web Browser)
        ↓
Streamlit User Interface
        ↓
Symptom Input & Validation
        ↓
Feature Encoding (Binary Vector)
        ↓
Random Forest ML Model
        ↓
Disease Prediction
        ↓
Risk Assessment
        ↓
Result Visualization
    """)
    st.info("💡 Illustrates a layered AI architecture for symptom-based disease prediction.")


elif PAGE == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    **Project Name:** AI Disease Prediction Platform  
    **Domain:** Healthcare & Artificial Intelligence  
    **Technologies:** Python, Streamlit, Scikit-learn  

    ### Highlights
    - Multi-page ML web application
    - Real-time symptom-based disease prediction
    - Risk-based health assessment
    - Demonstrates practical deployment of Random Forest in healthcare

    ⚠️ *Designed for AI experimentation and learning purposes.
    """)


st.divider()
st.caption(
    "Turning symptom data into actionable health intelligence."
)

