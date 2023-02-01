import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv("pet_diseases.csv")

# One-hot encode the string-based features
encoder = OneHotEncoder()
X = encoder.fit_transform(df[["symptom1", "symptom2", "symptom3", "symptom4"]])
y = df["disease"]

# Train a random forest classifier
model = RandomForestClassifier()
model.fit(X, y)

# Define the app
def predict_pet_disease():
    st.set_page_config(page_title='Vetcare')
    st.title("Pet Disease Predictor")

    symptom_1 = st.selectbox("Symptom 1", df["symptom1"].unique())
    symptom_2 = st.selectbox("Symptom 2", df["symptom2"].unique())
    symptom_3 = st.selectbox("Symptom 3", df["symptom3"].unique())
    symptom_4 = st.selectbox("Symptom 4", df["symptom4"].unique())

    symptoms = np.array([[symptom_1, symptom_2, symptom_3, symptom_4]])
    symptoms_encoded = encoder.transform(symptoms)

    prediction = model.predict(symptoms_encoded)
    prediction_proba = model.predict_proba(symptoms_encoded)[0]
    class_names = model.classes_

    st.write("Disease:", prediction[0])
    st.sidebar.write("Probabilities:")
    for class_name, proba in zip(class_names, prediction_proba):
        st.sidebar.write("- {}: {:.2f}%".format(class_name, proba * 100))

# Run the app
if __name__ == "__main__":
    predict_pet_disease()