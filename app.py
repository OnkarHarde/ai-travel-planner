import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------- LOAD ----------------
model = joblib.load("travel_recommendation_model.pkl")
le_dest = joblib.load("encoder_destination.pkl")
le_loc = joblib.load("encoder_location.pkl")
le_cat = joblib.load("encoder_category.pkl")

df = pd.read_csv("cleaned_travel_data.csv")

st.title("🌍 AI Travel Planner System")

# ---------------- SAFE INPUTS ----------------
destinations = list(le_dest.classes_)
locations = list(le_loc.classes_)

budget = st.number_input("Budget", 1000, 20000, 5000)
duration = st.slider("Duration (days)", 1, 10, 3)
rating = st.slider("Rating", 3.0, 5.0, 4.0)

destination = st.selectbox("Destination", destinations)
location = st.selectbox("Location", locations)

# ---------------- PREDICTION ----------------
if st.button("Recommend"):

    try:
        dest_enc = le_dest.transform([destination])[0]
        loc_enc = le_loc.transform([location])[0]

        input_data = np.array([[dest_enc, loc_enc, budget, duration, rating]])

        prediction = model.predict(input_data)[0]
        category = le_cat.inverse_transform([prediction])[0]

        st.success(f"🎯 Recommended Category: {category}")

        results = df[df["Category"] == category]
        st.dataframe(results.head(5))

    except Exception as e:
        st.error(f"Error: {str(e)}")
