import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# LOAD MODEL + ENCODERS
# -----------------------------
model = joblib.load("travel_recommendation_model.pkl")
le_dest = joblib.load("encoder_destination.pkl")
le_loc = joblib.load("encoder_location.pkl")
le_cat = joblib.load("encoder_category.pkl")

# -----------------------------
# LOAD DATA (for recommendations)
# -----------------------------
df = pd.read_csv("cleaned_travel_data.csv")

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🌍 AI Travel Planner System")
st.write("Get personalized travel category recommendations using Machine Learning")

# -----------------------------
# USER INPUTS
# -----------------------------
budget = st.number_input("Enter Your Budget (₹)", min_value=1000, max_value=20000, step=500)
duration = st.slider("Trip Duration (Days)", 1, 10, 3)
rating = st.slider("Preferred Rating", 3.0, 5.0, 4.0)

destination = st.selectbox("Select Destination", df["Destination"].unique())
location = st.selectbox("Select Location Type", df["Location"].unique())

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.button("Get Recommendation"):

    # Encode inputs
    dest_enc = le_dest.transform([destination])[0]
    loc_enc = le_loc.transform([location])[0]

    # Create feature vector
    input_data = np.array([[dest_enc, loc_enc, budget, duration, rating]])

    # Predict category
    prediction = model.predict(input_data)[0]
    category = le_cat.inverse_transform([prediction])[0]

    # -----------------------------
    # OUTPUT
    # -----------------------------
    st.success(f"🎯 Recommended Travel Category: {category}")

    # -----------------------------
    # FILTER REAL OPTIONS
    # -----------------------------
    results = df[df["Category"] == category]

    st.subheader("📍 Top Travel Options:")

    st.dataframe(results.head(5))

    # -----------------------------
    # SIMPLE INSIGHT
    # -----------------------------
    st.info("These recommendations are based on ML prediction from your preferences.")
