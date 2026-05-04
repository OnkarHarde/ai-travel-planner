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
# LOAD DATA
# -----------------------------
df = pd.read_csv("cleaned_travel_data.csv")

# Safety: remove any NaN rows just in case
df = df.dropna()

# -----------------------------
# APP UI
# -----------------------------
st.title("🧭 AI Travel Planner System")
st.write("Plan your trip using AI + smart filtering")

# -----------------------------
# SAFE INPUT OPTIONS
# -----------------------------
dest_options = list(le_dest.classes_)
loc_options = list(le_loc.classes_)

# -----------------------------
# USER INPUTS
# -----------------------------
budget = st.number_input("💰 Budget (₹)", min_value=1000, max_value=20000, value=5000, step=500)
duration = st.slider("📅 Duration (Days)", 1, 10, 3)
rating = st.slider("⭐ Minimum Rating", 3.0, 5.0, 4.0)

destination = st.selectbox("📍 Destination", dest_options)
location = st.selectbox("🏷 Location Type", loc_options)

# -----------------------------
# GENERATE PLAN
# -----------------------------
if st.button("Generate Travel Plan"):

    try:
        # Encode inputs safely
        dest_enc = le_dest.transform([destination])[0]
        loc_enc = le_loc.transform([location])[0]

        # Ensure correct feature order
        input_data = np.array([[dest_enc, loc_enc, budget, duration, rating]])

        # Predict category (ML suggestion only)
        pred = model.predict(input_data)[0]
        predicted_category = le_cat.inverse_transform([pred])[0]

        st.success(f"🧠 Suggested Category: {predicted_category}")

        # -----------------------------
        # RULE-BASED FILTERING (PLANNER CORE)
        # -----------------------------
        filtered = df[
            (df["Cost"] <= budget) &
            (df["Duration"] <= duration) &
            (df["Rating"] >= rating)
        ]

        # Prioritize ML category
        final_plan = filtered[filtered["Category"] == predicted_category]

        # Fallback if empty
        if final_plan.empty:
            final_plan = filtered

        # -----------------------------
        # OUTPUT
        # -----------------------------
        st.subheader("🧳 Your Travel Plan")

        if final_plan.empty:
            st.warning("No matching trips found. Try increasing budget or duration.")
        else:
            st.dataframe(
                final_plan.sort_values(by="Rating", ascending=False).head(10)
            )

    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")
