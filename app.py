import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import matplotlib.pyplot as plt

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
df = df.dropna()

# -----------------------------
# APP TITLE
# -----------------------------
st.title("🌍 AI Travel Planner System")
st.write("Plan smart trips using AI + rules + analytics")

# -----------------------------
# INPUTS
# -----------------------------
budget = st.number_input("💰 Budget (₹)", 1000, 20000, 5000, 500)
duration = st.slider("📅 Duration (Days)", 1, 10, 3)
rating = st.slider("⭐ Minimum Rating", 3.0, 5.0, 4.0)

destination = st.selectbox("📍 Destination", list(le_dest.classes_))
location = st.selectbox("🏷 Location Type", list(le_loc.classes_))

# -----------------------------
# SCORING FUNCTION
# -----------------------------
def compute_score(row, budget):
    cost_score = max(0, 1 - (row["Cost"] / budget))
    rating_score = row["Rating"] / 5
    return (0.5 * rating_score) + (0.5 * cost_score)

# -----------------------------
# ITINERARY GENERATOR
# -----------------------------
def generate_itinerary(df, days):
    itinerary = {}
    sample = df.sample(min(days, len(df))).reset_index(drop=True)

    for i in range(days):
        row = sample.iloc[i % len(sample)]

        itinerary[f"Day {i+1}"] = {
            "Destination": row["Destination"],
            "Category": row["Category"],
            "Cost": row["Cost"],
            "Rating": row["Rating"],
            "Plan": random.choice([
                "Local sightseeing & food tour",
                "Adventure activities",
                "Historical exploration",
                "Nature walk & relaxation",
                "City tour & shopping"
            ])
        }

    return itinerary

# -----------------------------
# MAIN BUTTON
# -----------------------------
if st.button("Generate Travel Plan"):

    try:
        # Encode inputs
        dest_enc = le_dest.transform([destination])[0]
        loc_enc = le_loc.transform([location])[0]

        # ML Prediction (soft suggestion)
        input_data = np.array([[dest_enc, loc_enc, budget, duration, rating]])
        pred = model.predict(input_data)[0]
        predicted_category = le_cat.inverse_transform([pred])[0]

        st.success(f"🧠 Suggested Category: {predicted_category}")

        # -----------------------------
        # RULE-BASED FILTERING
        # -----------------------------
        filtered = df[
            (df["Cost"] <= budget) &
            (df["Duration"] <= duration) &
            (df["Rating"] >= rating)
        ]

        # Category priority
        final_plan = filtered[filtered["Category"] == predicted_category]

        if final_plan.empty:
            final_plan = filtered

        if final_plan.empty:
            st.warning("No trips found. Try increasing budget/duration.")
        else:

            # -----------------------------
            # SCORING
            # -----------------------------
            final_plan["Score"] = final_plan.apply(lambda x: compute_score(x, budget), axis=1)
            final_plan = final_plan.sort_values(by="Score", ascending=False)

            st.subheader("🧳 Top Travel Plans")
            st.dataframe(final_plan.head(10))

            # -----------------------------
            # ITINERARY
            # -----------------------------
            st.subheader("📅 Day-wise Itinerary")
            itinerary = generate_itinerary(final_plan, duration)

            for day, details in itinerary.items():
                st.markdown(f"### {day}")
                st.write(details)

            # -----------------------------
            # VISUALIZATION
            # -----------------------------
            st.subheader("📊 Insights")

            fig, ax = plt.subplots()
            final_plan["Category"].value_counts().plot(kind="bar", ax=ax)
            st.pyplot(fig)

            fig2, ax2 = plt.subplots()
            ax2.hist(final_plan["Cost"], bins=10)
            st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error: {str(e)}")
