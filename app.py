import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium

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
df = pd.read_csv("cleaned_travel_data.csv").dropna()

# -----------------------------
# UI CONFIG
# -----------------------------
st.set_page_config(page_title="AI Travel Planner", layout="wide")

st.title("🌍🧭 AI Travel Planner System (With Map)")
st.write("Plan trips using AI + filters + interactive map")

# -----------------------------
# SIDEBAR INPUTS (clean UI)
# -----------------------------
st.sidebar.header("🎛 Your Preferences")

budget = st.sidebar.number_input("💰 Budget (₹)", 1000, 20000, 5000, 500)
duration = st.sidebar.slider("📅 Duration (Days)", 1, 10, 3)
rating = st.sidebar.slider("⭐ Minimum Rating", 3.0, 5.0, 4.0)

# -----------------------------
# MAP COORDINATES (STATIC CITY MAPS)
# -----------------------------
city_coords = {
    "Goa": [15.2993, 74.1240],
    "Manali": [32.2432, 77.1892],
    "Jaipur": [26.9124, 75.7873],
    "Ladakh": [34.1526, 77.5771],
    "Kerala": [10.8505, 76.2711],
    "Mumbai": [19.0760, 72.8777],
    "Delhi": [28.7041, 77.1025],
    "Agra": [27.1767, 78.0081],
    "Shimla": [31.1048, 77.1734],
    "Pune": [18.5204, 73.8567]
}

# -----------------------------
# SCORING FUNCTION
# -----------------------------
def compute_score(row, budget):
    cost_score = max(0, 1 - (row["Cost"] / budget))
    rating_score = row["Rating"] / 5
    return (0.5 * cost_score) + (0.5 * rating_score)

# -----------------------------
# ITINERARY GENERATOR
# -----------------------------
def generate_itinerary(df, days):
    itinerary = {}
    sample = df.sample(min(days, len(df))).reset_index(drop=True)

    plans = [
        "Local sightseeing & food tour",
        "Adventure activities",
        "Historical exploration",
        "Nature walk & relaxation",
        "City tour & shopping"
    ]

    for i in range(days):
        row = sample.iloc[i % len(sample)]

        itinerary[f"Day {i+1}"] = {
            "Destination": row["Destination"],
            "Category": row["Category"],
            "Cost": row["Cost"],
            "Rating": row["Rating"],
            "Plan": random.choice(plans)
        }

    return itinerary

# -----------------------------
# MAIN BUTTON
# -----------------------------
if st.button("🚀 Generate Travel Plan"):

    try:
        # -----------------------------
        # ML CATEGORY (NO DESTINATION INPUT NOW)
        # -----------------------------
        loc_enc = le_loc.transform([random.choice(le_loc.classes_)])[0]

        input_data = np.array([[0, loc_enc, budget, duration, rating]])
        pred = model.predict(input_data)[0]
        predicted_category = le_cat.inverse_transform([pred])[0]

        st.success(f"🧠 Suggested Category: {predicted_category}")

        # -----------------------------
        # FILTER DATA
        # -----------------------------
        filtered = df[
            (df["Cost"] <= budget) &
            (df["Duration"] <= duration) &
            (df["Rating"] >= rating)
        ]

        final_plan = filtered[filtered["Category"] == predicted_category]

        if final_plan.empty:
            final_plan = filtered

        if final_plan.empty:
            st.warning("No matching trips found.")
        else:

            # -----------------------------
            # SCORING
            # -----------------------------
            final_plan["Score"] = final_plan.apply(lambda x: compute_score(x, budget), axis=1)
            final_plan = final_plan.sort_values(by="Score", ascending=False)

            # -----------------------------
            # LAYOUT
            # -----------------------------
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🧳 Top Travel Plans")
                st.dataframe(final_plan.head(10))

            with col2:
                st.subheader("📊 Cost Distribution")
                fig, ax = plt.subplots()
                ax.hist(final_plan["Cost"], bins=10)
                st.pyplot(fig)

            # -----------------------------
            # ITINERARY
            # -----------------------------
            st.subheader("📅 Day-wise Itinerary")
            itinerary = generate_itinerary(final_plan, duration)

            for day, details in itinerary.items():
                with st.expander(day):
                    st.write(details)

            # -----------------------------
            # MAP SECTION
            # -----------------------------
            st.subheader("🗺 Travel Map")

            m = folium.Map(location=[22.9734, 78.6569], zoom_start=5)

            for _, row in final_plan.head(10).iterrows():
                city = row["Destination"]

                if city in city_coords:
                    folium.Marker(
                        location=city_coords[city],
                        popup=f"{city} | {row['Category']} | ₹{row['Cost']}",
                    ).add_to(m)

            st_folium(m, width=900, height=500)

    except Exception as e:
        st.error(f"Error: {str(e)}")
