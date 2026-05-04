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
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Travel Planner", layout="wide")

st.title("🌍🧭 AI Travel Planner System")
st.write("Smart itinerary planning with AI + filters + map")

# -----------------------------
# CITY COORDINATES
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
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("🎛 Travel Preferences")

budget = st.sidebar.number_input("💰 Budget (₹)", 1000, 20000, 5000, 500)
duration = st.sidebar.slider("📅 Duration (Days)", 1, 10, 3)
rating = st.sidebar.slider("⭐ Minimum Rating", 3.0, 5.0, 4.0)

category_input = st.sidebar.selectbox(
    "🏷 Travel Category",
    list(le_cat.classes_)
)

# -----------------------------
# SCORING FUNCTION
# -----------------------------
def compute_score(row, budget):
    cost_score = max(0, 1 - (row["Cost"] / budget))
    rating_score = row["Rating"] / 5
    return (0.5 * cost_score) + (0.5 * rating_score)

# -----------------------------
# ITINERARY FUNCTION
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
# SESSION STATE INIT (IMPORTANT FIX)
# -----------------------------
if "plan" not in st.session_state:
    st.session_state.plan = None

if "itinerary" not in st.session_state:
    st.session_state.itinerary = None

# -----------------------------
# GENERATE BUTTON
# -----------------------------
if st.button("🚀 Generate Travel Plan"):

    filtered = df[
        (df["Cost"] <= budget) &
        (df["Duration"] <= duration) &
        (df["Rating"] >= rating) &
        (df["Category"] == category_input)
    ]

    if not filtered.empty:

        filtered["Score"] = filtered.apply(lambda x: compute_score(x, budget), axis=1)
        filtered = filtered.sort_values(by="Score", ascending=False)

        # SAVE STATE (FIX FOR DISAPPEARING OUTPUT)
        st.session_state.plan = filtered
        st.session_state.itinerary = generate_itinerary(filtered, duration)

    else:
        st.session_state.plan = None
        st.warning("No matching travel plans found. Try increasing budget or duration.")

# -----------------------------
# DISPLAY RESULTS (PERSISTENT)
# -----------------------------
if st.session_state.plan is not None:

    plan = st.session_state.plan

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧳 Top Travel Plans")
        st.dataframe(plan.head(10))

    with col2:
        st.subheader("📊 Cost Analysis")
        fig, ax = plt.subplots()
        ax.hist(plan["Cost"], bins=10)
        st.pyplot(fig)

    # -----------------------------
    # ITINERARY
    # -----------------------------
    st.subheader("📅 Day-wise Itinerary")

    for day, details in st.session_state.itinerary.items():
        with st.expander(day):
            st.write(details)

    # -----------------------------
    # MAP
    # -----------------------------
    st.subheader("🗺 Travel Map")

    m = folium.Map(location=[22.9734, 78.6569], zoom_start=5)

    for _, row in plan.head(10).iterrows():
        city = row["Destination"]

        if city in city_coords:
            folium.Marker(
                location=city_coords[city],
                popup=f"{city} | {row['Category']} | ₹{row['Cost']}"
            ).add_to(m)

    st_folium(m, width=900, height=500)
