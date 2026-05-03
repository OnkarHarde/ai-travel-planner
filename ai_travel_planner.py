import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor

# =========================
# LOAD + CLEAN DATA (AUTO FIX)
# =========================

@st.cache_data
def load_data():
    df = pd.read_csv("travel_data.csv")
    df.dropna(inplace=True)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Auto-map columns
    column_map = {}
    for col in df.columns:
        c = col.lower()
        if "category" in c or "type" in c:
            column_map[col] = "Category"
        elif "destination" in c or "place" in c or "name" in c:
            column_map[col] = "Destination"
        elif "rating" in c:
            column_map[col] = "Rating"
        elif "cost" in c or "price" in c:
            column_map[col] = "Cost"
        elif "lat" in c:
            column_map[col] = "Latitude"
        elif "lon" in c or "lng" in c:
            column_map[col] = "Longitude"

    df = df.rename(columns=column_map)

    return df


# =========================
# TRAIN MODEL
# =========================

@st.cache_resource
def train_model(df):

    required_cols = ["Category", "Cost", "Rating"]

    for col in required_cols:
        if col not in df.columns:
            st.error(f"❌ Missing column: {col}")
            st.write("Your dataset columns:", df.columns)
            st.stop()

    df = df.copy()

    df["Category_enc"] = df["Category"].astype("category").cat.codes

    X = df[["Category_enc", "Cost"]]
    y = df["Rating"]

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    return model, df


# =========================
# RECOMMENDATION ENGINE
# =========================

def recommend_places(df, model, destination, preferences, budget):

    # Filter
    if "Destination" in df.columns:
        df = df[df["Destination"].astype(str).str.contains(destination, case=False)]

    if "Category" in df.columns:
        df = df[df["Category"].isin(preferences)]

    if "Cost" in df.columns:
        df = df[df["Cost"] <= budget]

    if df.empty:
        return df

    # Encode again
    df["Category_enc"] = df["Category"].astype("category").cat.codes

    # Predict
    df["Predicted_Rating"] = model.predict(df[["Category_enc", "Cost"]])

    # Sort best first
    df = df.sort_values(by="Predicted_Rating", ascending=False)

    return df.head(10)


# =========================
# MAP
# =========================

def create_map(df):

    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        st.warning("No location data available for map")
        return None

    m = folium.Map(
        location=[df.iloc[0]["Latitude"], df.iloc[0]["Longitude"]],
        zoom_start=12
    )

    for _, row in df.iterrows():
        folium.Marker(
            location=[row["Latitude"], row["Longitude"]],
            popup=f"{row.get('Destination','Place')} ({row.get('Rating',0)})"
        ).add_to(m)

    return m


# =========================
# UI
# =========================

st.set_page_config(layout="wide")
st.title("🎒 AI Travel Planner (Dataset + ML - Stable Version)")

df = load_data()

# Debug (remove later if needed)
st.write("Detected Columns:", df.columns)

model, df = train_model(df)

# Sidebar
st.sidebar.header("Trip Details")

destination = st.sidebar.text_input("Destination", "Delhi")

budget = st.sidebar.slider("Budget", 100, 5000, 500)

if "Category" in df.columns:
    categories = list(df["Category"].unique())
else:
    categories = []

preferences = st.sidebar.multiselect(
    "Preferences",
    categories,
    default=categories[:2] if len(categories) >= 2 else categories
)

generate = st.sidebar.button("Generate")

# =========================
# OUTPUT
# =========================

if generate:

    results = recommend_places(df, model, destination, preferences, budget)

    if results.empty:
        st.warning("No matching places found")
    else:
        st.subheader("Top Recommendations")

        cols_to_show = [col for col in ["Destination", "Category", "Rating", "Cost"] if col in results.columns]
        st.dataframe(results[cols_to_show])

        st.subheader("Map View")
        m = create_map(results)

        if m:
            st_folium(m, width=700, height=500)
