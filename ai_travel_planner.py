import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor

# =========================
# LOAD DATA
# =========================

@st.cache_data
def load_data():
    df = pd.read_csv("travel_data.csv")
    df.dropna(inplace=True)
    return df

# =========================
# TRAIN MODEL
# =========================

@st.cache_resource
def train_model(df):
    df = df.copy()
    
    # Encode category
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
    
    df = df[df["Destination"].str.contains(destination, case=False)]
    df = df[df["Category"].isin(preferences)]
    df = df[df["Cost"] <= budget]
    
    if df.empty:
        return df
    
    df["Category_enc"] = df["Category"].astype("category").cat.codes
    
    df["Predicted_Rating"] = model.predict(df[["Category_enc", "Cost"]])
    
    df = df.sort_values(by="Predicted_Rating", ascending=False)
    
    return df.head(10)

# =========================
# MAP
# =========================

def create_map(df):
    m = folium.Map(location=[df.iloc[0]["Latitude"], df.iloc[0]["Longitude"]], zoom_start=12)
    
    for _, row in df.iterrows():
        folium.Marker(
            location=[row["Latitude"], row["Longitude"]],
            popup=f"{row['Destination']} ({row['Rating']})"
        ).add_to(m)
    
    return m

# =========================
# UI
# =========================

st.set_page_config(layout="wide")
st.title("🎒 AI Travel Planner (Dataset + ML)")

df = load_data()
model, df = train_model(df)

# Sidebar
st.sidebar.header("Trip Details")

destination = st.sidebar.text_input("Destination", "Delhi")

budget = st.sidebar.slider("Budget", 100, 5000, 500)

preferences = st.sidebar.multiselect(
    "Preferences",
    df["Category"].unique(),
    default=list(df["Category"].unique())[:2]
)

generate = st.sidebar.button("Generate")

# Output
if generate:
    results = recommend_places(df, model, destination, preferences, budget)
    
    if results.empty:
        st.warning("No matching places found")
    else:
        st.subheader("Top Recommendations")
        st.dataframe(results[["Destination", "Category", "Rating", "Cost"]])
        
        st.subheader("Map View")
        st_folium(create_map(results), width=700, height=500)
