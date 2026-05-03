import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# =========================
# LOAD + FIX DATA
# =========================

@st.cache_data
def load_data():
    df = pd.read_csv("travel_data.csv")
    df.dropna(inplace=True)

    # Rename columns (based on your dataset)
    df = df.rename(columns={
        "city_name": "Destination",
        "hotel_name": "Place",
        "rating_category": "Category",
        "hotel_rating": "Rating",
        "num_reviews": "Reviews"
    })

    # Feature Engineering: Create Cost
    df["Cost"] = (
        df["Rating"] * 200 + 
        np.log1p(df["Reviews"]) * 100
    ).astype(int)

    df["Cost"] = df["Cost"].clip(100, 5000)

    return df


# =========================
# TRAIN MODEL
# =========================

@st.cache_resource
def train_model(df):
    df = df.copy()

    df["Category_enc"] = df["Category"].astype("category").cat.codes

    X = df[["Category_enc", "Cost", "Reviews"]]
    y = df["Rating"]

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    return model, df


# =========================
# RECOMMENDATION
# =========================

def recommend_places(df, model, destination, preferences, budget):

    df = df[df["Destination"].str.contains(destination, case=False)]
    df = df[df["Category"].isin(preferences)]
    df = df[df["Cost"] <= budget]

    if df.empty:
        return df

    df["Category_enc"] = df["Category"].astype("category").cat.codes

    df["Predicted_Rating"] = model.predict(
        df[["Category_enc", "Cost", "Reviews"]]
    )

    df = df.sort_values(by="Predicted_Rating", ascending=False)

    return df.head(10)


# =========================
# STREAMLIT UI
# =========================

st.set_page_config(layout="wide")
st.title("🎒 AI Travel Planner (Dataset + ML)")

df = load_data()
model, df = train_model(df)

# Sidebar
st.sidebar.header("Trip Details")

destination = st.sidebar.text_input("Destination", "Delhi")

budget = st.sidebar.slider("Budget", 100, 5000, 1000)

preferences = st.sidebar.multiselect(
    "Preferences",
    df["Category"].unique(),
    default=list(df["Category"].unique())[:2]
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

        st.dataframe(
            results[["Place", "Destination", "Category", "Rating", "Cost"]]
        )

        # Map not available
        st.info("Map not available (dataset has no coordinates)")
