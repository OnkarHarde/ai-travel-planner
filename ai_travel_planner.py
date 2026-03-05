
import streamlit as st
from datetime import date
import pandas as pd
import folium
from streamlit_folium import st_folium

# =========================
# Helper Functions
# =========================

def generate_itinerary(destination, start_date, end_date, budget, preferences, transport):
    """
    Placeholder AI function to simulate itinerary generation.
    Replace with OpenAI API call or local LLM integration.
    """
    itinerary = []
    days = (end_date - start_date).days + 1
    for i in range(days):
        day_plan = {
            "day": f"Day {i+1}",
            "activities": [
                {
                    "name": f"{pref} Spot {i+1}",
                    "time": f"{9+i}:00 AM - {12+i}:00 PM",
                    "cost": round(budget/days*0.3, 2),
                    "location": (28.6139 + 0.01*i, 77.2090 + 0.01*i),  # Example lat/lon
                    "rating": round(3 + i*0.2, 1)
                } for pref in preferences
            ]
        }
        itinerary.append(day_plan)
    return itinerary

def display_itinerary(itinerary):
    for day in itinerary:
        with st.expander(day["day"]):
            for act in day["activities"]:
                st.markdown(f"**{act['name']}**")
                st.write(f"Time: {act['time']}")
                st.write(f"Cost: ${act['cost']}")
                st.write(f"Rating: {act['rating']}/5")
                st.markdown(f"[View on Map](https://www.google.com/maps?q={act['location'][0]},{act['location'][1]})")
                st.markdown("---")

def create_map(itinerary):
    # Center map on first activity
    start_coords = itinerary[0]["activities"][0]["location"]
    m = folium.Map(location=start_coords, zoom_start=12)
    for day in itinerary:
        for act in day["activities"]:
            folium.Marker(
                location=act["location"],
                popup=f"{act['name']} (${act['cost']})",
                tooltip=act["name"]
            ).add_to(m)
    return m

def calculate_budget(itinerary):
    total_cost = sum(act["cost"] for day in itinerary for act in day["activities"])
    return total_cost

# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="AI Travel Planner for Students", layout="wide")

st.title("🎒 AI Travel Planner for Students")
st.markdown("Plan budget-friendly trips with AI-generated personalized itineraries!")

# -------------------------
# Sidebar Inputs
# -------------------------
st.sidebar.header("Trip Details")

destination = st.sidebar.text_input("Destination city or region", value="Delhi")

start_date = st.sidebar.date_input("Start date", date.today())
end_date = st.sidebar.date_input("End date", date.today())

budget = st.sidebar.slider("Total Budget ($)", min_value=100, max_value=5000, value=500, step=50)

preferences = st.sidebar.multiselect(
    "Preferences",
    options=["Museums", "Nightlife", "Adventure", "Food", "Parks", "Shopping"],
    default=["Museums", "Food"]
)

transport = st.sidebar.selectbox(
    "Transportation Preference",
    options=["Walk", "Public Transport", "Ride-Share"]
)

generate_btn = st.sidebar.button("Generate Itinerary")

# -------------------------
# Main Tabs
# -------------------------
tabs = st.tabs(["Overview", "Daily Plan", "Map View", "Budget Summary"])

if generate_btn:
    itinerary = generate_itinerary(destination, start_date, end_date, budget, preferences, transport)
    total_cost = calculate_budget(itinerary)

    # Tab 1: Overview
    with tabs[0]:
        st.header("Trip Overview")
        st.write(f"**Destination:** {destination}")
        st.write(f"**Dates:** {start_date} to {end_date}")
        st.write(f"**Budget:** ${budget}")
        st.write(f"**Estimated Total Cost:** ${total_cost}")
        st.write(f"**Preferences:** {', '.join(preferences)}")
        st.write(f"**Transportation:** {transport}")

    # Tab 2: Daily Plan
    with tabs[1]:
        st.header("Daily Itinerary")
        display_itinerary(itinerary)

    # Tab 3: Map View
    with tabs[2]:
        st.header("Map of Activities")
        m = create_map(itinerary)
        st_folium(m, width=700, height=500)

    # Tab 4: Budget Summary
    with tabs[3]:
        st.header("Budget Summary")
        df = pd.DataFrame([
            {"Day": day["day"], "Activity": act["name"], "Cost": act["cost"]}
            for day in itinerary for act in day["activities"]
        ])
        st.dataframe(df)
        st.write(f"**Total Estimated Cost:** ${total_cost}")

    # Export options
    st.download_button(
        "📥 Download Itinerary CSV",
        df.to_csv(index=False).encode('utf-8'),
        file_name=f"{destination}_itinerary.csv",
        mime="text/csv"
    )
