# ai_travel_planner.py
import streamlit as st
from datetime import date
import pandas as pd
import folium
from streamlit_folium import st_folium
import random

# =========================
# Realistic Itinerary Generator
# =========================

def generate_realistic_itinerary(destination, start_date, end_date, budget, preferences, transport):
    """
    Simulates a more realistic itinerary with:
    - Morning/Afternoon/Evening activities
    - Approximate costs per activity type
    - Nearby locations clustered
    """
    activity_costs = {
        "Museums": 10,
        "Nightlife": 20,
        "Adventure": 30,
        "Food": 15,
        "Parks": 5,
        "Shopping": 25
    }
    
    # Example coordinates (replace with city-specific later)
    base_lat, base_lon = 28.6139, 77.2090  # Delhi example
    
    itinerary = []
    days = (end_date - start_date).days + 1
    
    for day_idx in range(days):
        day_plan = {"day": f"Day {day_idx+1}", "activities": []}
        time_slots = ["09:00-12:00", "12:30-15:30", "16:00-19:00", "19:30-22:00"]
        
        for slot_idx, pref in enumerate(preferences):
            if slot_idx >= len(time_slots):
                break
            act = {
                "name": f"{pref} Spot {day_idx+1}",
                "time": time_slots[slot_idx],
                "cost": activity_costs.get(pref, 10) + random.randint(0,5),
                "location": (
                    base_lat + random.uniform(0,0.02), 
                    base_lon + random.uniform(0,0.02)
                ),
                "rating": round(random.uniform(3.5, 5.0), 1)
            }
            day_plan["activities"].append(act)
        itinerary.append(day_plan)
    return itinerary

# =========================
# Helper Functions
# =========================

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
    return sum(act["cost"] for day in itinerary for act in day["activities"])

# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="AI Travel Planner for Students", layout="wide")

st.title("🎒 AI Travel Planner for Students")
st.markdown("Plan realistic, budget-friendly trips with AI-simulated itineraries!")

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
# Session State Initialization
# -------------------------
if "generated" not in st.session_state:
    st.session_state.generated = False
if "itinerary" not in st.session_state:
    st.session_state.itinerary = []

# -------------------------
# Main Tabs
# -------------------------
tabs = st.tabs(["Overview", "Daily Plan", "Map View", "Budget Summary"])

# Generate itinerary when button is clicked
if generate_btn:
    with st.spinner("Generating your realistic itinerary..."):
        st.session_state.itinerary = generate_realistic_itinerary(destination, start_date, end_date, budget, preferences, transport)
        st.session_state.generated = True

# Display itinerary if generated
if st.session_state.generated:
    itinerary = st.session_state.itinerary
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

        # Export CSV
        st.download_button(
            "📥 Download Itinerary CSV",
            df.to_csv(index=False).encode('utf-8'),
            file_name=f"{destination}_itinerary.csv",
            mime="text/csv"
        )
