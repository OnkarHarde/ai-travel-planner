import streamlit as st
from datetime import date
import pandas as pd
import folium
from streamlit_folium import st_folium
import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# =========================
# ML MODEL (Random Forest)
# =========================

@st.cache_resource
def train_model():
    activity_map = {"Museums":0, "Nightlife":1, "Adventure":2, "Food":3, "Parks":4, "Shopping":5}
    transport_map = {"Walk":0, "Public Transport":1, "Ride-Share":2}
    
    data = []
    
    # Simulated dataset
    for _ in range(500):
        pref = random.choice(list(activity_map.values()))
        budget = random.randint(100, 5000)
        transport = random.choice(list(transport_map.values()))
        
        rating = 3 + (budget/5000)*2 + random.uniform(-0.5, 0.5)
        data.append([pref, budget, transport, rating])
    
    df = pd.DataFrame(data, columns=["pref", "budget", "transport", "rating"])
    
    X = df[["pref", "budget", "transport"]]
    y = df["rating"]
    
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    
    return model, activity_map, transport_map


# =========================
# Itinerary Generator (ML-powered)
# =========================

def generate_realistic_itinerary(destination, start_date, end_date, budget, preferences, transport):
    
    model, activity_map, transport_map = train_model()
    
    activity_costs = {
        "Museums": 10,
        "Nightlife": 20,
        "Adventure": 30,
        "Food": 15,
        "Parks": 5,
        "Shopping": 25
    }
    
    base_lat, base_lon = 28.6139, 77.2090
    
    itinerary = []
    days = (end_date - start_date).days + 1
    
    for day_idx in range(days):
        day_plan = {"day": f"Day {day_idx+1}", "activities": []}
        time_slots = ["09:00-12:00", "12:30-15:30", "16:00-19:00", "19:30-22:00"]
        
        for slot_idx, pref in enumerate(preferences):
            if slot_idx >= len(time_slots):
                break
            
            pref_encoded = activity_map[pref]
            transport_encoded = transport_map[transport]
            
            predicted_rating = model.predict([[pref_encoded, budget, transport_encoded]])[0]
            
            act = {
                "name": f"{pref} Spot {day_idx+1}",
                "time": time_slots[slot_idx],
                "cost": activity_costs.get(pref, 10) + random.randint(0,5),
                "location": (
                    base_lat + random.uniform(0,0.02), 
                    base_lon + random.uniform(0,0.02)
                ),
                "rating": round(float(predicted_rating), 1)
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

st.set_page_config(page_title="AI Travel Planner", layout="wide")

st.title("🎒 AI Travel Planner (ML Powered)")
st.markdown("Now powered with **Random Forest AI recommendations** 🚀")

# Sidebar
st.sidebar.header("Trip Details")

destination = st.sidebar.text_input("Destination", value="Delhi")

start_date = st.sidebar.date_input("Start date", date.today())
end_date = st.sidebar.date_input("End date", date.today())

budget = st.sidebar.slider("Budget ($)", 100, 5000, 500)

preferences = st.sidebar.multiselect(
    "Preferences",
    ["Museums", "Nightlife", "Adventure", "Food", "Parks", "Shopping"],
    default=["Museums", "Food"]
)

transport = st.sidebar.selectbox(
    "Transport",
    ["Walk", "Public Transport", "Ride-Share"]
)

generate_btn = st.sidebar.button("Generate Itinerary")

# Session state
if "generated" not in st.session_state:
    st.session_state.generated = False
if "itinerary" not in st.session_state:
    st.session_state.itinerary = []

# Tabs
tabs = st.tabs(["Overview", "Daily Plan", "Map", "Budget"])

# Generate
if generate_btn:
    with st.spinner("Generating AI-powered itinerary..."):
        st.session_state.itinerary = generate_realistic_itinerary(
            destination, start_date, end_date, budget, preferences, transport
        )
        st.session_state.generated = True

# Display
if st.session_state.generated:
    itinerary = st.session_state.itinerary
    total_cost = calculate_budget(itinerary)

    with tabs[0]:
        st.header("Overview")
        st.write(destination, start_date, end_date)
        st.write(f"Budget: ${budget}")
        st.write(f"Estimated Cost: ${total_cost}")

    with tabs[1]:
        display_itinerary(itinerary)

    with tabs[2]:
        st_folium(create_map(itinerary), width=700, height=500)

    with tabs[3]:
        df = pd.DataFrame([
            {"Day": day["day"], "Activity": act["name"], "Cost": act["cost"]}
            for day in itinerary for act in day["activities"]
        ])
        st.dataframe(df)
        st.write(f"Total Cost: ${total_cost}")
