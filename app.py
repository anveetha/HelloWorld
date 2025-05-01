import streamlit as st
import pandas as pd
import pydeck as pdk
import requests
from streamlit_javascript import st_javascript
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
import numpy as np
import pydeck as pdk
from geopy.distance import geodesic
from streamlit_javascript import st_javascript

# --- Ticketmaster API Key ---
API_KEY = "sPTGoDBnMjr6gfs9TqQYd4FomA5oDBYC"
BASE_URL = "https://app.ticketmaster.com/discovery/v2/events.json"

# --- Reverse Geocoding Function ---
def reverse_geocode(lat, lon):
    url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"
    headers = {"User-Agent": "NearbyEventsApp/1.0 (your_email@example.com)"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        address = data.get("address", {})
        city = address.get("city") or address.get("town") or address.get("village")
        state = address.get("state")
        country = address.get("country")
        full_location = f"{city}, {state}, {country}" if city and state and country else None
        return full_location
    else:
        return None

# --- Page Setup ---
if "page" not in st.session_state:
    st.session_state.page = "home"

query_params = st.query_params
page = query_params.get("page", st.session_state.page)
st.session_state.page = page
# this is my comment
# --- Location Fetching Function ---
def fetch_location():
    if "location" not in st.session_state:
        coords = st_javascript("""await new Promise((resolve, reject) => {
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    resolve({
                        coords: {
                            latitude: position.coords.latitude,
                            longitude: position.coords.longitude
                        }
                    });
                },
                (error) => {
                    resolve({error: error.message});
                }
            );
        });""")

        if coords and "coords" in coords:
            st.session_state.location = coords["coords"]

            lat = coords["coords"]["latitude"]
            lon = coords["coords"]["longitude"]

            # Reverse geocode city
            full_location = reverse_geocode(lat, lon)

            if full_location:
                st.session_state.location_details = full_location
                st.success(f"📍 Location saved: {full_location}")
            else:
                st.success(f"📍 Location saved: {lat:.4f}, {lon:.4f} (Location unknown)")
        else:
            st.warning("⚠️ Waiting for geolocation or permission denied.")

# ------------------------------ HOME PAGE -----------------------------------------------------------
if st.session_state.page == "home":
    st.title("🌎 Hello World")
    
    # Fetch location on Home page
    fetch_location()

#------------------------------------ EVENTS PAGE ------------------------------------------------------
elif st.session_state.page == "events":
    st.title("Events Near You:")

    fetch_location()  # Ensure location is fetched on this page

    #if "location" not in st.session_state:
    #    st.error("⚠️ Location not set. Please go to the Home page first to allow location access.")
    #    st.stop()

    lat = st.session_state.location['latitude']
    lon = st.session_state.location['longitude']
    city = st.session_state.get("city", None)


    # Map display: ----------------------------------------------
    user_location = pd.DataFrame({'lat': [lat], 'lon': [lon]})

    layer = pdk.Layer(
        'ScatterplotLayer',
        data=user_location,
        get_position='[lon, lat]',
        get_color='[255, 0, 0, 160]',
        get_radius=30
    )

    view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=15, pitch=0)
    st.pydeck_chart(
    pdk.Deck(
        map_style="mapbox://styles/mapbox/streets-v11",
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "You are here"}
    ),
    use_container_width=True
)

    #
   
    
    # Event Search
    st.subheader("🔍 Search Nearby Events")
    keyword = st.text_input("What are you looking for? (e.g. concerts, sports, comedy)")
    radius = st.slider("Radius (miles)", min_value=5, max_value=100, value=25)

    if st.button("Search Events"):
        if keyword:
            params = {
                "apikey": API_KEY,
                "keyword": keyword,
                "latlong": f"{lat},{lon}",
                "radius": radius,
                "unit": "miles"
            }

            response = requests.get(BASE_URL, params=params)
            data = response.json()
            events = data.get("_embedded", {}).get("events", [])

            if events:
                st.success(f"Found {len(events)} event(s) near you!")
                for event in events:
                    name = event.get("name")
                    venue = event["_embedded"]["venues"][0]["name"]
                    date = event["dates"]["start"].get("localDate")
                    st.subheader(name)
                    st.write("📍", venue)
                    st.write("📅", date)
                    if event.get("url"):
                        st.markdown(f"[More Info]({event.get('url')})")
                    st.markdown("---")
            else:
                st.info("No events found nearby for that keyword.")
        else:
            st.warning("Please enter a keyword to search.")

#------------------------------ AI ITINERARY PAGE ------------------------------------------------------
elif st.session_state.page == "itinerary":
    fetch_location()  # Ensure location is fetched on this page

    # Mocked user search history
    mock_search_history = [
        "Live Jazz Concert",
        "Green Leaf Vegan Cafe",
        "City Museum"
    ]

    # Available options
    events = [
        "Art Exhibition at City Gallery",
        "Live Jazz Concert",
        "Food Festival in Downtown",
        "Tech Meetup at Innovation Hub",
        "Yoga in the Park"
    ]

    restaurants = [
        "Joe's Italian Bistro",
        "Green Leaf Vegan Cafe",
        "Sushi Samba Lounge",
        "Downtown Steakhouse",
        "Spicy Indian Kitchen"
    ]

    locations = [
        "Central Park",
        "City Museum",
        "Historic Downtown",
        "Beachfront Promenade",
        "Mountain Hiking Trail"
    ]

    all_items = events + restaurants + locations

    st.title("AI Itinerary Planner")

    st.subheader("Mocked User Search History")
    st.write(mock_search_history)

    # Recommendation system
    def get_recommendations(history, all_items):
        combined = list(set(history + all_items))

        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(combined)

        history_indices = [combined.index(h) for h in history]
        similarity_matrix = cosine_similarity(tfidf_matrix[history_indices], tfidf_matrix)

        scores = similarity_matrix.mean(axis=0)
        ranked_items = [(combined[i], scores[i]) for i in range(len(combined)) if combined[i] not in history]
        ranked_items = sorted(ranked_items, key=lambda x: x[1], reverse=True)

        return [item[0] for item in ranked_items[:10]]

    # Itinerary generation
    def generate_itinerary(recommendations, start_time_str="09:00"):
        itinerary = []
        start_time = datetime.strptime(start_time_str, "%H:%M")
        time_slots = [("Activity", 90), ("Break", 30)]  # alternate activities and breaks

        i = 0
        while i < len(recommendations):
            for label, duration in time_slots:
                if i >= len(recommendations):
                    break
                if label == "Activity":
                    itinerary.append({
                        "time": start_time.strftime("%I:%M %p"),
                        "activity": recommendations[i]
                    })
                    start_time += timedelta(minutes=duration)
                    i += 1
                else:
                    start_time += timedelta(minutes=duration)
        return itinerary

    # Generate and display itinerary
    st.subheader("AI-Powered Daily Itinerary")
    recommendations = get_recommendations(mock_search_history, all_items)
    itinerary = generate_itinerary(recommendations)

    for item in itinerary:
        st.markdown(f"**{item['time']}** - {item['activity']}")

# -------------------------- RESTAURANT PAGE --------------------
elif st.session_state.page == "restaurant":
    st.title("🍽️ Restaurant Recommender")

    fetch_location()  # Ensure location is fetched and stored in session state

    if "location" not in st.session_state:
        st.warning("📍 Location not available. Please allow location access from the Home page.")
        st.stop()

    lat = st.session_state.location['latitude']
    lon = st.session_state.location['longitude']
    full_location = st.session_state.get("location_details", "Unknown Location")

    st.markdown(f"📍 **You are in:** {full_location}")

    # --- Load and Prepare Data ---
    DATA_PATH = r"C:\Users\trinh\Downloads\grubhub.csv\grubhub.csv"
    data = pd.read_csv(DATA_PATH)

    data.drop(['delivery_fee_raw', 'delivery_fee', 'delivery_time_raw', 'delivery_time', 'service_fee'], 
            axis=1, inplace=True, errors='ignore')
    price_categories = ['low', 'medium', 'high']
    if 'prices' not in data.columns:
        data['prices'] = np.random.choice(price_categories, size=len(data))
    data['cuisines'] = data['cuisines'].fillna('')  # Fill NaN cuisines

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['cuisines'])

    def location_similarity(user_location, restaurant_location):
        return 1 / (1 + geodesic(user_location, restaurant_location).km)

    def recommend_restaurants(user_cuisine, user_lat, user_lon, user_price=None, top_n=5):
        tfidf_user = tfidf.transform([user_cuisine])
        cuisine_sim_scores = cosine_similarity(tfidf_user, tfidf_matrix).flatten()

        user_location = (user_lat, user_lon)
        location_sim_scores = np.array([
            location_similarity(user_location, (lat, lon)) 
            for lat, lon in zip(data['latitude'], data['longitude'])
        ])

        combined_scores = 0.7 * cuisine_sim_scores + 0.3 * location_sim_scores

        filtered_data = data
        filtered_scores = combined_scores
        if user_price:
            price_mask = (data['prices'] == user_price)
            filtered_data = data[price_mask]
            filtered_scores = combined_scores[price_mask.values]

        top_indices = filtered_scores.argsort()[-top_n:][::-1]
        recommended = filtered_data.iloc[top_indices][['loc_name', 'latitude', 'longitude', 'cuisines', 'review_rating', 'prices']]
        recommended['distance_km'] = [
            geodesic(user_location, (lat, lon)).km for lat, lon in zip(recommended['latitude'], recommended['longitude'])
        ]
        return recommended

    user_location_df = pd.DataFrame({'lat': [lat], 'lon': [lon]})
    layer = pdk.Layer(
        'ScatterplotLayer',
        data=user_location_df,
        get_position='[lon, lat]',
        get_color='[255, 0, 0, 160]',
        get_radius=30
    )
    view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=15, pitch=0)
    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/streets-v11",
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"text": "You are here"}
        ),
        use_container_width=True
    )

    with st.form("recommendation_form"):
        user_cuisine = st.text_input("What cuisines are you craving?", "chinese italian")
        user_price = st.selectbox("Price range", ["Any", "low", "medium", "high"])
        top_n = st.slider("Number of recommendations", 1, 10, 5)
        submitted = st.form_submit_button("Find Restaurants")

    recommendations = None

    if submitted:
        recommendations = recommend_restaurants(
            user_cuisine, lat, lon,
            user_price if user_price != "Any" else None,
            top_n
        )

    if submitted and user_cuisine:
        st.subheader("You're craving:")
        cuisine_list = [c.strip().capitalize() for c in user_cuisine.split()]
        cols = st.columns(len(cuisine_list))
        for idx, cuisine in enumerate(cuisine_list):
            with cols[idx]:
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 8px 12px; border-radius: 20px; text-align: center; font-weight: 600; color: #000000">
                    {cuisine}
                </div>
                """, unsafe_allow_html=True)

    if recommendations is not None and not recommendations.empty:
        st.subheader("Top Recommendations")
        for idx, row in recommendations.iterrows():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.map(pd.DataFrame({'lat': [row['latitude']], 'lon': [row['longitude']]}), zoom=15, use_container_width=True)
            with col2:
                st.markdown(f"""
                <div style="background-color: #ffffff; border-radius: 12px; padding: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);  color: #000000">
                    <h3>🍴 {row['loc_name']}</h3>
                    <p><strong>Cuisines:</strong> {row['cuisines']}</p>
                    <p><strong>Rating:</strong> ⭐ {row['review_rating']}</p>
                    <p><strong>Price:</strong> 💵 {row['prices']}</p>
                    <p><strong>Distance:</strong> 📍 {row['distance_km']:.2f} km</p>
                </div>
                """, unsafe_allow_html=True)

        st.subheader("📋 Summary Table")
        st.dataframe(
            recommendations[['loc_name', 'cuisines', 'review_rating', 'prices', 'distance_km']].rename(
                columns={
                    "loc_name": "Restaurant",
                    "cuisines": "Cuisines",
                    "review_rating": "Rating",
                    "prices": "Price",
                    "distance_km": "Distance (km)"
                }
            )
        )
    elif submitted:
        st.warning("No recommendations found for your criteria.")


# ---------------------------- Navigation ----------------------------
else:
    # All non-home pages just show the page name
    st.markdown(f"## 🧭 {st.session_state.page.capitalize()} Page")

# --- Custom Navigation Bar ---
st.markdown("""
    <style>
    .bottom-nav {
        position: fixed;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        max-width: 700px;
        width: 90%;
        background-color: #fff;
        border-top: 2px solid #ccc;
        display: flex;
        justify-content: space-around;
        align-items: center;
        padding: 6px 0;
        z-index: 9999;
        box-shadow: 0 -1px 5px rgba(0,0,0,0.1);
        border-radius: 12px 12px 0 0;
    }
    .bottom-nav a {
        text-decoration: none;
        color: black;
        font-weight: 500;
        text-align: center;
        flex-grow: 1;
        font-size: 14px;
        padding: 6px 0;
        border-right: 1px solid #eee;
    }
    .bottom-nav a:last-child {
        border-right: none;
    }
    .bottom-nav a:hover {
        color: #00aced;
    }
    </style>
    <div class="bottom-nav">
        <a href="?page=home">Home</a>
        <a href="?page=events">Events</a>
        <a href="?page=restaurant">Restaurant</a>
        <a href="?page=itinerary">AI Itinerary</a>
        <a href="?page=settings">Settings</a>
    </div>
""", unsafe_allow_html=True)
