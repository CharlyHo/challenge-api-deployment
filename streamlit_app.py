import streamlit as st
from predict.prediction import predict
import os


st.set_page_config(page_title="🏠 Real Estate Valuation", layout="centered")




st.title("📊 Real estate price predictor")
st.markdown("Fill in the information below to estimate the **sale price of a property**.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input("Habitable surface (m²)", min_value=10, step=1)
        rooms = st.number_input("Bedroom count", min_value=0, step=1)
        subtype = st.selectbox(
            "Subtype property",
            {
                "APARTMENT": 1,
                "HOUSE": 2,
                "FLAT STUDIO": 3,
                "DUPLEX": 4,
                "PENTHOUSE": 5,
                "GROUNDFLOOR": 6,
                "APARTMENT BLOCK": 7,
                "MANSION": 8,
                "EXCEPTIONAL PROPERTY": 9,
                "MIXED USE BUILDING": 10,
                "TRIPLEX": 11,
                "LOFT": 12,
                "VILLA": 13,
                "TOWN HOUSE": 14,
                "CHALET": 15,
                "MANOR HOUSE": 16,
                "SERVICE FLAT": 17,
                "KOT": 18,
                "FARMHOUSE": 19,
                "BUNGALOW": 20,
                "COUNTRY COTTAGE": 21,
                "OTHER PROPERTY": 22,
                "CASTLE": 23,
                "PAVILION": 24
            }
        )
        property_type = st.selectbox("Type property", {"HOUSE": 2, "APARTMENT": 1})
        condition = st.selectbox("Building condition", [6, 5, 4, 3, 2, 1], format_func=lambda x: {
            6: "TO RESTORE", 5: "JUST RENOVATED", 4: "TO BE DONE UP", 3: "TO RENOVATE", 2: "AS NEW", 1: "GOOD"
        }[x])
        heating_type = st.selectbox("Heating type", {
            "GAS": 1, "FUELOIL": 2, "ELECTRIC": 3, "PELLET": 4, "WOOD": 5, "SOLAR": 6, "CARBON": 7
        })
        kitchen_types = st.selectbox("Kitchen type", {
            "NOT_INSTALLED": 0, "SEMI_EQUIPPED": 1, "INSTALLED": 2, "HYPER_EQUIPPED": 3
        })
        full_address = st.text_input("Full address (optional)")

    with col2:
        zip_code = st.number_input("Postal code", min_value=1000, max_value=9999, step=1)
        land_area = st.number_input("Land surface (m²)", min_value=0, step=1)
        garden_area = st.number_input("Garden surface (m²)", min_value=0, step=1)
        has_garden = garden_area > 0
        facades_number = st.number_input("Number of facades", min_value=1, max_value=4, step=1)
        flood_zone_type = st.selectbox("Flood zone type", {
            'Unknown': -1,
            'NON FLOOD ZONE': 1,
            'POSSIBLE FLOOD ZONE': 2,
            'RECOGNIZED FLOOD ZONE': 3,
            'RECOGNIZED N CIRCUMSCRIBED FLOOD ZONE': 4,
            'CIRCUMSCRIBED WATERSIDE ZONE': 5,
            'CIRCUMSCRIBED FLOOD ZONE': 6,
            'POSSIBLE N CIRCUMSCRIBED FLOOD ZONE': 7,
            'POSSIBLE N CIRCUMSCRIBED WATERSIDE ZONE': 8,
            'RECOGNIZED N CIRCUMSCRIBED WATERSIDE FLOOD ZONE': 9
        })
        has_terrace = st.checkbox("Terrace")
        terrace_area = st.number_input("Terrace surface (m²)", min_value=0, step=1) if has_terrace else 0
        has_parking = st.checkbox("Parking")
        swimming_pool = st.checkbox("Swimming pool")
        open_fire = st.checkbox("Open fire")
        epc_score = st.slider("Estimate EPC score (1=G ➜ 9=A++)", min_value=1, max_value=9, value=5)
      
        

    submitted = st.form_submit_button("Price prediction")

if submitted:
    input_data = {
        "area": area,
        "property-type": property_type,
        "subtype of property": subtype,
        "rooms-number": rooms,
        "zip-code": zip_code,
        "land-area": land_area,
        "garden": has_garden,
        "garden-area": garden_area,
        "terrace": has_terrace,
        "terrace-area": terrace_area,
        "building condition": condition,
        "parking": has_parking,
        "epcScore": epc_score,
        "heating type": heating_type,
        "flood zone type": flood_zone_type,
        "kitchen types": kitchen_types,
        "swimming-pool": swimming_pool,
        "open-fire": open_fire,
        "facades-number": facades_number,
        "full-address": full_address
    }

    try:
        predicted_price = predict(input_data)
        st.success(f"Estimated price: **{predicted_price:,.0f} €**")
    except Exception as e:

        st.error(f"Prediction error: {str(e)}")

