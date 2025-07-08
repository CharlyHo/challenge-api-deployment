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
        subtype = st.selectbox("Subtype property", ["HOUSE", "VILLA", "APARTMENT", "STUDIO", "PENTHOUSE", "DUPLEX"])
        property_type = st.selectbox("Type property", ["HOUSE", "APARTMENT", "OTHERS"])
        condition = st.selectbox("Building condition", [6, 5, 4, 3, 2, 1], format_func=lambda x: {
            6: "AS_NEW", 5: "GOOD", 4: "JUST RENOVATED", 3: "TO_BE_DONE_UP", 2: "TO_RENOVATE", 1: "TO_RESTORE"
        }[x])

    with col2:
        zip_code = st.number_input("Postal code", min_value=1000, max_value=9999, step=1)
        land_area = st.number_input("Land surface (m²)", min_value=0, step=1)
        garden_area = st.number_input("Garden surface (m²)", min_value=0, step=1)
        has_terrace = st.checkbox("Terrasse")
        has_parking = st.checkbox("Parking")
        epc_score = st.slider("Estimate epcScore (1=G ➜ 9=A++)", min_value=1, max_value=9, value=5)

    submitted = st.form_submit_button("price prediction")

if submitted:
    input_data = {
        "area": area,
        "property-type": property_type,
        "subtype of property": subtype,
        "rooms-number": rooms,
        "zip-code": zip_code,
        "land-area": land_area,
        "garden": garden_area > 0,
        "garden-area": garden_area,
        "terrace": has_terrace,
        "terrace-area": 10 if has_terrace else 0,
        "building condition": condition,
        "parking": has_parking,
        "epcScore": epc_score
    }

    try:
        predicted_price = predict(input_data)
        st.success(f"💰 Estimate price : **{predicted_price:,.0f} €**")
    except Exception as e:
        st.error(f"Error : {str(e)}")
