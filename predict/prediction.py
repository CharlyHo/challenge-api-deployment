import joblib
import pandas as pd
from typing import Dict, Any

# Load model
model = joblib.load("model/best_model.pkl")

# Define encodings
province_map = lambda zip_code: int(str(zip_code)[:2]) if zip_code else 10  # default province

REQUIRED_FIELDS = [
    "area", "property-type", "subtype of property", "rooms-number", "zip-code",
    "land-area", "garden", "garden-area", "terrace", "terrace-area",
    "parking", "epcScore", "heating type", "flood zone type",
    "kitchen types"
]

# Mappings for encoding
type_map = {"APARTMENT": 1, "HOUSE": 2}
subtype_map = {
    "APARTMENT": 1, "HOUSE": 2, "STUDIO": 3, "DUPLEX": 4, "PENTHOUSE": 5, "VILLA": 6
}

building_condition_map = {
    "GOOD": 1, "AS NEW": 2, "TO RENOVATE": 3, "TO BE DONE UP": 4, 
    "JUST RENOVATED": 5, "TO RESTORE": 6
}

heating_map = {"GAS": 1, "ELECTRIC": 2, "WOOD": 3, "FUELOIL": 4}
flood_zone_map = {
    "NON FLOOD ZONE": 1, "POSSIBLE FLOOD ZONE": 2, "RECOGNIZED FLOOD ZONE": 3
}
kitchen_map = {
    "NOT_INSTALLED": 1, "SEMI_EQUIPPED": 2, "INSTALLED": 3, "HYPER_EQUIPPED": 4
}


def to_bool(val):
    return str(val).lower() in ["true", "1", "yes"]


def check_missing_fields_strict(input_data: Dict[str, Any], expected_fields: list) -> None:
    missing = [key for key in expected_fields if key not in input_data]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")


def predict(input_data: Dict[str, Any]) -> float:
    check_missing_fields_strict(input_data, REQUIRED_FIELDS)

    df = pd.DataFrame([{
        "bedroomCount": int(input_data["rooms-number"]),
        "bathroomCount": 1,
        "habitableSurface": float(input_data["area"]),
        "toiletCount": 1,
        "terraceSurface": float(input_data.get("terrace-area", 0)),
        #"postCode": int(input_data["zip-code"]),
        "gardenSurface": float(input_data.get("garden-area", 0)),
        "province_encoded": province_map(input_data["zip-code"]),
        "type_encoded": type_map.get(input_data["property-type"].upper(), 2),
        "subtype_encoded": subtype_map.get(input_data["subtype of property"].upper(), 1),
        "epcScore_encoded": int(input_data["epcScore"]),
        "hasAttic_encoded": 0,
        "hasGarden_encoded": int(to_bool(input_data.get("garden", False))),
        "hasAirConditioning_encoded": 0,
        "hasArmoredDoor_encoded": 0,
        "hasVisiophone_encoded": 0,
        "hasTerrace_encoded": int(to_bool(input_data.get("terrace", False))),
        "hasOffice_encoded": 0,
        "hasSwimmingPool_encoded": int(to_bool(input_data.get("swimming_pool", False))),
        "hasFireplace_encoded": int(to_bool(input_data.get("open_fire", False))),
        "hasBasement_encoded": 0,
        "hasDressingRoom_encoded": 0,
        "hasDiningRoom_encoded": 0,
        "hasLift_encoded": 0,
        "hasHeatPump_encoded": 0,
        "hasPhotovoltaicPanels_encoded": 0,
        "hasLivingRoom_encoded": 1,
    }])


    predicted_price = model.predict(df)[0]
    return round(predicted_price, 2)


# Run a test manually
if __name__ == "__main__":
    example_input = {
        "area": 120,
        "property-type": "house",
        "subtype of property": "house",
        "rooms-number": 3,
        "zip-code": 1000,
        "land-area": 300,
        #"building_condition": 5,
        "garden": True,
        "garden-area": 50,
        "equipped-kitchen": True,
        "full-address": "boulevard edouard, 1000 laeken",
        "swimming_pool": False,
        "furnished": False,
        "open_fire": True,
        "terrace": True,
        "terrace-area": 20,
        "facades-number": 2,
        "building_condition": 5,
        "parking": False,
        "epcScore": 4,
        "heating type": 1,
        "flood zone type": 1,
        "kitchen types": 3
    }

    price = predict(example_input)
    print(f"Predicted price: {price} €")
