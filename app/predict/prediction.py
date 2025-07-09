import joblib
import pandas as pd
from typing import Dict, Any

# Load model
model = joblib.load("best_model.pkl")

# Define encodings
province_map = lambda zip_code: int(str(zip_code)[:2]) if zip_code else 10  # default province

REQUIRED_FIELDS = [
    "area", "property-type", "subtype of property", "rooms-number", "zip-code",
    "land-area", "garden", "garden-area", "terrace", "terrace-area",
    "parking", "epcScore", "heating type", "flood zone type",
    "building_condition", "kitchen types"
]


def check_missing_fields_strict(input_data: Dict[str, Any], expected_fields: list) -> None:
    missing = [key for key in expected_fields if key not in input_data]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")


def predict(input_data: Dict[str, Any]) -> float:
    check_missing_fields_strict(input_data, REQUIRED_FIELDS)

    df = pd.DataFrame([{
        "bedroomCount": input_data["rooms-number"],
        "bathroomCount": 1,  # or derive from data if available
        "habitableSurface": input_data["area"],
        "toiletCount": 1,
        "terraceSurface": input_data.get("terrace-area", 0),
        "postCode": input_data["zip-code"],
        "gardenSurface": input_data.get("garden-area", 0),
        "province_encoded": province_map(input_data["zip-code"]),
        "type_encoded": input_data["property-type"],
        "subtype_encoded": input_data["subtype of property"],
        "epcScore_encoded": input_data["epcScore"],
        "hasAttic_encoded": 0,
        "hasGarden_encoded": int(input_data.get("garden", False)),
        "hasAirConditioning_encoded": 0,
        "hasArmoredDoor_encoded": 0,
        "hasVisiophone_encoded": 0,
        "hasTerrace_encoded": int(input_data.get("terrace", False)),
        "hasOffice_encoded": 0,
        "hasSwimmingPool_encoded": int(input_data.get("swimming_pool", False)),
        "hasFireplace_encoded": int(input_data.get("open_fire", False)),
        "hasBasement_encoded": 0,
        "hasDressingRoom_encoded": 0,
        "hasDiningRoom_encoded": 0,
        "hasLift_encoded": 0,
        "hasHeatPump_encoded": 0,
        "hasPhotovoltaicPanels_encoded": 0,
        "hasLivingRoom_encoded": 1,  # assumed
    }])

    predicted_price = model.predict(df)[0]
    return round(predicted_price, 2)


# Run a test manually
if __name__ == "__main__":
    example_input = {
        "area": 120,
        "property-type": 2,
        "subtype of property": 2,
        "rooms-number": 3,
        "zip-code": 1000,
        "land-area": 300,
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
