
import joblib
import pandas as pd
from typing import Dict, Any

# Load model
model = joblib.load("model/poly_model.pkl")

# Map text columns with the same encodings as in training
property_type_map = {
    "APARTMENT": "APARTMENT",
    "HOUSE": "HOUSE",
    "OTHERS": "OTHER"
}

subtype_map = {
    "APARTMENT": "APARTMENT",
    "DUPLEX": "DUPLEX",
    "PENTHOUSE": "PENTHOUSE",
    "STUDIO": "STUDIO",
    "HOUSE": "HOUSE",
    "VILLA": "VILLA",
    "BUNGALOW": "BUNGALOW",
    "MANSION": "MANSION",
    "OTHER": "OTHER",
    "MIXED_USE": "OTHER",
    "FARMHOUSE": "FARMHOUSE",
    "LOFT": "APARTMENT",
    "TOWN_HOUSE": "HOUSE"
}


def predict(input_data: Dict[str, Any]) -> float:
    """
    Takes a dictionary representing the preprocessed data
    of a property and returns the predicted price.
    """
    
    # Prepare the data in the correct format (single-row DataFrame)
    processed_data = {
        "subtype": [subtype_map.get(input_data["subtype of property"], "OTHER")],
        "type": [property_type_map.get(input_data["property-type"], "OTHER")],
        "bedroomCount": [input_data["rooms-number"]],
        "province": [str(input_data["zip-code"])[:2]],  # province extraction
        "habitableSurface": [input_data["area"]],
        "landArea": [input_data.get("land-area", 0)],
        "gardenSurface": [input_data.get("garden-area", 0)],
        "hasTerrace": [int(bool(input_data.get("terrace", False)))],
        "hasParking": [int(bool(input_data.get("parking", False)))],
        "epc_score": [int(bool(input_data.get("terrace", False)))],
        "building_condition": [input_data["building condition"]]
    }

    
    df = pd.DataFrame(processed_data)

    # Prediction
    predicted_price = model.predict(df)[0]
    return round(predicted_price, 2)


if __name__ == "__main__":
    example_input = {
        "area": 120,
        "property-type": "HOUSE",
        "subtype of property": "HOUSE",
        "rooms-number": 3,
        "zip-code": 1020,
        "land-area": 300,
        "garden": True,
        "garden-area": 50,
        "equipped-kitchen": True,
        "full-address": "boulevard edouard, 1000 laeken",
        "swimming-pool": False,
        "furnished": False,
        "open-fire": True,
        "terrace": True,
        "terrace-area": 20,
        "facades-number": 2,
        "building condition": 5,
        "parking": False,
        "epcScore": 4
    }


    price = predict(example_input)
    print(f"Predict price : {price} €")