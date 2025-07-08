
import joblib
import pandas as pd
from typing import Dict, Any

# Load model
model = joblib.load("model/best_model.pkl")

# Map text columns with the same encodings as in training

property_type = {1: "APARTMENT", 2: "HOUSE"}

subtype_mapping = {
    1: "APARTMENT",
    2: "HOUSE",
    3: "FLAT STUDIO",
    4: "DUPLEX",
    5: "PENTHOUSE",
    6: "GROUNDFLOOR",
    7: "APARTMENT BLOCK",
    8: "MANSION",
    9: "EXCEPTIONAL PROPERTY",
    10: "MIXED USE BUILDING",
    11: "TRIPLEX",
    12: "LOFT",
    13: "VILLA",
    14: "TOWN HOUSE",
    15: "CHALET",
    16: "MANOR HOUSE",
    17: "SERVICE FLAT",
    18: "KOT",
    19: "FARMHOUSE",
    20: "BUNGALOW",
    21: "COUNTRY COTTAGE",
    22: "OTHER PROPERTY",
    23: "CASTLE",
    24: "PAVILION"
    }

building_conditions = {
    -1: "Unknown",
    1: "GOOD",
    2: "AS NEW",
    3: "TO RENOVATE", 
    4: "TO BE DONE UP",
    5: "JUST RENOVATED",
    6: "TO RESTORE"
}

epc_scores = {
    1:'A++',
    2: 'A+',
    3: 'A',
    4: 'B',
    5: 'C',
    6: 'D',
    7: 'E',
    8: 'F',
    9: 'G',
    -1: 'Unknown'
}

heating_types = {
    -1: "Unknown",
    1: 'GAS',
    2: 'FUELOIL',
    3: 'ELECTRIC',
    4: 'PELLET',
    5: 'WOOD',
    6: 'SOLAR',
    7: 'CARBON'
}

flood_zone_types = {
    -1: 'Unknown',
    1: 'NON FLOOD ZONE',
    2: 'POSSIBLE FLOOD ZONE',
    3: 'RECOGNIZED FLOOD ZONE',
    4: 'RECOGNIZED N CIRCUMSCRIBED FLOOD ZONE',
    5: 'CIRCUMSCRIBED WATERSIDE ZONE',
    6: 'CIRCUMSCRIBED FLOOD ZONE',
    7: 'POSSIBLE N CIRCUMSCRIBED FLOOD ZONE',
    8: 'POSSIBLE N CIRCUMSCRIBED WATERSIDE ZONE',
    9: 'RECOGNIZED N CIRCUMSCRIBED WATERSIDE FLOOD ZONE'
}

kitchen_types = {
    -1: "Unknow",
    0: 'NOT_INSTALLED',
    1: 'SEMI_EQUIPPED',
    2: 'INSTALLED',
    3: 'HYPER_EQUIPPED'
}



REQUIRED_FIELDS = [
    "property-type",
    "subtype of property",
    "rooms-number",
    "zip-code",
    "area",
    "land-area",
    "garden-area",
    "terrace",
    "parking",
    "Energy performance certificate score",
    "heating type",
    "flood zone type",
    "building_condition",
    "kitchen types"
]



def check_missing_fields_strict(input_data: Dict[str, Any], expected_fields: list) -> None:
    missing = [key for key in expected_fields if key not in input_data]
    if missing:
        print("[WARNING] mandatory missing field :")
        for key in missing:
            print(f" → '{key}' is mandatory but missing in the gave data.")



def predict(input_data: Dict[str, Any]) -> float:
    """
    Takes a dictionary representing the preprocessed data
    of a property and returns the predicted price.
    """

    check_missing_fields_strict(input_data, REQUIRED_FIELDS)
    
    # Prepare the data in the correct format (single-row DataFrame)
    processed_data = {

        "type": [property_type.get(input_data["property-type"])],
        "subtype": [subtype_mapping.get(input_data["subtype of property"])],
        "bedroomCount": [input_data["rooms-number"]],
        "postCode": [str(input_data["zip-code"])],  # province extraction
        "province": [str(input_data["zip-code"])[:2]],  # province extraction
        "habitableSurface": [input_data["area"]],
        "landArea": [input_data.get("land-area", 0)],
        "gardenSurface": [input_data.get("garden-area", 0)],
        "hasTerrace": [int(bool(input_data.get("terrace", False)))],
        "hasParking": [int(bool(input_data.get("parking", False)))],
        "epc_score": [input_data["epcScore"]],
        "heatingType": [heating_types.get(input_data["heating type"])], 
        "floodZoneTypes": [flood_zone_types.get(input_data["flood zone type"])], 
        "building_condition": [input_data["building condition"]],
        "kitchenType": [kitchen_types.get(input_data["kitchen types"])] 
    }

    

    df = pd.DataFrame(processed_data)

    # Prediction
    predicted_price = model.predict(df)[0]
    return round(predicted_price, 2)


if __name__ == "__main__":
    example_input = {
        "area": 120,
        "property-type": 2,
        "subtype of property": 2,
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
        "epcScore": 4,
        "heating type": 1,
        "flood zone type": 1, 
        "kitchen types": 3

    }




    price = predict(example_input)
    print(f"Predict price : {price} €")