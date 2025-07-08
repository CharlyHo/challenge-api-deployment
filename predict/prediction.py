
import joblib
import pandas as pd
from typing import Dict, Any

# Load model
model = joblib.load("model/poly_model.pkl")

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



def predict(input_data: Dict[str, Any]) -> float:
    """
    Takes a dictionary representing the preprocessed data
    of a property and returns the predicted price.
    """
    
    # Prepare the data in the correct format (single-row DataFrame)
    processed_data = {
        "type": [property_type.get(input_data["property-type"])],
        "subtype": [subtype_map.get(input_data["subtype of property"])],
        "bedroomCount": [input_data["bedroomCount"]],
        "postCode": [str(input_data["postCode"])],
        "habitableSurface": [input_data["area"]],
        "landArea": [input_data.get("land-area", 0)],
        "gardenSurface": [input_data.get("garden-area", 0)],
        "hasTerrace": [int(bool(input_data.get("terrace", False)))],
        "hasParking": [int(bool(input_data.get("parking", False)))],
        "epc_score": [int(bool(input_data.get("terrace", False)))],
        "building_condition": [input_data["building condition"]],
        "kitchen_types": [input_data.get["kitchen type"]]
    }

    df = pd.DataFrame(processed_data)

    # Prediction
    predicted_price = model.predict(df)[0]
    return round(predicted_price, 2)

    price = predict(example_input)
    print(f"Predict price : {price} €")