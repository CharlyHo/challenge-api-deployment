from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from predict.prediction import predict 

app = FastAPI()

class PropertyInput(BaseModel):
    area: int
    property_type: int
    subtype_of_property: int
    rooms_number: int
    zip_code: int
    land_area: Optional[int] = 0
    garden: Optional[bool] = False
    garden_area: Optional[int] = 0
    full_address: Optional[str] = None
    swimming_pool: Optional[bool] = False
    open_fire: Optional[bool] = False
    terrace: Optional[bool] = False
    terrace_area: Optional[int] = 0
    facades_number: Optional[int] = None
    building_condition: int
    parking: Optional[bool] = False
    epcScore: int
    heating_type: int  
    flood_zone_type: int
    kitchen_types: int  

@app.post("/predict")
def predict_price(data: PropertyInput):
    input_dict = {
        "area": data.area,
        "property-type": data.property_type,
        "subtype of property": data.subtype_of_property,
        "rooms-number": data.rooms_number,
        "zip-code": data.zip_code,
        "land-area": data.land_area,
        "garden": data.garden,
        "garden-area": data.garden_area,
        "full-address": data.full_address,
        "swimming-pool": data.swimming_pool,
        "open-fire": data.open_fire,
        "terrace": data.terrace,
        "terrace-area": data.terrace_area,
        "facades-number": data.facades_number,
        "building condition": data.building_condition,
        "parking": data.parking,
        "epcScore": data.epcScore,
        "heating type": data.heating_type,
        "flood zone type": data.flood_zone_type,
        "kitchen types": data.kitchen_types
    }

    
    price = predict(input_dict)
    return {"predicted_price": price}