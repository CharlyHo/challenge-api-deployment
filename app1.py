from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from predict.prediction import predict 

app = FastAPI()

class PropertyInput(BaseModel):
    area: int
    property_type: str
    subtype_of_property: str
    bedroomCount: int
    postCode: int
    land_area: Optional[int] = 0
    garden: Optional[bool] = False
    garden_area: Optional[int] = 0
    equipped_kitchen: Optional[bool] = False
    swimming_pool: Optional[bool] = False
    furnished: Optional[bool] = False
    open_fire: Optional[bool] = False
    terrace: Optional[bool] = False
    terrace_area: Optional[int] = 0
    facades_number: Optional[int] = None
    building_condition: int
    parking: Optional[bool] = False
    epcScore: Optional[int] = 5

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
        "equipped-kitchen": data.equipped_kitchen,
        "full-address": data.full_address,
        "swimming-pool": data.swimming_pool,
        "furnished": data.furnished,
        "open-fire": data.open_fire,
        "terrace": data.terrace,
        "terrace-area": data.terrace_area,
        "facades-number": data.facades_number,
        "building condition": data.building_condition,
        "parking": data.parking,
        "epcScore": data.epcScore
    }
    price = predict(input_dict)
    return {"predicted_price": price}