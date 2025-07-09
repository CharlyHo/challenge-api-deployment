from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from predict.prediction import predict

app = FastAPI()

templates = Jinja2Templates(directory="templates")
#app.mount("/static", StaticFiles(directory="static"), name="static")


class PropertyInput(BaseModel):
    area: int
    property_type: str
    subtype_of_property: str
    rooms_number: int
    zip_code: str
    land_area: Optional[int] = 0
    garden: Optional[bool] = False
    garden_area: Optional[int] = 0
    full_address: Optional[str] = None
    swimming_pool: Optional[bool] = False
    open_fire: Optional[bool] = False
    terrace: Optional[bool] = False
    terrace_area: Optional[int] = 0
    facades_number: Optional[int] = None
    building_condition: Optional[str] = None
    parking: Optional[bool] = False
    epcScore: Optional[str] = None
    heating_type: Optional[str] = None
    flood_zone_type: Optional[str] = None
    kitchen_types: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(request: Request):
    form = await request.form()

    area = float(form.get("area"))
    property_type = form.get("property_type")
    subtype_of_property = form.get("subtype_of_property")
    rooms_number = int(form.get("rooms_number"))
    zip_code = form.get("zip_code")
    land_area = form.get("land_area")
    land_area = float(land_area) if land_area else None

    garden = form.get("garden") == "true"
    garden_area = form.get("garden_area")
    garden_area = float(garden_area) if garden_area else None

    full_address = form.get("full_address")
    swimming_pool = form.get("swimming_pool") == "true"
    open_fire = form.get("open_fire") == "true"
    terrace = form.get("terrace") == "true"
    terrace_area = form.get("terrace_area")
    terrace_area = float(terrace_area) if terrace_area else None

    facades_number = form.get("facades_number")
    facades_number = int(facades_number) if facades_number else None

    building_condition = form.get("building_condition")
    parking = form.get("parking") == "true"
    epc_score = form.get("epcScore")
    epc_score = int(epc_score) if epc_score else None

    heating_type = form.get("heating_type")
    flood_zone_type = form.get("flood_zone_type")
    kitchen_types = form.get("kitchen_types")

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
    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})
    return {"predicted_price": price}