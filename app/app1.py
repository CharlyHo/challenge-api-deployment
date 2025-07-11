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
    building_condition: Optional[str] = None
    parking: Optional[bool] = False
    epcScore: Optional[str] = None
    heating_type: Optional[str] = None
    flood_zone_type: Optional[str] = None
    kitchen_types: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict_view(request: Request):
    form = await request.form() 

    area = float(form.get("area", 0)),
    property_type = form.get("property_type"),
    subtype_of_property = form.get("subtype_of_property"),
    rooms_number = int(form.get("rooms_number", 0)),
    zip_code = form.get("zip_code"),
    land_area = float(form.get("land_area") or 0),
    garden = form.get("garden") == "true",
    garden_area = float(form.get("garden_area") or 0),
    full_address = form.get("full_address"),
    swimming_pool = form.get("swimming_pool") == "true",
    open_fire = form.get("open_fire") == "true",
    terrace = form.get("terrace") == "true",
    terrace_area = float(form.get("terrace_area") or 0),
    facades_number = int(form.get("facades_number") or 0),
    building_condition = form.get("building_condition"),
    parking = form.get("parking") == "true",
    epcScore = form.get("epcScore"),
    heating_type = form.get("heating_type"),
    flood_zone_type = form.get("flood_zone_type"),
    kitchen_types = form.get("kitchen_types")

    """area = float(form("area")),
    property_type = form("property_type"),
    subtype_of_property = form("subtype_of_property"),
    rooms_number = int(form("rooms_number")),
    zip_code = form("zip_code"),
    land_area = float(form("land_area")) if land_area else None,

    garden = form("garden") == "true",
    garden_area = float(form("garden_area")) if garden_area else None,

    full_address = form("full_address"),
    swimming_pool = form("swimming_pool") == "true",
    open_fire = form("open_fire") == "true",
    terrace = form("terrace") == "true",
    terrace_area = float(form("terrace_area")) if terrace_area else None,

    facades_number = int(form("facades_number")) if facades_number else None,

    building_condition = form("building_condition"),
    parking = form("parking") == "true",
    epcScore = int(form("epcScore")) if epcScore else None,

    heating_type = form("heating_type"),
    flood_zone_type = form("flood_zone_type"),
    kitchen_types = form("kitchen_types")
"""

    input_dict = {

    "area": area,
    "property-type": property_type,
    "subtype of property": subtype_of_property,
    "rooms-number": rooms_number,
    "zip-code": zip_code,
    "land-area": land_area,
    "garden": garden,
    "garden-area": garden_area,
    "full-address": full_address,
    "swimming_pool": swimming_pool,
    "open_fire": open_fire,
    "terrace": terrace,
    "terrace-area": terrace_area,
    "facades-number": facades_number,
    "building_condition": building_condition,
    "parking": parking,
    "epcScore": epcScore,
    "heating type": heating_type,
    "flood zone type": flood_zone_type,
    "kitchen types": kitchen_types
}


    price = predict(input_dict)
    return templates.TemplateResponse("index.html", {"request": request, "prediction": price})
