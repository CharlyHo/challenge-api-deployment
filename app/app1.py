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
    epcScore = form.get("epcScore")
    epcScore = int(epcScore) if epcScore else None

    heating_type = form.get("heating_type")
    flood_zone_type = form.get("flood_zone_type")
    kitchen_types = form.get("kitchen_types")

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
        "swimming-pool": swimming_pool,
        "open-fire": open_fire,
        "terrace": terrace,
        "terrace-area": terrace_area,
        "facades-number": facades_number,
        "building condition": building_condition,
        "parking": parking,
        "epcScore": epcScore,
        "heating type": heating_type,
        "flood zone type": flood_zone_type,
        "kitchen types": kitchen_types
    }

    price = predict(input_dict)
    return templates.TemplateResponse("index.html", {"request": request, "prediction": price})
