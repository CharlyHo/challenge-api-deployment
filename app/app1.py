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


@app.post("/predict", response_class=HTMLResponse)
def predict_price(
    request: Request,
    area: int = Form(...),
    property_type: str = Form(...),
    subtype_of_property: str = Form(...),
    rooms_number: int = Form(...),
    zip_code: str = Form(...),
    land_area: Optional[int] = Form(0),
    garden: Optional[bool] = Form(False),
    garden_area: Optional[int] = Form(0),
    full_address: Optional[str] = Form(None),
    swimming_pool: Optional[bool] = Form(False),
    open_fire: Optional[bool] = Form(False),
    terrace: Optional[bool] = Form(False),
    terrace_area: Optional[int] = Form(0),
    facades_number: Optional[int] = Form(None),
    building_condition: Optional[str] = Form(None),
    parking: Optional[bool] = Form(False),
    epcScore: Optional[str] = Form(None),
    heating_type: Optional[str] = Form(None),
    flood_zone_type: Optional[str] = Form(None),
    kitchen_types: Optional[str] = Form(None)
):

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
    return {"predicted_price": price}