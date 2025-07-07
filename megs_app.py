from fastapi import FastAPI, Depends, HTTPException, status
import os
from pydantic import BaseModel
import pandas as pd

app = FastAPI()


class Data(BaseModel):
    type: ""
    subtype: ""
    bedroomCount: int
    bathroomCount: Optional[int] = 0
    postCode: ""
    habitableSurface: int
    roomCount: int
    hasAttic: bool
    hasBasement: bool
    hasDressingRoom: bool
    diningRoomSurface: Optional[int] = 0
    hasDiningRoom: bool
    buildingCondition: 
    buildingConstructionYear,facedeCount,floorCount,streetFacadeWidth,hasLift,floodZoneType,heatingType,hasHeatPump,hasPhotovoltaicPanels,hasThermicPanels,kitchenSurface,kitchenType,landSurface,hasLivingRoom,livingRoomSurface,hasGarden,gardenSurface,gardenOrientation,parkingCountIndoor,parkingCountOutdoor,hasAirConditioning,hasArmoredDoor,hasVisiophone,hasOffice,toiletCount,hasSwimmingPool,hasFireplace,hasTerrace,terraceSurface,terraceOrientation,epcScore,price,buildingConditionNormalize,epcScoreNormalize,heatingTypeNormalize,floodZoneTypeNormalize,kitchenTypeNormalize

        'bedroomCount': 1.0,
        'bathroomCount': 1.0,
        'habitableSurface': 65.0,
        'toiletCount': 1.0,
        'terraceSurface': 0.0,
        'gardenSurface': 0.0,
        'province_encoded': 1.0,
        'type_encoded': 1,
        'subtype_encoded': 1,
        'epcScore_encoded': 3.0,
        'hasAttic_encoded': 0,
        'hasGarden_encoded': 0,
        'hasAirConditioning_encoded': 0,
        'hasArmoredDoor_encoded': 1,
        'hasVisiophone_encoded': 1,
        'hasTerrace_encoded': 0,
        'hasOffice_encoded': 0,
        'hasSwimmingPool_encoded': 0,
        'hasFireplace_encoded': 0,
        'hasBasement_encoded': 0,
        'hasDressingRoom_encoded': 0,
        'hasDiningRoom_encoded': 0,
        'hasLift_encoded': 1,
        'hasHeatPump_encoded': 0,
        'hasPhotovoltaicPanels_encoded': 0,
        'hasLivingRoom_encoded': 1,