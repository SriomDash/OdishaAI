# it defines the shape of the data the voice agent will return 

from pydantic import BaseModel
from typing import List, Optional

class VoiceTripData(BaseModel):
    group_size: Optional[int] = None  # number of people in the trip
    seniors: Optional[int] = None  # number of seniors in the trip
    children: Optional[int] = None  # number of children in the trip
    duration: Optional[int] = None  # number of days in the trip
    start_date: Optional[str] = None  # start date of the trip
    budget: Optional[int] = None  # budget for the trip
    vibes: Optional[List[str]] = None  # vibes of the trip
    specially_abled: Optional[int] = None  # number of specially abled travellers
    specific_places: Optional[str] = None  # specific places to include in the trip
    preferences: Optional[str] = None  # preferences for the trip

    transcript: Optional[str] = None  # English transcript from Whisper translation
    confidence: float

    class Config:
        extra = "ignore"
