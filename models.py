from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import date


class ItineraryForm(BaseModel):
    group_size: int = Field(..., ge=1, le=20)
    seniors: int = Field(..., ge=0)
    children: int = Field(..., ge=0)

    duration: int = Field(..., ge=1, le=30)
    start_date: date

    budget: int = Field(..., ge=5000, le=200000)

    vibes: List[str]
    specific_places: Optional[str] = ""
    preferences: Optional[str] = ""

    # Validation rules
    @validator("vibes")
    def at_least_one_vibe(cls, v):
        if len(v) == 0:
            raise ValueError("At least one vibe must be selected")
        return v

    @validator("children")
    def validate_group_composition(cls, children, values):
        seniors = values.get("seniors", 0)
        group_size = values.get("group_size", 0)
        if seniors + children > group_size:
            raise ValueError("Seniors + children cannot exceed total group size")
        return children
