from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import ItineraryForm

app = FastAPI()

# Allow frontend (HTML) to talk to API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

@app.post("/itinerary")
async def receive_itinerary(data: ItineraryForm):
    """
    Receives validated itinerary form data.
    """
    print("\n📥 Incoming Itinerary Form:")
    print(data.dict())

    return {"status": "yes"}
