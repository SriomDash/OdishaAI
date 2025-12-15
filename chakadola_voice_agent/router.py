from fastapi import APIRouter, UploadFile, File
import tempfile
import os
import pathlib

from .stt import speech_to_text
from .nlu import extract_trip_data
from .schemas import VoiceTripData


router = APIRouter(prefix="/voice", tags=["Voice Agent"])


@router.post("/process", response_model=VoiceTripData)
async def process_voice_input(file: UploadFile = File(...)) -> VoiceTripData:
    """
    Accepts an audio file, converts speech to text,
    extracts structured trip data using Gemini,
    and returns validated JSON for manual verification.
    """

    # Save the uploaded file temporarily (keep extension for Whisper compatibility)
    suffix = pathlib.Path(file.filename or "").suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        audio_path = tmp.name

    try:
        # Convert speech to text (Whisper)
        transcript = speech_to_text(audio_path)

        # Extract structured data from the text (Gemini)
        extracted_data = extract_trip_data(transcript)
        extracted_data["transcript"] = transcript

        # Validate and return JSON
        validated_data = VoiceTripData(**extracted_data)
        return validated_data

    finally:
        # Delete the temporary file
        if os.path.exists(audio_path):
            os.remove(audio_path)