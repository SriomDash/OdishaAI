import os
import json
import logging
from dotenv import load_dotenv
from .prompts import VOICE_EXTRACTION_PROMPT
from .schemas import VoiceTripData
import google.generativeai as genai

# Ensure .env is loaded even when imported directly
load_dotenv()

logger = logging.getLogger(__name__)

genai.configure(api_key=os.getenv("GEMINI_API_KEY")) 
model = genai.GenerativeModel( 
    model_name="gemini-2.5-flash", 
    generation_config={ 
        "temperature": 0.1,  # Slightly higher for better translation
        "response_mime_type": "application/json"
    } )



def extract_trip_data(text: str) -> dict:
    """
    Extract structured trip data from transcribed speech.
    Returns a safe fallback if extraction fails.
    """
    prompt = f"""
{VOICE_EXTRACTION_PROMPT}

User speech (translated to English):
{text}
"""

    logger.info(f"🎤 Extracting trip data from: {text[:100]}...")
    
    try:
        response = model.generate_content(prompt)

        if not response or not response.text:
            logger.error("❌ Empty response from Gemini")
            raise ValueError("Empty response from Gemini")

        logger.info(f"✅ Gemini response: {response.text[:200]}...")
        result = json.loads(response.text)
        return result

    except Exception as e:
        logger.error(f"❌ Extraction failed: {e}", exc_info=True)
        # Safe fallback: return empty extraction with low confidence
        return {
            "group_size": None,
            "seniors": None,
            "children": None,
            "specially_abled": None,
            "duration": None,
            "start_date": None,
            "budget": None,
            "vibes": None,
            "specific_places": None,
            "preferences": None,
            "confidence": 0.0
        }

