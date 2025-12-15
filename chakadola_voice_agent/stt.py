import os

from dotenv import load_dotenv
from groq import Groq

# Ensure .env is loaded even when this module is imported independently
load_dotenv()

_client = None


def _get_client() -> Groq:
    """Lazy init the Groq client; fail fast with a clear error if key missing."""
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Please export it to use the voice agent (Whisper)."
        )

    _client = Groq(api_key=api_key)
    return _client


def speech_to_text(audiofile: str) -> str:
    """
    Convert audio file to English text using Groq's whisper-large-v3 model.
    Uses the TRANSLATIONS endpoint to automatically translate Odia/Hindi/other 
    languages to English.
    
    Note: Only whisper-large-v3 supports translation, NOT whisper-large-v3-turbo.
    """
    client = _get_client()
    with open(audiofile, "rb") as audio_file:
        # Use TRANSLATIONS endpoint with whisper-large-v3 for direct English output
        # This translates any language (Odia, Hindi, Bengali, etc.) to English
        translation = client.audio.translations.create(
            file=(audiofile, audio_file.read()),
            model="whisper-large-v3",  # MUST use whisper-large-v3 for translation support
            prompt="Odisha tourism travel planning. Places: Puri, Konark, Bhubaneswar, Chilika, Jagannath Temple.",
            response_format="text",
            temperature=0.0
        )

    return translation.text if hasattr(translation, 'text') else str(translation)


