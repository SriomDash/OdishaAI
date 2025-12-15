🎤 Chakadola Voice Agent
Odia & English Voice Interface for AI Travel Planning


🌟 Overview

The Chakadola Voice Agent enables users to plan trips to Odisha using voice, making the system inclusive, accessible, and user-friendly—especially for Odia-first speakers, senior citizens, and non-tech users.

Instead of filling long forms, users simply speak their trip details, which are then:

Transcribed,

Structurally understood,

Auto-filled into the travel planner,

Verified manually before final itinerary generation.

This ensures accuracy, transparency, and trust.


🎯 Problem Addressed

Many users face difficulty with:

    Typing long travel forms

    English-only interfaces

    Complex UI flows

    Accessibility limitations

Voice interaction solves this by:

    Reducing cognitive load

    Supporting Odia language

    Allowing human verification

    Avoiding blind AI assumptions

Solution Approach :


User Speech (Odia / English)
        ↓
Speech-to-Text (Whisper)
        ↓
Natural Language Understanding (Gemini)
        ↓
Structured JSON Extraction
        ↓
Auto-Fill Travel Form
        ↓
Manual User Verification


voice_agent/
│
├── stt.py        # Speech → Text (Whisper)
├── prompts.py   # Extraction rules & constraints
├── nlu.py       # Text → Structured JSON (Gemini)
├── schemas.py   # Data validation contracts
├── router.py    # FastAPI voice endpoint
└── __init__.py


| Task             | Model            |
| ---------------- | ---------------- |
| Speech-to-Text   | OpenAI Whisper   |
| NLU / Extraction | Gemini 1.5 Flash |
| Validation       | Pydantic         |


🔌 API Endpoint
POST /voice/process

Input

Audio file (.wav, .mp3, etc.)

Output

Structured JSON for auto-filling travel form

Example Response :

{
  "group_size": 4,
  "seniors": 1,
  "children": null,
  "duration": 3,
  "start_date": "14 Feb 2025",
  "budget": 15000,
  "vibes": ["Spiritual"],
  "specific_places": "Puri, Konark",
  "preferences": "Pure veg",
  "confidence": 0.82
}


🧪 Error Handling Strategy

| Scenario       | Behavior                |
| -------------- | ----------------------- |
| Unclear speech | Fields set to `null`    |
| AI failure     | Confidence = `0.0`      |
| Invalid audio  | Safe empty response     |
| Partial input  | Partial extraction only |


No assumptions are ever made.