import os
import json
import random
from typing import Any, Dict, List, TypedDict
from dotenv import load_dotenv

load_dotenv()

# -------------------------------------------------------------------
#  Gemini LLM Wrapper (OpenAI-Compatible API for Gemini)
# -------------------------------------------------------------------
from openai import OpenAI

class GeminiLLM:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        print("✅ GeminiLLM initialized.")

    def invoke(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": "You are a helpful travel planning assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content


# Instantiate LLM
llm = GeminiLLM(api_key=os.getenv("GEMINI_API_KEY"))


# -------------------------------------------------------------------
#  Embedding Model (must match ChromaDB loader)
# -------------------------------------------------------------------
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-mpnet-base-v2"
embedder = SentenceTransformer(EMBED_MODEL)
EMBED_DIM = embedder.get_sentence_embedding_dimension()

print(f"✅ Loaded embedder {EMBED_MODEL} (dim={EMBED_DIM})")


# -------------------------------------------------------------------
#  ChromaDB Client (NEW 0.5+ architecture)
# -------------------------------------------------------------------
from chromadb import Client
from chromadb.config import Settings

CHROMA_DIR = "./odisha_chroma"
COLLECTION_NAME = "odisha_tourism"

try:
    chroma_client = Client(
        Settings(
            is_persistent=True,
            persist_directory=CHROMA_DIR
        )
    )
    collection = chroma_client.get_collection(COLLECTION_NAME)
    print("✅ Chroma collection loaded successfully.")

except Exception as e:
    print("⚠ Failed to initialize Chroma client/collection:", e)
    collection = None  # Graph will safely crash with clear error


# -------------------------------------------------------------------
#  LangGraph Imports
# -------------------------------------------------------------------
from langgraph.graph import StateGraph, START, END


# -------------------------------------------------------------------
#  STATE MODEL
# -------------------------------------------------------------------
class State(TypedDict, total=False):
    request: Dict[str, Any]
    selected_places: List[str]
    rag_results: List[Dict[str, Any]]
    weather_cost_info: List[Dict[str, Any]]
    map_info: Dict[str, Any]
    final_itinerary: Dict[str, Any]


# -------------------------------------------------------------------
#  HELPERS
# -------------------------------------------------------------------
def extract_places(text: str) -> List[str]:
    return [p.strip() for p in text.split(",") if p.strip()]


def categorize_temperature(c: float) -> str:
    if c < 20:
        return "Cool"
    if c <= 30:
        return "Warm"
    return "Hot"


def categorize_humidity(h: float) -> str:
    if h < 40:
        return "Dry"
    if h <= 70:
        return "Comfortable"
    return "Humid"


def rough_cost_model(budget: int, duration: int):
    per_day = budget / duration
    return {
        "stay": round(per_day * 0.45),
        "food": round(per_day * 0.25),
        "travel": round(per_day * 0.20),
        "misc": round(per_day * 0.10),
        "total_per_day": round(per_day),
    }


# -------------------------------------------------------------------
# 1️⃣ MAIN NODE → Select Places
# -------------------------------------------------------------------
def main_function(state: State):
    req = state["request"]
    user_places = req.get("specific_places", "").strip()

    if user_places:
        return {"selected_places": extract_places(user_places)}

    prompt = f"""
    Suggest 3–6 best places in Odisha based on:
    - vibes: {req.get("vibes")}
    - budget: {req.get("budget")}
    - duration: {req.get("duration")}
    - preferences: {req.get("preferences")}
    Return ONLY comma-separated place names.
    """

    generated = extract_places(llm.invoke(prompt))

    if not generated:
        generated = ["Puri", "Konark", "Chilika"]

    return {"selected_places": generated}


# -------------------------------------------------------------------
# 2️⃣ RAG NODE → Query ChromaDB for metadata
# -------------------------------------------------------------------
def rag_function(state: State):
    if collection is None:
        raise RuntimeError("Chroma collection is not available.")

    selected = state["selected_places"]
    rag_results = []

    for place in selected:

        # 🔥 embed the query using SAME model (all-mpnet-base-v2)
        query_vec = embedder.encode([place])[0]

        # 🔍 Query using embeddings, not raw text
        response = collection.query(
            query_embeddings=[query_vec],
            n_results=3
        )

        if not response or not response.get("metadatas") or len(response["metadatas"][0]) == 0:
            rag_results.append({
                "place_name": place,
                "description": None,
                "lat": None,
                "lng": None,
                "district": None,
                "city": None,
                "entry_fee": None,
                "stay_cost": None,
                "travel_cost": None,
                "raw_meta": {}
            })
            continue

        meta = response["metadatas"][0][0]

        rag_results.append({
            "place_name": meta.get("place_name", place),
            "description": meta.get("description"),
            "lat": meta.get("lat"),
            "lng": meta.get("lng"),
            "district": meta.get("district"),
            "city": meta.get("city"),
            "entry_fee": meta.get("entry_fee"),
            "stay_cost": meta.get("stay_cost"),
            "travel_cost": meta.get("travel_cost"),
            "raw_meta": meta
        })

    return {"rag_results": rag_results}


# -------------------------------------------------------------------
# 3️⃣ WEATHER + COST NODE (Mock weather)
# -------------------------------------------------------------------
def weather_cost_function(state: State):
    req = state["request"]
    rag = state["rag_results"]

    duration = req["duration"]
    budget = req["budget"]

    output = []

    for place in rag:
        temp = round(random.uniform(26, 36), 1)
        humidity = round(random.uniform(45, 80), 1)

        weather = {
            "temp_c": temp,
            "humidity": humidity,
            "temp_desc": categorize_temperature(temp),
            "humidity_desc": categorize_humidity(humidity),
            "summary": f"{categorize_temperature(temp)} ({temp}°C), {categorize_humidity(humidity)} humidity ({humidity}%)",
        }

        cost = rough_cost_model(budget, duration)

        output.append({
            "place": place["place_name"],
            "weather": weather,
            "cost": cost,
        })

    return {"weather_cost_info": output}


# -------------------------------------------------------------------
# 4️⃣ MAP NODE → Leaflet-ready coordinates
# -------------------------------------------------------------------
def map_function(state: State):
    rag = state["rag_results"]

    leaflet_points = [
        {"name": p["place_name"], "lat": p["lat"], "lng": p["lng"]}
        for p in rag
        if p.get("lat") and p.get("lng")
    ]

    # Ask Gemini to optimize travel path
    prompt = f"""
    Arrange these Odisha places in the best travel route:
    {leaflet_points}
    Return ONLY comma-separated names.
    """

    try:
        route_order = extract_places(llm.invoke(prompt))
    except:
        route_order = [p["name"] for p in leaflet_points]

    coords_array = [[p["lat"], p["lng"]] for p in leaflet_points]

    return {
        "map_info": {
            "leaflet_points": leaflet_points,
            "route_order": route_order,
            "coords_array": coords_array,
        }
    }


# -------------------------------------------------------------------
# 5️⃣ FINAL NODE → Build Day-wise Itinerary JSON
# -------------------------------------------------------------------
def final_function(state: State):
    req = state["request"]
    places = state["selected_places"]
    weather_cost = state["weather_cost_info"]
    mapinfo = state["map_info"]

    days = []
    duration = req["duration"]

    chunk = max(1, len(places) // duration)
    idx = 0

    for day in range(1, duration + 1):
        today_places = places[idx: idx + chunk]
        idx += chunk

        entry = {
            "day": day,
            "destinations": today_places,
            "time_plan": f"Start 8 AM → explore {', '.join(today_places)} → return by 7 PM",
            "weather": weather_cost[0]["weather"],
            "stay": "Recommended: clean budget homestay/hotel",
            "cost": weather_cost[0]["cost"],
        }
        days.append(entry)

    return {
        "final_itinerary": {
            "days": days,
            "route": mapinfo["route_order"],
        }
    }


# -------------------------------------------------------------------
# BUILD GRAPH
# -------------------------------------------------------------------
def build_graph():
    graph = StateGraph(State)

    graph.add_node("main_function", main_function)
    graph.add_node("rag_function", rag_function)
    graph.add_node("weather_cost_function", weather_cost_function)
    graph.add_node("map_function", map_function)
    graph.add_node("final_function", final_function)

    graph.add_edge(START, "main_function")
    graph.add_edge("main_function", "rag_function")
    graph.add_edge("rag_function", "weather_cost_function")
    graph.add_edge("rag_function", "map_function")
    graph.add_edge("weather_cost_function", "final_function")
    graph.add_edge("map_function", "final_function")
    graph.add_edge("final_function", END)

    print("✅ Chakadola graph compiled successfully.")
    return graph.compile()


chakadola_chain = build_graph()
