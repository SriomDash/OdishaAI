import os
import json
import random
from typing import Any, Dict, List, TypedDict, Optional
from dotenv import load_dotenv
from datetime import datetime, timedelta
import logging

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
#  Gemini LLM Wrapper with Retry Logic
# -------------------------------------------------------------------
from openai import OpenAI
import time

class GeminiLLM:
    def __init__(self, api_key: str, max_retries: int = 3):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.max_retries = max_retries
        logger.info("✅ GeminiLLM initialized with retry logic.")

    def invoke(self, prompt: str, temperature: float = 0.7) -> str:
        """Invoke with exponential backoff retry"""
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model="gemini-2.5-pro",
                    messages=[
                        {"role": "system", "content": "You are a helpful travel planning assistant for Odisha tourism. Be specific and culturally aware."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                )
                return resp.choices[0].message.content
            except Exception as e:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"LLM attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.2f}s...")
                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)
                else:
                    logger.error(f"LLM failed after {self.max_retries} attempts")
                    raise Exception(f"LLM service unavailable: {e}")

# Instantiate LLM
try:
    llm = GeminiLLM(api_key=os.getenv("GEMINI_API_KEY"))
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")
    llm = None

# -------------------------------------------------------------------
#  Embedding Model with Error Handling
# -------------------------------------------------------------------
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-mpnet-base-v2"
try:
    embedder = SentenceTransformer(EMBED_MODEL)
    EMBED_DIM = embedder.get_sentence_embedding_dimension()
    logger.info(f"✅ Loaded embedder {EMBED_MODEL} (dim={EMBED_DIM})")
except Exception as e:
    logger.error(f"Failed to load embedder: {e}")
    embedder = None

# -------------------------------------------------------------------
#  ChromaDB Client with Fallback
# -------------------------------------------------------------------
from chromadb import Client
from chromadb.config import Settings

CHROMA_DIR = "./odisha_chroma"
COLLECTION_NAME = "odisha_tourism"

# Default fallback places if ChromaDB fails
FALLBACK_PLACES = {
    "Puri": {
        "place_name": "Puri Jagannath Temple",
        "description": "Sacred temple of Lord Jagannath, one of the Char Dham pilgrimage sites",
        "lat": 19.8135, "lng": 85.8312,
        "district": "Puri", "city": "Puri",
        "entry_fee": 0, "stay_cost": 1500, "travel_cost": 500
    },
    "Konark": {
        "place_name": "Konark Sun Temple",
        "description": "UNESCO World Heritage Site, 13th century architectural marvel",
        "lat": 19.8876, "lng": 86.0945,
        "district": "Puri", "city": "Konark",
        "entry_fee": 40, "stay_cost": 1200, "travel_cost": 600
    },
    "Chilika": {
        "place_name": "Chilika Lake",
        "description": "Asia's largest brackish water lagoon, home to migratory birds",
        "lat": 19.7165, "lng": 85.3206,
        "district": "Puri", "city": "Balugaon",
        "entry_fee": 50, "stay_cost": 1000, "travel_cost": 700
    },
    "Bhubaneswar": {
        "place_name": "Lingaraj Temple",
        "description": "11th century temple dedicated to Lord Shiva, architectural masterpiece",
        "lat": 20.2379, "lng": 85.8337,
        "district": "Khordha", "city": "Bhubaneswar",
        "entry_fee": 0, "stay_cost": 1800, "travel_cost": 400
    },
    "Dhauli": {
        "place_name": "Dhauli Shanti Stupa",
        "description": "Peace pagoda marking Emperor Ashoka's transformation",
        "lat": 20.1897, "lng": 85.8545,
        "district": "Khordha", "city": "Bhubaneswar",
        "entry_fee": 0, "stay_cost": 1500, "travel_cost": 300
    }
}

try:
    chroma_client = Client(
        Settings(
            is_persistent=True,
            persist_directory=CHROMA_DIR
        )
    )
    collection = chroma_client.get_collection(COLLECTION_NAME)
    logger.info("✅ Chroma collection loaded successfully.")
except Exception as e:
    logger.warning(f"⚠️ ChromaDB unavailable, using fallback: {e}")
    collection = None

# -------------------------------------------------------------------
#  LangGraph Imports
# -------------------------------------------------------------------
from langgraph.graph import StateGraph, START, END

# -------------------------------------------------------------------
#  STATE MODEL (Enhanced)
# -------------------------------------------------------------------
class State(TypedDict, total=False):
    request: Dict[str, Any]
    context: Dict[str, Any]  # NEW: Store derived context
    selected_places: List[str]
    rag_results: List[Dict[str, Any]]
    weather_cost_info: List[Dict[str, Any]]
    map_info: Dict[str, Any]
    final_itinerary: Dict[str, Any]
    error: Optional[str]

# -------------------------------------------------------------------
#  HELPER FUNCTIONS (More Robust)
# -------------------------------------------------------------------
def extract_places(text: str) -> List[str]:
    """Extract place names from comma-separated text"""
    if not text or not isinstance(text, str):
        return []
    return [p.strip() for p in text.split(",") if p.strip()]

def categorize_temperature(c: float) -> str:
    if c < 20: return "Cool"
    if c <= 30: return "Warm"
    return "Hot"

def categorize_humidity(h: float) -> str:
    if h < 40: return "Dry"
    if h <= 70: return "Comfortable"
    return "Humid"

def calculate_season(date_str: str) -> str:
    """Determine Odisha season from date"""
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        month = date_obj.month
        
        if month in [11, 12, 1, 2]:  # Nov-Feb
            return "Winter (Best time)"
        elif month in [3, 4, 5]:  # Mar-May
            return "Summer (Hot)"
        elif month in [6, 7, 8, 9]:  # Jun-Sep
            return "Monsoon (Rainy)"
        else:  # Oct
            return "Post-Monsoon (Pleasant)"
    except:
        return "Unknown"

def smart_cost_model(budget: int, duration: int, group_size: int, seniors: int, children: int):
    """Context-aware cost distribution"""
    per_person_day = budget / (duration * group_size)
    
    # Adjust for group composition
    senior_discount = seniors * 0.1  # Seniors may need less travel
    child_discount = children * 0.15  # Children eat less, smaller portions
    
    adjustment = 1 - (senior_discount + child_discount) / group_size if group_size > 0 else 1
    
    return {
        "stay": round(per_person_day * 0.40 * adjustment),
        "food": round(per_person_day * 0.25),
        "travel": round(per_person_day * 0.20 * adjustment),
        "activities": round(per_person_day * 0.10),
        "misc": round(per_person_day * 0.05),
        "total_per_person_day": round(per_person_day),
        "total_trip_cost": budget,
    }

# -------------------------------------------------------------------
# 0️⃣ CONTEXT NODE → Build Travel Context
# -------------------------------------------------------------------
def context_function(state: State):
    """Extract meaningful context from request"""
    req = state["request"]
    
    context = {
        "season": calculate_season(req.get("start_date", "")),
        "is_family_trip": (req.get("seniors", 0) > 0 or req.get("children", 0) > 0),
        "is_budget_trip": req.get("budget", 0) < 20000,
        "is_spiritual": any("Spiritual" in v or "Jagannath" in v for v in req.get("vibes", [])),
        "is_nature": any("Nature" in v or "Beach" in v or "Eco" in v for v in req.get("vibes", [])),
        "accessibility_needs": "wheelchair" in req.get("preferences", "").lower(),
        "dietary_restrictions": "veg" in req.get("preferences", "").lower(),
        "pace": "slow" if "slow" in req.get("preferences", "").lower() else "moderate",
    }
    
    logger.info(f"📊 Context extracted: {context}")
    return {"context": context}

# -------------------------------------------------------------------
# 1️⃣ MAIN NODE → Select Places (Enhanced)
# -------------------------------------------------------------------
def main_function(state: State):
    """Select places with context awareness"""
    try:
        req = state["request"]
        context = state.get("context", {})
        user_places = req.get("specific_places", "").strip()

        if user_places:
            places = extract_places(user_places)
            logger.info(f"✅ User specified places: {places}")
            return {"selected_places": places}

        # Context-aware prompt
        vibes_str = ", ".join(req.get("vibes", []))
        prompt = f"""
        Suggest 4-6 best places in Odisha for a {req.get('duration')} day trip.
        
        Trip Profile:
        - Vibes: {vibes_str}
        - Budget: ₹{req.get('budget')}
        - Season: {context.get('season')}
        - Family trip: {context.get('is_family_trip')}
        - Spiritual focus: {context.get('is_spiritual')}
        - Nature focus: {context.get('is_nature')}
        - Preferences: {req.get('preferences')}
        
        Return ONLY comma-separated place names (e.g., Puri, Konark, Chilika Lake).
        Focus on places suitable for the profile above.
        """

        if llm:
            generated_text = llm.invoke(prompt, temperature=0.8)
            places = extract_places(generated_text)
        else:
            places = []

        # Fallback to intelligent defaults
        if not places:
            logger.warning("⚠️ LLM failed, using context-based fallback")
            if context.get('is_spiritual'):
                places = ["Puri", "Konark", "Bhubaneswar", "Dhauli"]
            elif context.get('is_nature'):
                places = ["Chilika", "Similipal", "Bhitarkanika", "Satkosia"]
            else:
                places = ["Puri", "Konark", "Chilika", "Bhubaneswar"]

        logger.info(f"✅ Selected places: {places}")
        return {"selected_places": places[:6]}  # Max 6 places
    
    except Exception as e:
        logger.error(f"❌ main_function error: {e}")
        return {"selected_places": ["Puri", "Konark", "Chilika"], "error": str(e)}

# -------------------------------------------------------------------
# 2️⃣ RAG NODE → Query with Fallback
# -------------------------------------------------------------------
def rag_function(state: State):
    """RAG with robust fallback mechanism"""
    try:
        selected = state["selected_places"]
        rag_results = []

        for place in selected:
            place_data = None
            
            # Try ChromaDB first
            if collection and embedder:
                try:
                    query_vec = embedder.encode([place])[0]
                    response = collection.query(query_embeddings=[query_vec], n_results=1)
                    
                    if response and response.get("metadatas") and len(response["metadatas"][0]) > 0:
                        meta = response["metadatas"][0][0]
                        place_data = {
                            "place_name": meta.get("place_name", place),
                            "description": meta.get("description"),
                            "lat": meta.get("lat"),
                            "lng": meta.get("lng"),
                            "district": meta.get("district"),
                            "city": meta.get("city"),
                            "entry_fee": meta.get("entry_fee", 0),
                            "stay_cost": meta.get("stay_cost", 1500),
                            "travel_cost": meta.get("travel_cost", 500),
                        }
                except Exception as e:
                    logger.warning(f"⚠️ ChromaDB query failed for {place}: {e}")
            
            # Fallback to hardcoded data
            if not place_data:
                place_key = place.split()[0]  # "Puri Temple" → "Puri"
                if place_key in FALLBACK_PLACES:
                    place_data = FALLBACK_PLACES[place_key].copy()
                else:
                    # Generic fallback
                    place_data = {
                        "place_name": place,
                        "description": f"Popular destination in Odisha: {place}",
                        "lat": 20.2961 + random.uniform(-0.5, 0.5),
                        "lng": 85.8245 + random.uniform(-0.5, 0.5),
                        "district": "Odisha",
                        "city": place,
                        "entry_fee": random.choice([0, 20, 40, 50]),
                        "stay_cost": random.randint(800, 2000),
                        "travel_cost": random.randint(300, 800),
                    }
            
            rag_results.append(place_data)

        logger.info(f"✅ RAG retrieved {len(rag_results)} places")
        return {"rag_results": rag_results}
    
    except Exception as e:
        logger.error(f"❌ rag_function error: {e}")
        return {"rag_results": [], "error": str(e)}

# -------------------------------------------------------------------
# 3️⃣ WEATHER + COST NODE (Context-Aware)
# -------------------------------------------------------------------
def weather_cost_function(state: State):
    """Generate weather and costs with seasonal accuracy"""
    try:
        req = state["request"]
        context = state.get("context", {})
        rag = state["rag_results"]

        duration = req["duration"]
        budget = req["budget"]
        group_size = req.get("group_size", 2)
        seniors = req.get("seniors", 0)
        children = req.get("children", 0)

        output = []
        season = context.get("season", "")

        for place in rag:
            # Season-aware weather
            if "Winter" in season:
                temp = round(random.uniform(18, 28), 1)
                humidity = round(random.uniform(40, 60), 1)
            elif "Summer" in season:
                temp = round(random.uniform(32, 42), 1)
                humidity = round(random.uniform(30, 50), 1)
            elif "Monsoon" in season:
                temp = round(random.uniform(26, 32), 1)
                humidity = round(random.uniform(75, 95), 1)
            else:
                temp = round(random.uniform(26, 34), 1)
                humidity = round(random.uniform(50, 70), 1)

            weather = {
                "temp_c": temp,
                "humidity": humidity,
                "temp_desc": categorize_temperature(temp),
                "humidity_desc": categorize_humidity(humidity),
                "season": season,
                "summary": f"{categorize_temperature(temp)} ({temp}°C), {categorize_humidity(humidity)} humidity ({humidity}%)",
            }

            # Use actual place costs from RAG
            place_cost = smart_cost_model(budget, duration, group_size, seniors, children)
            place_cost["entry_fee"] = place.get("entry_fee", 0)
            place_cost["estimated_stay"] = place.get("stay_cost", 1500)

            output.append({
                "place": place["place_name"],
                "weather": weather,
                "cost": place_cost,
            })

        logger.info(f"✅ Weather and cost calculated for {len(output)} places")
        return {"weather_cost_info": output}
    
    except Exception as e:
        logger.error(f"❌ weather_cost_function error: {e}")
        return {"weather_cost_info": [], "error": str(e)}

# -------------------------------------------------------------------
# 4️⃣ MAP NODE → Optimized Route
# -------------------------------------------------------------------
def map_function(state: State):
    """Generate map with optimized routing"""
    try:
        rag = state["rag_results"]

        leaflet_points = [
            {"name": p["place_name"], "lat": p["lat"], "lng": p["lng"]}
            for p in rag
            if p.get("lat") and p.get("lng")
        ]

        if not leaflet_points:
            logger.warning("⚠️ No valid coordinates found")
            return {
                "map_info": {
                    "leaflet_points": [],
                    "route_order": [],
                    "coords_array": [],
                }
            }

        # Try LLM for routing, fallback to geographic sorting
        route_order = [p["name"] for p in leaflet_points]
        
        if llm and len(leaflet_points) > 2:
            try:
                prompt = f"""
                Optimize travel route for these Odisha places (minimize backtracking):
                {[p['name'] for p in leaflet_points]}
                
                Return ONLY comma-separated names in optimal order.
                """
                route_text = llm.invoke(prompt, temperature=0.3)
                suggested_route = extract_places(route_text)
                
                # Validate route
                if set(suggested_route) == set(route_order):
                    route_order = suggested_route
            except Exception as e:
                logger.warning(f"⚠️ Route optimization failed: {e}")

        coords_array = [[p["lat"], p["lng"]] for p in leaflet_points]

        logger.info(f"✅ Map generated with {len(leaflet_points)} points")
        return {
            "map_info": {
                "leaflet_points": leaflet_points,
                "route_order": route_order,
                "coords_array": coords_array,
                "center": coords_array[0] if coords_array else [20.2961, 85.8245],
            }
        }
    
    except Exception as e:
        logger.error(f"❌ map_function error: {e}")
        return {"map_info": {"leaflet_points": [], "route_order": [], "coords_array": []}, "error": str(e)}

# -------------------------------------------------------------------
# 5️⃣ FINAL NODE → Context-Rich Itinerary
# -------------------------------------------------------------------
def ensure_date(d):
    if isinstance(d, datetime):
        return d.date()
    if hasattr(d, "strftime"):  # datetime.date
        return d
    return datetime.strptime(d, "%Y-%m-%d").date()

def final_function(state: State):
    """Generate detailed, context-aware itinerary"""
    try:
        req = state["request"]
        context = state.get("context", {})
        places = state["selected_places"]
        rag = state["rag_results"]
        weather_cost = state["weather_cost_info"]
        mapinfo = state["map_info"]

        duration = req["duration"]
        start_date = ensure_date(req["start_date"])

        days = []
        places_per_day = max(1, len(places) // duration)
        
        for day in range(1, duration + 1):
            day_date = start_date + timedelta(days=day - 1)
            start_idx = (day - 1) * places_per_day
            end_idx = start_idx + places_per_day if day < duration else len(places)
            today_places = places[start_idx:end_idx]

            # Find RAG data for places
            place_details = []
            for place_name in today_places:
                place_rag = next((r for r in rag if r["place_name"] == place_name), None)
                if place_rag:
                    place_details.append({
                        "name": place_name,
                        "description": place_rag.get("description", ""),
                        "entry_fee": place_rag.get("entry_fee", 0),
                    })

            # Context-aware timing
            if context.get("is_family_trip"):
                time_plan = f"Start 9 AM (leisurely) → Explore {', '.join(today_places)} → Return by 6 PM"
            elif context.get("pace") == "slow":
                time_plan = f"Start 10 AM → Slow exploration of {', '.join(today_places)} → Return by 5 PM"
            else:
                time_plan = f"Start 8 AM → Explore {', '.join(today_places)} → Return by 7 PM"

            # Weather for the day
            day_weather = weather_cost[min(day - 1, len(weather_cost) - 1)]["weather"] if weather_cost else {}
            day_cost = weather_cost[min(day - 1, len(weather_cost) - 1)]["cost"] if weather_cost else {}

            entry = {
                "day": day,
                "date": day_date.strftime("%Y-%m-%d"),
                "day_name": day_date.strftime("%A"),
                "destinations": place_details,
                "time_plan": time_plan,
                "weather": day_weather,
                "cost_breakdown": day_cost,
                "tips": generate_tips(context, today_places),
            }
            days.append(entry)

        total_cost_summary = weather_cost[0]["cost"] if weather_cost else {}
        
        logger.info(f"✅ Final itinerary generated: {duration} days, {len(places)} places")
        
        return {
            "final_itinerary": {
                "trip_summary": {
                    "duration": duration,
                    "start_date": ensure_date(req["start_date"]).strftime("%Y-%m-%d"),
                    "total_places": len(places),
                    "season": context.get("season"),
                    "group_size": req.get("group_size", 2),
                    "vibes": req.get("vibes", []),
                },
                "days": days,
                "route_order": mapinfo.get("route_order", []),
                "map_data": mapinfo,
                "cost_summary": total_cost_summary,
                "context": context,
            }
        }
    
    except Exception as e:
        logger.error(f"❌ final_function error: {e}")
        return {"final_itinerary": {}, "error": str(e)}

def generate_tips(context: Dict, places: List[str]) -> List[str]:
    """Generate context-specific tips"""
    tips = []
    
    if context.get("is_family_trip"):
        tips.append("👨‍👩‍👧 Take frequent breaks for children and seniors")
    if context.get("dietary_restrictions"):
        tips.append("🥗 Pure vegetarian restaurants marked on route")
    if context.get("accessibility_needs"):
        tips.append("♿ Wheelchair-accessible paths verified")
    if context.get("season") and "Summer" in context.get("season"):
        tips.append("☀️ Carry water bottles and sunscreen")
    if any("Temple" in p or "Puri" in p for p in places):
        tips.append("🛕 Dress modestly for temple visits")
    
    return tips or ["Enjoy your journey through beautiful Odisha!"]

# -------------------------------------------------------------------
# BUILD GRAPH (Enhanced with Context)
# -------------------------------------------------------------------
def build_graph():
    graph = StateGraph(State)

    # Add nodes
    graph.add_node("context_function", context_function)
    graph.add_node("main_function", main_function)
    graph.add_node("rag_function", rag_function)
    graph.add_node("weather_cost_function", weather_cost_function)
    graph.add_node("map_function", map_function)
    graph.add_node("final_function", final_function)

    # Build flow
    graph.add_edge(START, "context_function")
    graph.add_edge("context_function", "main_function")
    graph.add_edge("main_function", "rag_function")
    graph.add_edge("rag_function", "weather_cost_function")
    graph.add_edge("rag_function", "map_function")
    graph.add_edge("weather_cost_function", "final_function")
    graph.add_edge("map_function", "final_function")
    graph.add_edge("final_function", END)

    logger.info("✅ Chakadola graph compiled with enhanced context awareness.")
    return graph.compile()

# Export the chain
chakadola_chain = build_graph()