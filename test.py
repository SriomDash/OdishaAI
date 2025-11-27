# test_chakadola.py

from chakadola_graph import chakadola_chain

def pretty(title, data):
    print("\n" + "="*80)
    print(title)
    print("="*80)
    print(data)

# -----------------------------
#  SAMPLE REQUEST PAYLOAD
# -----------------------------

test_request = {
    "group_size": 4,
    "duration": 3,
    "start_date": "2025-02-14",
    "budget": 15000,
    "specific_places": "Puri, Konark, Chilika Lake",
    "vibes": ["Spiritual", "Nature"],
    "preferences": "Pure veg, slow travel"
}

# -----------------------------
#  RUN THE WORKFLOW
# -----------------------------
print("\n🚀 Running Chakadola Itinerary Pipeline...\n")

result = chakadola_chain.invoke({"request": test_request})

# -----------------------------
#  PRINT RESULTS
# -----------------------------

if "selected_places" in result:
    pretty("SELECTED PLACES", result["selected_places"])

if "rag_results" in result:
    pretty("RAG RESULTS", result["rag_results"])

if "weather_cost_info" in result:
    pretty("WEATHER + COST INFO", result["weather_cost_info"])

if "map_info" in result:
    pretty("MAP INFO (LEAFLET READY)", result["map_info"])

if "final_itinerary" in result:
    pretty("FINAL ITINERARY (DAY-WISE)", result["final_itinerary"])


print("\n🎉 TEST COMPLETED SUCCESSFULLY!\n")
