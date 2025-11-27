import json
import uuid
import os
import shutil
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# New Chroma 0.5+ API
from chromadb import Client
from chromadb.config import Settings

# ------------------------------
# CONFIG
# ------------------------------
DATA_FILE = "odisha_data.json"
CHROMA_DIR = "./odisha_chroma"
COLLECTION_NAME = "odisha_tourism"
TARGET_DIM = 768  # all-mpnet-base-v2


print("\n🔥 Odisha Tourism Loader (Auto-Fix Version)")
print("============================================")


# ------------------------------
# Step 0 — FULL DELETE old DB folder (auto-clean)
# ------------------------------
if os.path.exists(CHROMA_DIR):
    print(f"🧹 Removing old ChromaDB folder: {CHROMA_DIR}")
    shutil.rmtree(CHROMA_DIR, ignore_errors=True)

print("✅ Old Chroma folder removed (fresh rebuild will happen)")


# ------------------------------
# Step 1 — Load JSON
# ------------------------------
print("📁 Loading Odisha Tourism JSON...")

try:
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
except Exception as e:
    print(f"❌ Failed to load JSON file: {e}")
    raise

print("✅ JSON Loaded")


# ------------------------------
# Step 2 — Init NEW Chroma Client
# ------------------------------
print("🧠 Initializing NEW ChromaDB (v0.5+) ...")

client = Client(
    Settings(
        is_persistent=True,
        persist_directory=CHROMA_DIR
    )
)

# Always create a clean collection
try:
    client.delete_collection(COLLECTION_NAME)
except:
    pass

collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

print(f"✅ New Chroma collection `{COLLECTION_NAME}` created")


# ------------------------------
# Step 3 — Load Embedder (all-mpnet-base-v2)
# ------------------------------
print("🧠 Loading embedding model all-mpnet-base-v2 (dim=768)...")
embedder = SentenceTransformer("all-mpnet-base-v2")
print("✅ Embedder ready!")


# ------------------------------
# Step 4 — Flatten JSON → Documents
# ------------------------------
def flatten(data_json):
    docs = []

    for dist in data_json["districts"]:
        district = dist["district_name"]

        for city in dist["cities"]:
            city_name = city["city_name"]

            attributes = city.get("attributes", {})
            costs = city.get("approx_costs_inr", {})

            for place in city["places"]:
                meta = {
                    "district": district,
                    "city": city_name,
                    "place_name": place["name"],
                    "description": place["description"],
                    "lat": place["coords"]["lat"],
                    "lng": place["coords"]["lng"],
                    "food": attributes.get("food"),
                    "art_culture": attributes.get("art_culture"),
                    "transport": attributes.get("transport"),
                    "stay_cost": costs.get("stay_per_night"),
                    "travel_cost": costs.get("local_travel_daily"),
                    "entry_fee": costs.get("entry_fees_avg"),
                }

                text = (
                    f"{district}. {city_name}. {place['name']}. "
                    f"{place['description']}. "
                    f"Food: {attributes.get('food')}. "
                    f"Culture: {attributes.get('art_culture')}. "
                    f"Transport: {attributes.get('transport')}."
                )

                docs.append((str(uuid.uuid4()), text, meta))

    return docs


print("🔄 Flattening Odisha JSON...")
docs = flatten(data)
print(f"📌 Total places extracted: {len(docs)}")


# ------------------------------
# Step 5 — Compute Embeddings + Insert into Chroma
# ------------------------------
print("🧠 Generating embeddings (dim=768) and inserting into Chroma...")

ids, texts, metas = [], [], []

for uid, txt, meta in tqdm(docs):
    ids.append(uid)
    texts.append(txt)
    metas.append(meta)

# Embed in batches
embeddings = embedder.encode(texts, batch_size=32, show_progress_bar=True)

print(f"🔍 Embedding dimension detected: {len(embeddings[0])}")
if len(embeddings[0]) != TARGET_DIM:
    raise RuntimeError(
        f"❌ ERROR: Embedding dim mismatch! Expected {TARGET_DIM}, got {len(embeddings[0])}"
    )

# Insert into Chroma
collection.add(
    ids=ids,
    documents=texts,
    embeddings=embeddings,
    metadatas=metas
)

print("\n🎉 Odisha Tourism Vector DB Created Successfully!")
print(f"📍 Saved at: {CHROMA_DIR}")
print("============================================\n")
