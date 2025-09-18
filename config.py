API_KEY = "AIzaSyCuG2JKE1Vr_TXhCbDu4o0sVdwfje_xgGo"  # replace with your key
OPENAI_API_KEY = "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"


# Models
EMBEDDING_MODEL = "embedding-001"   
GENERATION_MODEL = "gemini-1.5-flash-8b"   
# Endpoints
EMBEDDING_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBEDDING_MODEL}:embedContent?key={API_KEY}"
GENERATION_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GENERATION_MODEL}:generateContent?key={API_KEY}"

# Files
DATASET_FILE = "dataset.json"
CACHE_FILE = "dataset_with_embeddings.json"

