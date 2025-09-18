import json
import os
import requests
import numpy as np
from config import EMBEDDING_ENDPOINT, EMBEDDING_MODEL, DATASET_FILE, CACHE_FILE


def create_embedding(text: str):
    payload = {
        "model": EMBEDDING_MODEL,
        "content": {"parts": [{"text": text}]}
    }
    response = requests.post(
        EMBEDDING_ENDPOINT,
        headers={"Content-Type": "application/json"},
        json=payload
    )
    if response.status_code != 200:
        raise Exception(f"Gemini API error: {response.status_code}, {response.text}")
    return response.json()["embedding"]["values"]


def chunk_text(text: str, max_chars=1500):
    """Split text into smaller chunks to stay within API limits."""
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]


def embed_text_with_chunking(text: str):
    """Embed long text by splitting into chunks and averaging embeddings."""
    chunks = chunk_text(text)
    vectors = []
    for chunk in chunks:
        try:
            vec = create_embedding(chunk)
            vectors.append(vec)
        except Exception as e:
            print(f"⚠️ Failed to embed chunk: {e}")
    if not vectors:
        return []
    return np.mean(np.array(vectors), axis=0).tolist()


def load_dataset():
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_or_create_embeddings():
    """
    Load dataset with embeddings from cache, or create them if missing/invalid.
    Ensures dataset_with_embeddings.json is never left empty or corrupted.
    """
    if os.path.exists(CACHE_FILE) and os.path.getsize(CACHE_FILE) > 0:
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("⚠️ Cache file is corrupted. Rebuilding embeddings...")

    print("🔄 Building embeddings from dataset...")
    dataset = load_dataset()
    for i, entry in enumerate(dataset):
        try:
            context_text = " ".join(entry["contexts"])
            entry["embedding"] = embed_text_with_chunking(context_text)
            print(f"✅ Embedded entry {i+1}/{len(dataset)}")
        except Exception as e:
            print(f"⚠️ Failed to embed entry {i}: {e}")
            entry["embedding"] = []

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved dataset with embeddings to {CACHE_FILE}")
    return dataset
