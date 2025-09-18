import numpy as np
from embeddings import create_embedding

def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def retrieve_top_k(question, dataset, k=3):
    q_emb = create_embedding(question)
    sims = [(entry, cosine_similarity(q_emb, entry["embedding"])) for entry in dataset]
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:k]
