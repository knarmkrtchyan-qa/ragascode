import re

def evaluate_retrieval(scores, threshold=0.6):
    """
    Evaluate retrieval quality based on similarity scores.
    Returns:
        - Number of relevant contexts
        - Total retrieved
        - Average score
        - Max score
        - Score spread (max - min)
    """
    relevant = [s for s in scores if s >= threshold]
    total = len(scores)
    avg_score = sum(scores) / total if total else 0
    max_score = max(scores) if scores else 0
    spread = max_score - min(scores) if scores else 0

    return {
        "relevant_count": len(relevant),
        "total": total,
        "avg_score": avg_score,
        "max_score": max_score,
        "spread": spread,
    }


def normalize_text(text: str) -> str:
    """Lowercase and normalize text for fair comparison."""
    text = text.lower()
    # Replace hyphens and underscores with spaces
    text = text.replace("-", " ").replace("_", " ")
    # Remove punctuation except spaces
    text = re.sub(r"[^a-z0-9\s]", "", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text



def evaluate_with_ground_truth(question, retrieved_contexts, generated_answer, ground_truth):
    """
    Extended RAG evaluation with ground truth checks.

    Args:
        question (str): User question
        retrieved_contexts (list[dict]): Retrieved dataset entries
        generated_answer (str): Final LLM answer
        ground_truth (str): Expected key fact from dataset

    Returns:
        dict with:
            - context_contains_gt: % of contexts that include ground truth
            - answer_correct: does generated answer contain ground truth
            - faithful: does answer only use retrieved info
    """
    norm_gt = normalize_text(ground_truth)

    # --- Context Coverage ---
    retrieved_texts = " ".join([" ".join(entry.get("contexts", [])) for entry in retrieved_contexts])
    context_contains_gt = norm_gt in normalize_text(retrieved_texts)

    # --- Answer Correctness ---
    answer_contains_gt = norm_gt in normalize_text(generated_answer)

    # --- Faithfulness (rough check) ---
    # If the answer contains words not in contexts, it's suspicious
    retrieved_words = set(normalize_text(retrieved_texts).split())
    answer_words = set(normalize_text(generated_answer).split())
    hallucinated_words = answer_words - retrieved_words
    faithful = len(hallucinated_words) < 5  # allow a few function words

    return {
        "context_contains_gt": context_contains_gt,
        "answer_correct": answer_contains_gt,
        "faithful": faithful,
        "hallucinated_words": list(hallucinated_words)[:10],  # preview only
    }
