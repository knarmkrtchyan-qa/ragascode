import requests
from config import GENERATION_ENDPOINT

def generate_answer(question, contexts):
    """
    Generates an answer from Gemini based on the given question and top-k contexts.
    """

    # Combine the context entries into a single string (only answers)
    context_text = "\n".join([entry["answer"] for entry in contexts])

    #  Improved prompt for natural and grounded answers
    prompt = f"""
You are a precise and factual assistant. 
Answer strictly using ONLY the information provided in the contexts below. 

⚠️ Rules:
- Copy important terms and definitions exactly as they appear in the context. 
- Do NOT replace key phrases with synonyms. 
- Do NOT add extra explanations, reasoning, or filler words. 
- If the answer is not explicitly in the context, reply: "Sorry, I don't know the answer."

Question: {question}

Contexts:
{context_text}

Now provide the answer using the exact wording from the context where possible:
"""

    # Prepare payload for Gemini API
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    # Send request to Gemini API
    response = requests.post(
        GENERATION_ENDPOINT,
        headers={"Content-Type": "application/json"},
        json=payload
    )

    if response.status_code != 200:
        raise Exception(f"Gemini generation error: {response.status_code}, {response.text}")

    # Extract and return the generated answer text
    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]
