from retrieval import retrieve_top_k
from generation import generate_answer
from embeddings import load_or_create_embeddings

class RAGPipeline:
    def __init__(self, dataset):
        self.dataset = dataset

    def run(self, question, top_k=None):
        # if no top_k is provided, use full dataset length
        if top_k is None:
            top_k = len(self.dataset)

        top_contexts_with_scores = retrieve_top_k(question, self.dataset, k=top_k)
        top_contexts = [entry for entry, _ in top_contexts_with_scores]
        answer = generate_answer(question, top_contexts)
        return {
            "question": question,
            "contexts": top_contexts,
            "answer": answer,
            "scores": [score for _, score in top_contexts_with_scores],
        }
