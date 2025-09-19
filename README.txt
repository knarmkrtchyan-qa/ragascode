📖 README
Gemini RAG Chatbot with RAGAS Evaluation
This project implements a Retrieval-Augmented Generation (RAG) chatbot that uses Google Gemini for embeddings and answer generation, and integrates RAGAS for evaluation.
It allows you to ask domain-specific questions (e.g., about Spark.work HR system) and evaluates responses for precision, recall, faithfulness, and relevancy.
________________


🚀 Features
Data Preparation

   * prepare_dataset.py → Scrapes content and generates Q&A pairs with Gemini.
   * Stores Q&A in dataset.json.

 Embeddings

   * embeddings.py → Creates or loads embeddings using Gemini embedding API.
   * Cached in dataset_with_embeddings.json.

Retrieval

   * retrieval.py → Retrieves top-k relevant contexts using cosine similarity.

Answer Generation

   * generation.py → Calls Gemini generation API with strict grounding rules.
   * Ensures answers only use retrieved context.

Pipeline

   * pipeline.py → RAGPipeline class to connect retrieval + generation.

Evaluation

   * evaluation.py → Custom metrics for retrieval and answer correctness.
   * chatbot.py → Interactive chatbot with RAGAS evaluation.

________________


📂 Project Structure

├── chatbot.py                 # Main chatbot interface (Rich TUI + RAGAS)
├── config.py                  # API keys, model endpoints, file paths
├── dataset.json               # Clean Q&A dataset
├── dataset_with_embeddings.json # Cached dataset with embeddings
├── embeddings.py              # Embedding creation & caching
├── evaluation.py              # Custom evaluation metrics
├── generation.py              # Answer generation using Gemini
├── pipeline.py                # RAGPipeline class
├── prepare_dataset.py         # Web scraping + Q&A generation
├── retrieval.py               # Context retrieval (cosine similarity)


________________


🔑 Configuration
Update config.py with your API keys and model settings:
API_KEY = "YOUR_GEMINI_API_KEY"
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"   # optional, for RAGAS
EMBEDDING_MODEL = "embedding-001"
GENERATION_MODEL = "gemini-1.5-flash-8b"


________________


⚙️ Installation
   1. Clone repository & install dependencies:

git clone <your-repo>
cd <your-repo>
pip install -r requirements.txt


Dependencies include:
   * requests

   * numpy

   * rich

   * datasets

   * ragas

   * selenium

   * pandas


  2. Make sure Chrome + ChromeDriver are installed if you plan to run prepare_dataset.py.

________________


▶️ Usage
1. Prepare Dataset
python prepare_dataset.py

This scrapes articles and generates Q&A pairs into dataset.json.

2. Build Embeddings
python embeddings.py

Embeds dataset and saves into dataset_with_embeddings.json.

3. Run Chatbot
python chatbot.py


You’ll see:
Welcome to your Gemini RAGA Chatbot!
Type 'exit' to quit.


Ask questions, and the chatbot will:

   * Retrieve relevant contexts
   * Generate answers with Gemini
   * Evaluate using RAGAS
   * Display results in a Rich table

________________


🧪 Example Evaluation Metrics
For each question, the chatbot logs metrics such as:

   * context_precision
   * context_recall
   * faithfulness
   * answer_relevancy
   * answer_correctness

All results are saved to ragas_eval_log.csv.
________________


📊 Custom Evaluation
Use evaluation.py directly for:

   * Retrieval score analysis (evaluate_retrieval)
   * Ground truth checks (evaluate_with_ground_truth)

________________


⚠️ Notes
   * Ensure API keys are valid in config.py.
   * Large datasets may take time to embed.
   * If cache (dataset_with_embeddings.json) is corrupted, embeddings will rebuild.

________________

