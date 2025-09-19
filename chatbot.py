# ----------------------------
# Rich library for pretty printing
# ----------------------------
from rich import print
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

# ----------------------------
# Project-specific imports
# ----------------------------
from embeddings import load_or_create_embeddings  # function to load or generate text embeddings
from pipeline import RAGPipeline  # custom RAG pipeline for querying with context
import os
import pandas as pd  # required for handling evaluation results as DataFrame

# ----------------------------
# Gemini imports
# ----------------------------
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas.llms import LangchainLLMWrapper  # wrapper so RAGAS can use Gemini

# ----------------------------
# RAGAS evaluation imports
# ----------------------------
from datasets import Dataset  # HuggingFace Dataset format required by RAGAS
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    AnswerRelevancy,
    AnswerCorrectness,
)
from ragas import evaluate  # main function to run RAGAS evaluation

# ----------------------------
# Setup Gemini API Key
# ----------------------------
from config import API_KEY  
os.environ["GOOGLE_API_KEY"] = API_KEY  

# ----------------------------
# Load dataset with embeddings
# ----------------------------
dataset = load_or_create_embeddings()  # returns dataset containing text and embeddings
pipeline = RAGPipeline(dataset)        # initializes retrieval-augmented generation pipeline

# ----------------------------
# Setup Gemini as evaluator
# ----------------------------
gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
wrapped_llm = LangchainLLMWrapper(gemini_llm)

# Setup Gemini embeddings for RAGAS
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Attach Gemini LLM to built-in metrics
context_precision.llm = wrapped_llm
context_recall.llm = wrapped_llm
faithfulness.llm = wrapped_llm

# ----------------------------
# Start interactive chat
# ----------------------------
print("\n[bold green]Welcome to your Gemini RAG Chatbot![/bold green]")
print("[dim]Type 'exit' to quit.[/dim]\n")

while True:
    # Prompt user for input
    user_input = Prompt.ask("[bold blue]Your question[/bold blue]")

    if user_input.strip().lower() == "exit":
        print("[bold yellow]Goodbye![/bold yellow]")
        break

    # ----------------------------
    # Run RAG pipeline
    # ----------------------------
    result = pipeline.run(user_input, top_k=3)

    # Display generated answer in a Rich panel
    print(Panel(f"[white]{result['answer']}[/white]", title="[bold green]Answer[/bold green]"))

    # ----------------------------
    # Prepare data for RAGAS evaluation
    # ----------------------------
    eval_data = {
        "question": [result["question"]],
        "contexts": [[" ".join(ctx.get("contexts", [])) for ctx in result["contexts"]]],
        "answer": [result["answer"]],
    }

    # Optionally add reference (ground truth) if available
    reference = ""
    for ctx in result["contexts"]:
        if "ground_truth" in ctx and ctx["ground_truth"].strip():
            reference = ctx["ground_truth"]
            break  # take the first valid ground truth
    eval_data["reference"] = [reference]

    # Convert dictionary to HuggingFace Dataset for RAGAS
    ragas_dataset = Dataset.from_dict(eval_data)

    # ----------------------------
    # Run automatic evaluation using RAGAS with Gemini
    # ----------------------------
    answer_relevancy = AnswerRelevancy(llm=wrapped_llm)
    answer_correctness = AnswerCorrectness(llm=wrapped_llm)

    ragas_results = evaluate(
        ragas_dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
            answer_correctness,
        ],
        embeddings=gemini_embeddings,   # prevent OpenAI fallback
    )

    # Convert evaluation results to Pandas DataFrame
    results_df = ragas_results.to_pandas()

    # Save evaluation logs for debugging or analysis
    results_df.to_csv("ragas_eval_log.csv", index=False)

    # ----------------------------
    # Display evaluation legend
    # ----------------------------
    print("\n[bold cyan]Note: 1.000 means fully correct/relevant; 0 means incorrect/irrelevant.[/bold cyan]\n")

    # ----------------------------
    # Display evaluation results in a clean Rich table
    # ----------------------------
    ragas_table = Table(
        title=f"RAGAS Evaluation for: {user_input}",
        show_header=True,
        header_style="bold magenta",
    )
    ragas_table.add_column("Metric", justify="left", style="cyan")
    ragas_table.add_column("Value", justify="right", style="white")

    for metric in ["context_precision", "context_recall", "faithfulness", "answer_relevancy", "answer_correctness"]:
        if metric in results_df.columns:
            value = results_df.iloc[0][metric]
            ragas_table.add_row(metric, f"{value:.3f}" if not pd.isna(value) else "—")

    print(ragas_table)
