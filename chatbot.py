from rich import print
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from embeddings import load_or_create_embeddings
from pipeline import RAGPipeline
import os
from config import OPENAI_API_KEY
import pandas as pd  # ✅ required for DataFrame handling

# ⚠️ If you want to use RAGAS with OpenAI, set OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- NEW: RAGAS imports ---
from datasets import Dataset
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    AnswerRelevancy,
    AnswerCorrectness,
)
from ragas import evaluate


# ----------------------------
# Load dataset with embeddings
# ----------------------------
print("[bold cyan]Loading dataset with embeddings...[/bold cyan]")
dataset = load_or_create_embeddings()
pipeline = RAGPipeline(dataset)

# ----------------------------
# Start Chat Interface
# ----------------------------
print("\n[bold green]Welcome to your Gemini RAGA Chatbot![/bold green]")
print("[dim]Type 'exit' to quit.[/dim]\n")

while True:
    user_input = Prompt.ask("[bold blue]Your question[/bold blue]")

    if user_input.strip().lower() == "exit":
        print("[bold yellow]Goodbye![/bold yellow]")
        break

    # Run the pipeline
    result = pipeline.run(user_input, top_k=3)

    # Display the generated answer
    print(Panel(f"[white]{result['answer']}[/white]", title="[bold green]Answer[/bold green]"))

    # ----------------------------
    # NEW EVALUATION WITH RAGAS
    # ----------------------------
    eval_data = {
        "question": [result["question"]],
        "contexts": [[" ".join(ctx.get("contexts", [])) for ctx in result["contexts"]]],
        "answer": [result["answer"]],
    }

   # Add reference (ground truth) if available
    reference = ""
    for ctx in result["contexts"]:
        if "ground_truth" in ctx and ctx["ground_truth"].strip():
            reference = ctx["ground_truth"]
        break   # take the first valid ground truth
    eval_data["reference"] = [reference]

    # Convert to HuggingFace Dataset
    ragas_dataset = Dataset.from_dict(eval_data)

    # Run RAGAS evaluation
    ragas_results = evaluate(
        ragas_dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            AnswerRelevancy(),
            AnswerCorrectness(),
        ],
    )

    # Convert results to Pandas DataFrame
    results_df = ragas_results.to_pandas()

    # Save results for debugging
    results_df.to_csv("ragas_eval_log.csv", index=False)

    # ----------------------------
    # Display clean Rich table
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
