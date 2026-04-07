from langsmith.evaluation import evaluate
from langsmith.schemas import Example, Run
from task_14 import basic_rag_pipeline

RAG_DOCUMENTS = [
    "LangChain v0.2 introduced LangChain Expression Language (LCEL) for composing chains.",
    "pgvector is a PostgreSQL extension supporting L2, inner product, and cosine distance.",
    "LangSmith provides tracing for every LLM call including token counts and latency.",
    "RAG stands for Retrieval-Augmented Generation and improves factual accuracy of LLMs.",
    "OpenAI's text-embedding-3-small produces 1536-dimensional embedding vectors.",
    "LangChain agents use a ReAct loop: Thought → Action → Observation → Answer.",
]

def run_langsmith_evaluation() -> dict:
    """Evaluates the RAG pipeline on the LangSmith dataset."""
    # ── YOUR CODE BELOW ──────────────────────────────────────
    
    dataset_name = "rag-eval-dataset"

    # 1. Define the target function
    # It must take a dict and return a dict mapping to your RAG output
    def target(inputs: dict) -> dict:
        answer = basic_rag_pipeline(RAG_DOCUMENTS, inputs["question"])
        return {"answer": answer}

    # 2. Define a custom evaluator
    # This checks if the 'reference' answer exists within the 'prediction'
    def contains_correct_answer(run: Run, example: Example) -> dict:
        prediction = run.outputs.get("answer", "").lower()
        reference = example.outputs.get("answer", "").lower()
        score = 1 if reference in prediction else 0
        return {"key": "answer_contains_expected", "score": score}

    # 3. Run the evaluation
    results = evaluate(
        target,
        data=dataset_name,
        evaluators=[contains_correct_answer],
        experiment_prefix="rag-challenge-eval",
    )

    # 4. Extract and return the summary
    # results is an ExperimentResults object; we aggregate the scores here
    num_examples = len(results)
    # Calculate pass rate based on our custom evaluator key
    pass_count = sum(1 for r in results if r["evaluation"]["answer_contains_expected"] == 1)
    pass_rate = pass_count / num_examples if num_examples > 0 else 0.0

    return {
        "dataset": dataset_name,
        "num_examples": num_examples,
        "pass_rate": pass_rate
    }

    # ── END OF YOUR CODE ─────────────────────────────────────

print("\n[Task 20] Run LangSmith Evaluation")
eval_summary = run_langsmith_evaluation()
print(f"  Dataset     : {eval_summary.get('dataset')}")
print(f"  # Examples  : {eval_summary.get('num_examples')}")
print(f"  Pass rate   : {eval_summary.get('pass_rate')}")