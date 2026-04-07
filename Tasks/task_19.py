from langsmith import Client
from dotenv import load_dotenv

load_dotenv(override=True)

def create_langsmith_dataset() -> str:
    """Creates a LangSmith dataset with 3 examples. Returns dataset id."""
    # ── YOUR CODE BELOW ──────────────────────────────────────
    
    client = Client()

    # Define the data to be uploaded
    questions = [
        "What does RAG stand for?",
        "What PostgreSQL extension enables vector search?",
        "What LangChain tool provides observability?"
    ]
    answers = [
        "Retrieval-Augmented Generation",
        "pgvector",
        "LangSmith"
    ]

    # 1. Create the dataset
    dataset = client.create_dataset(
        dataset_name="rag-eval-dataset", 
        description="Dataset for RAG evaluation questions"
    )

    # 2. Add the 3 question-answer example pairs
    client.create_examples(
        inputs=[{"question": q} for q in questions],
        outputs=[{"answer": a} for a in answers],
        dataset_id=dataset.id
    )

    # 3. Return the dataset id as a string
    return str(dataset.id)

print("\n[Task 19] Create LangSmith Dataset")
dataset_id = create_langsmith_dataset()
print(f"  Dataset ID: {dataset_id}")

    # ── END OF YOUR CODE ─────────────────────────────────────