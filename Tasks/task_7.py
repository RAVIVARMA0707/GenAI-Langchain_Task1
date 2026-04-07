from langchain_text_splitters import RecursiveCharacterTextSplitter
SAMPLE_DOCUMENT = """
LangChain is a framework for developing applications powered by language models.
It provides tools for prompt management, chains, agents, and memory.
LangChain integrates with many LLM providers including OpenAI, Anthropic, and Cohere.
The framework also supports vector stores, document loaders, and output parsers.
RAG (Retrieval-Augmented Generation) is a technique that enhances LLM responses
by fetching relevant documents from a knowledge base at query time.
pgvector is a PostgreSQL extension that enables efficient storage and similarity
search of high-dimensional vector embeddings directly inside a relational database.
LangSmith is an observability platform for LangChain applications that provides
tracing, evaluation, and debugging of LLM pipelines.
"""

def batch_embed_with_chunks(text: str, chunk_size: int, overlap: int) -> dict:
    """Splits text into chunks, embeds them, and returns metadata."""
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = splitter.split_text(text)
    
    mock_dim = 1536 
    
    return {
        "num_chunks": len(chunks),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "embedding_dim": mock_dim,
        "chunks": chunks
    }

result = batch_embed_with_chunks(SAMPLE_DOCUMENT, 200, 40)

print(f"Split into {result['num_chunks']} chunks.")
print(f"First chunk snippet: {result['chunks']}")