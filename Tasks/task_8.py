from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv(override=True)

def compare_embedding_models(sentence: str) -> dict:
    """Embeds a sentence with two models and compares their dimensions."""
    
    model_a_name = "text-embedding-3-small"
    embeddings_a = OpenAIEmbeddings(model=model_a_name)
    vector_a = embeddings_a.embed_query(sentence)
    
    model_b_name = "text-embedding-3-large"
    embeddings_b = OpenAIEmbeddings(model=model_b_name)
    vector_b = embeddings_b.embed_query(sentence)
    
    dims_a = len(vector_a)
    dims_b = len(vector_b)
    dim_ratio = dims_b / dims_a
    
    return {
        "sentence": sentence,
        "model_a": {
            "model": model_a_name,
            "dims": dims_a,
            "first_3": vector_a[:3]
        },
        "model_b": {
            "model": model_b_name,
            "dims": dims_b,
            "first_3": vector_b[:3]
        },
        "dim_ratio": dim_ratio
    }

result = compare_embedding_models("Vector databases power semantic search.")
print(result)