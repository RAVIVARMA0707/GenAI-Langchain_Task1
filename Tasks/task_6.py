import numpy as np

def cosine_similarity_manual(v1: list, v2: list) -> float:
    """
    Compute the cosine similarity between two vectors using manual calculations.

    """
    dot_product = sum(a * b for a, b in zip(v1, v2))
    
    mag1 = sum(x**2 for x in v1) ** 0.5
    mag2 = sum(x**2 for x in v2) ** 0.5
    
    if not mag1 or not mag2:
        return 0.0
        
    return dot_product / (mag1 * mag2)


def cosine_similarity_numpy(v1: list, v2: list) -> float:
    """
    Compute the cosine similarity between two vectors using numpy.

    """
    v1_arr, v2_arr = np.array(v1), np.array(v2)
    
    dot_product = np.dot(v1_arr, v2_arr)
    norm_product = np.linalg.norm(v1_arr) * np.linalg.norm(v2_arr)
    
    if norm_product == 0:
        return 0.0
        
    return float(dot_product / norm_product)


def compare_word_pairs() -> dict:
    """
    Compare the similarity of the following word pairs:
    - dog vs puppy
    - dog vs automobile
    Return the most similar pair.
    """
    
    embeddings = {
        "dog": [0.9, 0.1, 0.0, 0.05],
        "puppy": [0.85, 0.15, 0.02, 0.01],
        "automobile": [0.01, 0.05, 0.8, 0.9]
    }
    
    sim_dog_puppy = cosine_similarity_manual(embeddings["dog"], embeddings["puppy"])
    sim_dog_auto = cosine_similarity_manual(embeddings["dog"], embeddings["automobile"])
    
    more_similar = "Pair 1: dog vs puppy" if sim_dog_puppy > sim_dog_auto else "Pair 2: dog vs automobile"
    
    return {
        "dog_vs_puppy": round(sim_dog_puppy, 4),
        "dog_vs_automobile": round(sim_dog_auto, 4),
        "more_similar_pair": more_similar
    }

if __name__ == "__main__":
    results = compare_word_pairs()
    print(f"Dog vs Puppy similarity: {results['dog_vs_puppy']}")
    print(f"Dog vs Automobile similarity: {results['dog_vs_automobile']}")
    print(f"Winner: {results['more_similar_pair']}")