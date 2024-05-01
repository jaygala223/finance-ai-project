import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def find_closest_chunk(prompt_embedding, chunk_embeddings):
    """Find the closest chunk embeddings to the prompt embedding.

       returns closest_chunk_idx, closest_chunk_similarity, closest_chunk_embedding
    
    """
    similarities = cosine_similarity([prompt_embedding], chunk_embeddings)[0]
    closest_chunk_index = np.argmax(similarities)
    closest_chunk_similarity = similarities[closest_chunk_index]
    closest_chunk_embedding = chunk_embeddings[closest_chunk_index]
    return closest_chunk_index, closest_chunk_similarity, closest_chunk_embedding

