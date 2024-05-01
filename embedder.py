from sentence_transformers import SentenceTransformer

def create_chunk_embeddings(chunks):

    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # embedding_model.to(device)

    embeddings = embedding_model.encode(chunks)
    
    return embeddings


def create_prompt_embeddings(prompt: str):
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # embedding_model.to(device)

    embeddings = embedding_model.encode(prompt)
    
    return embeddings