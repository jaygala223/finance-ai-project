def chunk_text(text, chunk_size=512):
    """Chunk the text into smaller pieces."""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks
