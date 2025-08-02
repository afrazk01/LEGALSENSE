import numpy as np
import pandas as pd

def find_relevant_chunks(query_embedding, df, top_k=2):
    """Find top legal provisions using mpnet embeddings"""
    chunk_embeds = np.stack(df['embedding'].apply(
        lambda x: np.array(eval(x)) if isinstance(x, str) else x
    ).values)
    
    # Cosine similarity
    similarities = np.dot(chunk_embeds, query_embedding) / (
        np.linalg.norm(chunk_embeds, axis=1) * np.linalg.norm(query_embedding)
    )
    
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return df.iloc[top_indices]