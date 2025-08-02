from sentence_transformers import SentenceTransformer
import torch

def load_embedding_model():
    
    return SentenceTransformer(
        'U:\\LEGALSENSE\\models\\all-mpnet-base-v2',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

def generate_embeddings(text, model):
    """Generate embeddings with mpnet"""
    return model.encode(text)