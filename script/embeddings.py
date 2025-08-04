from sentence_transformers import SentenceTransformer
import torch
import os

def load_embedding_model():
    """Load optimized embedding model with faster loading"""
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'all-mpnet-base-v2')
    
    # Determine optimal device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model with optimizations
    model = SentenceTransformer(
        model_path,
        device=device,
        # Use half precision for faster inference and less memory
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32
    )
    
    # Optimize for inference
    model.eval()
    
    return model

def generate_embeddings(text, model):
    """Generate embeddings with mpnet - optimized for speed"""
    with torch.no_grad():  # Disable gradient computation for inference
        return model.encode(
            text,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False  # Disable progress bar for faster processing
        )