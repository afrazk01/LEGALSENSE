"""
Model optimization utilities for faster loading and inference
"""
import torch
import os
from typing import Optional
import gc

def optimize_torch_settings():
    """Optimize PyTorch settings for faster model loading"""
    try:
        # Enable memory efficient attention if available
        if hasattr(torch.backends, 'flash_attn'):
            torch.backends.flash_attn.enabled = True
    except:
        pass
    
    # Optimize for inference
    torch.set_grad_enabled(False)
    
    # Set memory fraction for GPU
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            # Use 90% of available GPU memory
            torch.cuda.set_per_process_memory_fraction(0.9)
        except:
            pass

def clear_memory():
    """Clear GPU and CPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def get_optimal_device() -> str:
    """Get the optimal device for model loading"""
    if torch.cuda.is_available():
        # Check available GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        if gpu_memory >= 8:  # 8GB or more
            return "cuda"
        else:
            return "cpu"  # Use CPU for smaller GPUs
    return "cpu"

def optimize_model_for_inference(model):
    """Apply inference optimizations to a loaded model"""
    model.eval()
    
    # Use JIT compilation if possible
    try:
        if hasattr(model, 'half'):
            model = model.half()  # Use half precision
    except:
        pass
    
    return model

def preload_models_async():
    """Preload models in background for faster subsequent access"""
    import threading
    import time
    
    def load_models_background():
        try:
            from .embeddings import load_embedding_model
            from .generation import load_falcon_model
            
            # Load models in background
            load_embedding_model()
            load_falcon_model()
        except Exception as e:
            print(f"Background model loading failed: {e}")
    
    # Start background loading
    thread = threading.Thread(target=load_models_background, daemon=True)
    thread.start()
    
    return thread

def get_model_size_mb(model_path: str) -> float:
    """Get the size of a model in MB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(model_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)  # Convert to MB

def check_system_requirements():
    """Check if system meets requirements for optimal performance"""
    requirements = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0,
        "cpu_count": os.cpu_count(),
        "available_ram_gb": None  # Would need psutil to get this
    }
    
    return requirements 