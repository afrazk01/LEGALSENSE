"""
Performance monitoring utilities for tracking model loading and inference times
"""
import time
import psutil
import torch
from typing import Dict, Any
from functools import wraps

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, name: str):
        """Start timing an operation"""
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End timing an operation and return duration"""
        if name in self.start_times:
            duration = time.time() - self.start_times[name]
            self.metrics[name] = duration
            del self.start_times[name]
            return duration
        return 0.0
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all recorded metrics"""
        return self.metrics.copy()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get current system information"""
        info = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "disk_usage_percent": psutil.disk_usage('/').percent
        }
        
        if torch.cuda.is_available():
            info.update({
                "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / (1024**2),
                "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / (1024**2),
                "gpu_memory_percent": (torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()) * 100
            })
        
        return info

# Global performance monitor instance
monitor = PerformanceMonitor()

def time_operation(operation_name: str):
    """Decorator to time operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor.start_timer(operation_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = monitor.end_timer(operation_name)
                print(f"⏱️ {operation_name} took {duration:.2f} seconds")
        return wrapper
    return decorator

def log_performance_metrics():
    """Log current performance metrics"""
    metrics = monitor.get_metrics()
    system_info = monitor.get_system_info()
    
    print("\n📊 Performance Metrics:")
    for operation, duration in metrics.items():
        print(f"  {operation}: {duration:.2f}s")
    
    print("\n💻 System Info:")
    print(f"  CPU Usage: {system_info['cpu_percent']:.1f}%")
    print(f"  Memory Usage: {system_info['memory_percent']:.1f}%")
    print(f"  Available Memory: {system_info['memory_available_gb']:.1f}GB")
    
    if torch.cuda.is_available():
        print(f"  GPU Memory: {system_info['gpu_memory_allocated_mb']:.1f}MB allocated")
        print(f"  GPU Memory Reserved: {system_info['gpu_memory_reserved_mb']:.1f}MB")

def get_model_loading_benchmark():
    """Benchmark model loading performance"""
    import os
    from .embeddings import load_embedding_model
    from .generation import load_falcon_model
    
    print("🚀 Starting Model Loading Benchmark...")
    
    # Benchmark embedding model
    monitor.start_timer("embedding_model_load")
    try:
        embed_model = load_embedding_model()
        embed_duration = monitor.end_timer("embedding_model_load")
        print(f"✅ Embedding model loaded in {embed_duration:.2f}s")
    except Exception as e:
        print(f"❌ Embedding model failed: {e}")
    
    # Benchmark Falcon model
    monitor.start_timer("falcon_model_load")
    try:
        falcon_tokenizer, falcon_model = load_falcon_model()
        falcon_duration = monitor.end_timer("falcon_model_load")
        print(f"✅ Falcon model loaded in {falcon_duration:.2f}s")
    except Exception as e:
        print(f"❌ Falcon model failed: {e}")
    
    # Log all metrics
    log_performance_metrics()
    
    return monitor.get_metrics() 