#!/usr/bin/env python3
"""
Benchmark script to test optimized model loading performance
"""
import time
import sys
import os

# Add the script directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'script'))

from script.performance_monitor import get_model_loading_benchmark, log_performance_metrics
from script.model_optimizer import check_system_requirements

def main():
    print("🔧 LegalSense Model Loading Benchmark")
    print("=" * 50)
    
    # Check system requirements
    print("\n📋 System Requirements Check:")
    requirements = check_system_requirements()
    for key, value in requirements.items():
        print(f"  {key}: {value}")
    
    # Run benchmark
    print("\n" + "=" * 50)
    metrics = get_model_loading_benchmark()
    
    # Summary
    print("\n" + "=" * 50)
    print("📈 Performance Summary:")
    total_time = sum(metrics.values())
    print(f"  Total loading time: {total_time:.2f}s")
    
    if len(metrics) >= 2:
        print(f"  Average per model: {total_time/len(metrics):.2f}s")
    
    print("\n✅ Benchmark completed!")
    print("💡 Tips for faster loading:")
    print("  - Use SSD storage for models")
    print("  - Ensure sufficient RAM (16GB+)")
    print("  - Use GPU with 8GB+ VRAM for best performance")
    print("  - Close other applications during first load")

if __name__ == "__main__":
    main() 