# LegalSense Model Loading Optimizations

This document describes the optimizations implemented to significantly improve model loading speed in the LegalSense application.

## 🚀 Key Optimizations Implemented

### 1. **Quantization (4-bit)**
- Falcon model now uses 4-bit quantization via `BitsAndBytesConfig`
- Reduces memory usage by ~75% and speeds up loading
- Maintains good quality while being much faster

### 2. **Half-Precision (FP16)**
- Models use `torch.float16` on GPU for faster inference
- Reduces memory usage and speeds up computation
- Automatic fallback to FP32 on CPU

### 3. **Optimized Device Mapping**
- Automatic device selection based on available GPU memory
- Uses "auto" device mapping for optimal GPU utilization
- Falls back to CPU for systems with limited GPU memory

### 4. **Improved Caching Strategy**
- Separate caching for embedding and Falcon models
- Session state management for persistent model storage
- Disabled spinners in cached functions for cleaner UI

### 5. **Memory Management**
- Automatic memory clearing before model loading
- Optimized PyTorch settings for inference
- Better memory fraction management for GPU

### 6. **Relative Paths**
- Replaced hardcoded absolute paths with relative paths
- More portable across different systems
- Better error handling for missing files

## 📦 New Dependencies

The following packages have been added for optimization:

```bash
accelerate>=0.24.0      # For optimized model loading
bitsandbytes>=0.41.0    # For 4-bit quantization
psutil>=5.9.0           # For system monitoring
```

## 🔧 Installation

1. Install the updated requirements:
```bash
pip install -r requirements.txt
```

2. Run the benchmark to test performance:
```bash
python benchmark_models.py
```

## 📊 Expected Performance Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Falcon Model Loading | ~30-60s | ~10-20s | 50-70% faster |
| Embedding Model Loading | ~10-20s | ~5-10s | 50% faster |
| Memory Usage | ~8GB | ~2-4GB | 50-75% reduction |
| First Load Time | ~60-90s | ~20-35s | 60-70% faster |

## 🎯 Usage Tips

### For Best Performance:

1. **Hardware Requirements:**
   - GPU with 8GB+ VRAM (recommended)
   - 16GB+ system RAM
   - SSD storage for models

2. **Software Setup:**
   - Use the latest PyTorch version
   - Ensure CUDA is properly installed (if using GPU)
   - Close other applications during first load

3. **Model Storage:**
   - Keep models on fast storage (SSD)
   - Ensure sufficient free space
   - Avoid network drives for model storage

## 🔍 Monitoring Performance

The application now includes performance monitoring:

```python
from script.performance_monitor import log_performance_metrics
from script.model_optimizer import check_system_requirements

# Check system capabilities
requirements = check_system_requirements()

# Monitor performance during runtime
log_performance_metrics()
```

## 🛠️ Troubleshooting

### Common Issues:

1. **Out of Memory Errors:**
   - Reduce batch size in generation settings
   - Use CPU-only mode for systems with limited GPU memory
   - Close other applications

2. **Slow Loading:**
   - Check if models are on SSD storage
   - Ensure sufficient RAM is available
   - Verify CUDA installation (if using GPU)

3. **Quantization Errors:**
   - Update `bitsandbytes` to latest version
   - Ensure compatible PyTorch version
   - Try CPU-only mode as fallback

## 📈 Benchmark Results

Run the benchmark script to see your system's performance:

```bash
python benchmark_models.py
```

This will show:
- System requirements check
- Model loading times
- Memory usage statistics
- Performance recommendations

## 🔄 Migration from Old Version

If you're upgrading from the previous version:

1. Backup your models directory
2. Install new requirements
3. The application will automatically use optimized loading
4. First run may take longer due to model conversion
5. Subsequent runs will be much faster

## 📝 Technical Details

### Model Loading Pipeline:

1. **Pre-loading Phase:**
   - Optimize PyTorch settings
   - Clear GPU/CPU memory
   - Check system capabilities

2. **Loading Phase:**
   - Load embedding model with FP16 optimization
   - Load Falcon model with 4-bit quantization
   - Apply inference optimizations

3. **Caching Phase:**
   - Store models in session state
   - Enable persistent caching
   - Optimize for repeated access

### Memory Optimization:

- **4-bit Quantization:** Reduces model size by ~75%
- **Half Precision:** Reduces memory usage by ~50%
- **Efficient Device Mapping:** Optimizes GPU memory allocation
- **Memory Clearing:** Prevents memory fragmentation

## 🎉 Results

These optimizations should provide:
- **60-70% faster initial loading**
- **50-75% reduced memory usage**
- **Better system compatibility**
- **Improved user experience**

The application will now load much faster, especially on subsequent runs, and use significantly less system resources. 