# LegalSense - Windows Installation Guide

This guide helps you install and run LegalSense on Windows systems, addressing common issues with PyTorch and model loading.

## 🚨 Common Windows Issues

### 1. **bitsandbytes GPU Support Error**
```
The installed version of bitsandbytes was compiled without GPU support.
```
**Solution:** The app now automatically detects this and falls back to CPU-optimized loading.

### 2. **PyTorch Classes Error**
```
RuntimeError: Tried to instantiate class '__path__._path', but it does not exist!
```
**Solution:** Use the simplified app version or set the environment variable.

## 🔧 Installation Steps

### Option 1: Quick Installation (Recommended)
```bash
# Run the Windows installation script
install_windows.bat
```

### Option 2: Manual Installation
```bash
# Install dependencies one by one
pip install streamlit>=1.28.0
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install torch>=2.0.0
pip install transformers>=4.35.0
pip install sentence-transformers>=2.2.0
pip install accelerate>=0.24.0
pip install nest-asyncio>=1.5.0
pip install scikit-learn>=1.3.0
pip install psutil>=5.9.0
```

**Note:** We skip `bitsandbytes` on Windows as it requires GPU support compilation.

## 🚀 Running the Application

### Option 1: Use Simplified App (Recommended for Windows)
```bash
streamlit run app_simple.py
```

### Option 2: Use Main App
```bash
streamlit run app.py
```

## 🛠️ Troubleshooting

### If you get PyTorch errors:

1. **Set environment variable:**
   ```bash
   set PYTORCH_ENABLE_MPS_FALLBACK=1
   streamlit run app.py
   ```

2. **Or use the simplified version:**
   ```bash
   streamlit run app_simple.py
   ```

### If models load slowly:

1. **Check your hardware:**
   - Ensure you have at least 8GB RAM
   - Use SSD storage for models
   - Close other applications

2. **First run optimization:**
   - The first run will be slower due to model loading
   - Subsequent runs will be much faster due to caching

### If you get memory errors:

1. **Reduce model precision:**
   - The app automatically uses CPU-optimized loading on Windows
   - Models will use FP32 precision instead of FP16

2. **Close other applications:**
   - Free up RAM before running the app
   - Close browser tabs and other memory-intensive apps

## 📊 Performance Expectations on Windows

| Hardware | Expected Load Time | Memory Usage |
|----------|-------------------|--------------|
| 8GB RAM, HDD | 45-60 seconds | 4-6GB |
| 16GB RAM, SSD | 25-35 seconds | 3-4GB |
| 32GB RAM, SSD | 15-25 seconds | 2-3GB |

## 🔍 System Requirements

### Minimum Requirements:
- Windows 10/11
- 8GB RAM
- 5GB free disk space
- Python 3.8+

### Recommended Requirements:
- Windows 10/11
- 16GB RAM
- SSD storage
- Python 3.9+

## 📝 File Structure

```
LEGALSENSE/
├── app.py              # Main application (may have PyTorch issues)
├── app_simple.py       # Simplified version (recommended for Windows)
├── install_windows.bat # Windows installation script
├── requirements.txt    # Dependencies
├── models/            # Model files
├── script/            # Core functionality
└── chunks_and_embeddings.csv # Knowledge base
```

## 🎯 Quick Start

1. **Install dependencies:**
   ```bash
   install_windows.bat
   ```

2. **Run the app:**
   ```bash
   streamlit run app_simple.py
   ```

3. **Open browser:**
   - Go to `http://localhost:8501`
   - Wait for models to load (first time only)
   - Start asking legal questions!

## 🔄 Updates and Maintenance

### To update the application:
1. Backup your models directory
2. Pull the latest code
3. Run `install_windows.bat` again
4. Restart the application

### To clear cache (if needed):
1. Stop the Streamlit app
2. Delete `.streamlit/cache` directory
3. Restart the app

## 📞 Support

If you encounter issues:

1. **Check the error messages** - they often contain helpful information
2. **Try the simplified app** - `app_simple.py` has fewer dependencies
3. **Ensure sufficient RAM** - close other applications
4. **Check Python version** - ensure you're using Python 3.8+

## 🎉 Success Indicators

You'll know everything is working when:
- ✅ Models load without errors
- ✅ You can ask legal questions
- ✅ Responses are generated quickly
- ✅ Sources are displayed correctly

The application will show success messages for each loaded component:
- ✅ Embedding model loaded
- ✅ Falcon model loaded
- ✅ Knowledge base loaded 