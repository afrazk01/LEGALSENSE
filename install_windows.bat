@echo off
echo Installing LegalSense dependencies for Windows...
echo.

echo Installing base requirements...
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

echo.
echo Note: bitsandbytes is not installed as it requires GPU support compilation on Windows.
echo The application will use CPU-optimized loading instead.
echo.

echo Installation complete!
echo You can now run: streamlit run app.py
pause 