#!/usr/bin/env python3
"""
Script to download required models for LegalSense
"""
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from sentence_transformers import SentenceTransformer
import torch

def download_models():
    """Download required models from Hugging Face Hub"""
    
    print("🚀 Starting model downloads...")
    print("This may take several minutes depending on your internet connection.")
    print()
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Download embedding model
    print("📥 Downloading embedding model (all-mpnet-base-v2)...")
    try:
        embed_model = SentenceTransformer('all-mpnet-base-v2')
        # Save to local directory
        embed_model.save('models/all-mpnet-base-v2')
        print("✅ Embedding model downloaded successfully!")
    except Exception as e:
        print(f"❌ Error downloading embedding model: {e}")
        return False
    
    # Download Falcon model
    print("\n📥 Downloading Falcon model (tiiuae/falcon-rw-1b)...")
    try:
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-rw-1b")
        tokenizer.save_pretrained("models/falcon-rw-1b")
        
        # Download model
        model = AutoModelForCausalLM.from_pretrained(
            "tiiuae/falcon-rw-1b",
            torch_dtype=torch.float32,  # Use float32 for compatibility
            trust_remote_code=True
        )
        model.save_pretrained("models/falcon-rw-1b")
        
        print("✅ Falcon model downloaded successfully!")
    except Exception as e:
        print(f"❌ Error downloading Falcon model: {e}")
        return False
    
    print("\n🎉 All models downloaded successfully!")
    print("You can now run: streamlit run app_simple.py")
    
    return True

def check_models():
    """Check if models are properly downloaded"""
    print("🔍 Checking model files...")
    
    # Check embedding model
    embed_path = "models/all-mpnet-base-v2"
    if os.path.exists(embed_path):
        files = os.listdir(embed_path)
        if any(f.endswith(('.bin', '.safetensors')) for f in files):
            print("✅ Embedding model files found")
        else:
            print("❌ Embedding model missing weight files")
            return False
    else:
        print("❌ Embedding model directory not found")
        return False
    
    # Check Falcon model
    falcon_path = "models/falcon-rw-1b"
    if os.path.exists(falcon_path):
        files = os.listdir(falcon_path)
        if any(f.endswith(('.bin', '.safetensors')) for f in files):
            print("✅ Falcon model files found")
        else:
            print("❌ Falcon model missing weight files")
            return False
    else:
        print("❌ Falcon model directory not found")
        return False
    
    return True

def main():
    print("🔧 LegalSense Model Downloader")
    print("=" * 50)
    
    # Check if models already exist
    if check_models():
        print("\n✅ Models are already downloaded!")
        response = input("Do you want to re-download them? (y/N): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    # Download models
    success = download_models()
    
    if success:
        print("\n✅ Setup complete! You can now run the application.")
    else:
        print("\n❌ Setup failed. Please check your internet connection and try again.")

if __name__ == "__main__":
    main() 