import os
import sys

# Disable PyTorch classes before any imports
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

# Import streamlit first
import streamlit as st

# Set page config early
st.set_page_config(
    page_title="Pakistan Legal Assistant", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Now import other modules with error handling
try:
    import pandas as pd
    import numpy as np
except Exception as e:
    st.error(f"Error importing pandas/numpy: {e}")
    st.stop()

# Import ML modules with error handling
try:
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception as e:
    st.error(f"Error importing ML libraries: {e}")
    st.error("Please ensure all dependencies are installed correctly.")
    st.stop()

# Import local modules
try:
    from script.retrieval import find_relevant_chunks
    from script.promp_engineering import create_legal_prompt
except Exception as e:
    st.error(f"Error importing local modules: {e}")
    st.stop()

# Initialize models with improved caching
@st.cache_resource(show_spinner=False)
def load_embedding_model_cached():
    """Load embedding model with caching - uses online model"""
    try:
        with st.spinner("Loading embedding model..."):
            model = SentenceTransformer('all-mpnet-base-v2')
        return model
    except Exception as e:
        st.error(f"Error loading embedding model: {str(e)}")
        return None

@st.cache_resource(show_spinner=False)
def load_falcon_model_cached():
    """Load Falcon model with caching - uses online model"""
    try:
        with st.spinner("Loading Falcon model..."):
            tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-rw-1b")
            model = AutoModelForCausalLM.from_pretrained(
                "tiiuae/falcon-rw-1b",
                torch_dtype=torch.float32,
                trust_remote_code=True,
                device_map="cpu"  # Force CPU to avoid GPU issues
            )
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading Falcon model: {str(e)}")
        return None, None

# Load data with improved caching
@st.cache_data(show_spinner=False)
def load_data():
    """Load knowledge base with caching"""
    try:
        # Use relative path
        data_path = os.path.join(os.path.dirname(__file__), "chunks_and_embeddings.csv")
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        st.error(f"Error loading knowledge base: {str(e)}")
        return pd.DataFrame()

def generate_embeddings(text, model):
    """Generate embeddings with mpnet - optimized for speed"""
    try:
        with torch.no_grad():  # Disable gradient computation for inference
            return model.encode(
                text,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False  # Disable progress bar for faster processing
            )
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return None

def generate_answer(prompt, tokenizer, model):
    """Generate answer with optimized inference"""
    try:
        # Tokenize with optimized settings
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=2048, 
            truncation=True,
            padding=True
        )
        
        # Move inputs to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate with optimized settings
        with torch.no_grad():  # Disable gradient computation
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.5,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,  # Enable KV cache for faster generation
                repetition_penalty=1.1  # Reduce repetition
            )
        
        # Decode and clean output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "<|assistant|>" in generated_text:
            return generated_text.split("<|assistant|>")[-1].strip()
        else:
            return generated_text.strip()
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return "Sorry, I encountered an error while generating the response."

def initialize_models():
    """Initialize all models efficiently"""
    # Load models in parallel using session state
    if "models_loaded" not in st.session_state:
        st.session_state.models_loaded = False
        st.session_state.embed_model = None
        st.session_state.falcon_tokenizer = None
        st.session_state.falcon_model = None
    
    if not st.session_state.models_loaded:
        with st.spinner("Loading AI models (this may take a moment on first run)..."):
            # Load embedding model
            embed_model = load_embedding_model_cached()
            if embed_model is not None:
                st.session_state.embed_model = embed_model
                st.success("✅ Embedding model loaded")
            
            # Load Falcon model
            falcon_tokenizer, falcon_model = load_falcon_model_cached()
            if falcon_tokenizer is not None and falcon_model is not None:
                st.session_state.falcon_tokenizer = falcon_tokenizer
                st.session_state.falcon_model = falcon_model
                st.success("✅ Falcon model loaded")
            
            st.session_state.models_loaded = True
    
    return (st.session_state.embed_model, 
            st.session_state.falcon_tokenizer, 
            st.session_state.falcon_model)

def main():
    st.title("🇵🇰 Pakistan Penal Code Expert")
    st.caption("AI Legal Assistant powered by Falcon-RW-1B and mpnet embeddings (Minimal Mode)")
    
    # Show info about minimal mode
    with st.expander("ℹ️ Minimal Mode Info"):
        st.info("""
        **Minimal Mode**: This version is optimized for Windows compatibility.
        - Bypasses PyTorch classes error
        - Uses CPU-only mode for stability
        - Downloads models automatically
        - Requires internet connection for first run
        """)
    
    # Initialize session state
    if "history" not in st.session_state:
        st.session_state.history = []
    
    # Load models efficiently
    embed_model, falcon_tokenizer, falcon_model = initialize_models()
    
    # Load data
    df = load_data()
    
    # Check if models are loaded
    if embed_model is None or falcon_tokenizer is None or falcon_model is None:
        st.error("❌ Failed to load required models. Please check your internet connection and refresh the page.")
        return
    
    # User input
    query = st.chat_input("Ask your legal question about Pakistan Penal Code...")
    
    if query:
        # Add to history
        st.session_state.history.append(("user", query))
        
        try:
            # Show processing status
            with st.status("Analyzing legal provisions...", expanded=False):
                # 1. Generate query embedding
                query_embedding = generate_embeddings(query, embed_model)
                if query_embedding is None:
                    st.error("Failed to generate embeddings")
                    return
                
                # 2. Retrieve relevant context
                relevant_chunks = find_relevant_chunks(query_embedding, df)
                context = "\n\n".join([
                    f"Page {row['page_number']}: {row['sentence_chunk']}" 
                    for _, row in relevant_chunks.iterrows()
                ])
                
                # 3. Create legal-optimized prompt
                prompt = create_legal_prompt(context, query)
                
                # 4. Generate answer
                answer = generate_answer(prompt, falcon_tokenizer, falcon_model)
                
                # 5. Format sources
                sources = []
                for _, row in relevant_chunks.iterrows():
                    sources.append({
                        "page": row["page_number"],
                        "content": row["sentence_chunk"]
                    })
            
            # Add to history
            st.session_state.history.append(("assistant", answer, sources))
            
        except Exception as e:
            st.error(f"Error processing your request: {str(e)}")
            st.session_state.history.append(("assistant", "Sorry, I encountered an error processing your request"))
    
    # Display conversation
    for i, entry in enumerate(st.session_state.history):
        role = entry[0]
        
        with st.chat_message(role.capitalize()):
            # Display message
            st.write(entry[1])
            
            # Display sources for assistant
            if role == "assistant" and len(entry) > 2:
                _, answer, sources = entry
                
                # Sources expander
                with st.expander(f"📚 Legal Sources ({len(sources)} provisions)"):
                    for source in sources:
                        st.caption(f"**Page {source['page']}**")
                        st.write(source["content"])
                        st.divider()

if __name__ == "__main__":
    main() 