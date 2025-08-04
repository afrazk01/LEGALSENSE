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
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
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
def load_text_generation_model_cached():
    """Load text generation model using pipeline - more compatible"""
    try:
        with st.spinner("Loading text generation model..."):
            # Use a smaller, more compatible model
            generator = pipeline(
                "text-generation",
                model="gpt2",  # Much smaller and more compatible
                torch_dtype=torch.float32,
                device_map="cpu"
            )
        return generator
    except Exception as e:
        st.error(f"Error loading text generation model: {str(e)}")
        return None

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

def generate_answer(prompt, generator):
    """Generate answer using pipeline - more compatible"""
    try:
        # Generate text using pipeline
        result = generator(
            prompt,
            max_length=len(prompt.split()) + 100,  # Add 100 words
            temperature=0.7,
            do_sample=True,
            pad_token_id=generator.tokenizer.eos_token_id
        )
        
        # Extract generated text
        generated_text = result[0]['generated_text']
        
        # Remove the original prompt
        if generated_text.startswith(prompt):
            answer = generated_text[len(prompt):].strip()
        else:
            answer = generated_text.strip()
        
        # Clean up the answer
        if answer:
            # Take first sentence or first 200 characters
            sentences = answer.split('.')
            if sentences:
                answer = sentences[0].strip()
                if len(answer) > 200:
                    answer = answer[:200] + "..."
        else:
            answer = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
        
        return answer
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return "Sorry, I encountered an error while generating the response."

def initialize_models():
    """Initialize all models efficiently"""
    # Load models in parallel using session state
    if "models_loaded" not in st.session_state:
        st.session_state.models_loaded = False
        st.session_state.embed_model = None
        st.session_state.text_generator = None
    
    if not st.session_state.models_loaded:
        with st.spinner("Loading AI models (this may take a moment on first run)..."):
            # Load embedding model
            embed_model = load_embedding_model_cached()
            if embed_model is not None:
                st.session_state.embed_model = embed_model
                st.success("✅ Embedding model loaded")
            
            # Load text generation model
            text_generator = load_text_generation_model_cached()
            if text_generator is not None:
                st.session_state.text_generator = text_generator
                st.success("✅ Text generation model loaded")
            
            st.session_state.models_loaded = True
    
    return (st.session_state.embed_model, 
            st.session_state.text_generator)

def main():
    st.title("🇵🇰 Pakistan Penal Code Expert")
    st.caption("AI Legal Assistant powered by GPT-2 and mpnet embeddings (Alternative Mode)")
    
    # Show info about alternative mode
    with st.expander("ℹ️ Alternative Mode Info"):
        st.info("""
        **Alternative Mode**: This version uses GPT-2 instead of Falcon for better compatibility.
        - Uses GPT-2 (smaller, more compatible model)
        - Bypasses PyTorch classes error
        - Uses CPU-only mode for stability
        - Downloads models automatically
        - May have slightly different response style
        """)
    
    # Initialize session state
    if "history" not in st.session_state:
        st.session_state.history = []
    
    # Load models efficiently
    embed_model, text_generator = initialize_models()
    
    # Load data
    df = load_data()
    
    # Check if models are loaded
    if embed_model is None or text_generator is None:
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
                answer = generate_answer(prompt, text_generator)
                
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