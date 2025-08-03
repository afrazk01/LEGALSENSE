import streamlit as st
import pandas as pd
from script.embeddings import load_embedding_model, generate_embeddings
from script.retrieval import find_relevant_chunks
from script.promp_engineering import create_legal_prompt
from script.generation import load_falcon_model, generate_answer
from script.confidence import calculate_confidence


@st.cache_resource
def load_models():
    embed_model = load_embedding_model()
    falcon_tokenizer, falcon_model = load_falcon_model()
    return embed_model, falcon_tokenizer, falcon_model


@st.cache_data
def load_data():
    df = pd.read_csv("U:\\LEGALSENSE\\chunks_and_embeddings.csv")
    return df

def main():
    st.set_page_config(page_title="Pakistan Legal Assistant", layout="wide")
    st.title("🇵🇰 Pakistan Penal Code Expert")
    st.caption("AI Legal Assistant powered by Falcon-RW-1B and mpnet embeddings")
    
    
    if "history" not in st.session_state:
        st.session_state.history = []
    
    
    embed_model, falcon_tokenizer, falcon_model = load_models()
    df = load_data()
    
    
    query = st.chat_input("Ask your legal question about Pakistan Penal Code...")
    
    if query:
        
        st.session_state.history.append(("user", query))
        
        
        with st.status("Analyzing legal provisions...", expanded=False):
            
            query_embedding = generate_embeddings(query, embed_model)
            
            
            relevant_chunks = find_relevant_chunks(query_embedding, df)
            context = "\n\n".join([
                f"Page {row['page_number']}: {row['sentence_chunk']}" 
                for _, row in relevant_chunks.iterrows()
            ])
            
            
            prompt = create_legal_prompt(context, query)
            
            
            answer = generate_answer(prompt, falcon_tokenizer, falcon_model)
            
            
            sources = []
            for _, row in relevant_chunks.iterrows():
                sources.append({
                    "page": row["page_number"],
                    "content": row["sentence_chunk"]
                })
        
        
        st.session_state.history.append(("assistant", answer, sources))
    
    
    for i, entry in enumerate(st.session_state.history):
        role = entry[0]
        
        with st.chat_message(role.capitalize()):
            
            st.write(entry[1])
            
            
            if role == "assistant":
                _, answer, sources = entry
                
                
                with st.expander(f"📚 Legal Sources ({len(sources)} provisions)"):
                    for source in sources:
                        st.caption(f"**Page {source['page']}**")
                        st.write(source["content"])
                        st.divider()

if __name__ == "__main__":
    main()