import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer


model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index('data/faiss_index.index')
texts = pickle.load(open('data/section_texts.pkl', 'rb'))

def search_similar_sections(query, top_k = 5):
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    results = [texts[i] for i in indices[0]]
    return results


