import faiss
import pickle
import numpy as np
import re
import string
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from collections import Counter

# Load FAISS index
index = faiss.read_index('data/faiss_index.index')

# Load section texts
with open('data/section_texts.pkl', 'rb') as f:
    section_texts = pickle.load(f)

# Load section numbers
with open('data/section_nums.pkl', 'rb') as f:  
    section_nums = pickle.load(f)

# Create section dictionary
section_dict = dict(zip(section_nums, section_texts))

model = SentenceTransformer('all-MiniLM-L6-v2')
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

def is_valid_answer(answer):
    """Check if answer is meaningful and complete"""
    # Check for empty answer
    if not answer.strip():
        return False
        
    # Check for too many fragmented words
    words = answer.split()
    fragmented_count = sum(1 for word in words if len(word) < 3 or 
                          (len(word) > 3 and not any(v in word for v in 'aeiouAEIOU')))
    
    # If more than 30% of words are fragmented, it's invalid
    if fragmented_count / len(words) > 0.3:
        return False
        
    # Check if answer is mostly punctuation or numbers
    if sum(c in string.punctuation for c in answer) / len(answer) > 0.3:
        return False
        
    return True

def get_legal_answer(query):
    # Extract section number from query
    section_match = re.search(
        r'(?:section|s\.?|sec\.?|article|art\.?)\s*(\d+[A-Z\-]*)|^(\d+[A-Z\-]*)$', 
        query, 
        re.IGNORECASE
    )
    section_num = section_match.group(1) or section_match.group(2) if section_match else None

    # Return full text if section number is requested
    if section_num and section_num in section_dict:
        return section_dict[section_num]
    
    # Semantic search for other queries
    query_embedding = model.encode([query]).astype('float32')
    _, indices = index.search(query_embedding, 5)  # Get top 5 results
    top_indices = indices[0]
    
    # Build context from top relevant sections
    context = ""
    for idx in top_indices:
        if idx < len(section_texts):
            context += section_texts[idx] + "\n\n"
    
    if not context:
        return "⚠️ Could not find relevant legal provisions."
    
    try:
        # Get QA result with increased max_answer_len
        result = qa_model(question=query, context=context, max_answer_len=100)
        answer = result['answer']
        confidence = result['score']
        
        # 1. Check if answer is meaningful
        if not is_valid_answer(answer):
            # Return the most relevant section instead
            return section_texts[top_indices[0]]
        
        # 2. Handle low-confidence answers
        if confidence < 0.3:
            return section_texts[top_indices[0]]
        
        # 3. Handle section references in answers
        if re.match(r'^\d+[A-Z\-]*$', answer.strip()) and answer in section_dict:
            return section_dict[answer]
        
        return answer
    except Exception as e:
        # Return the most relevant section on error
        return section_texts[top_indices[0]]