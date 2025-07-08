import faiss
import pickle
import numpy as np
import re
import string
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from collections import Counter


index = faiss.read_index('data/faiss_index.index')


with open('data/section_texts.pkl', 'rb') as f:
    section_texts = pickle.load(f)


with open('data/section_nums.pkl', 'rb') as f:  
    section_nums = pickle.load(f)


section_dict = dict(zip(section_nums, section_texts))

model = SentenceTransformer('all-MiniLM-L6-v2')
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

def is_valid_answer(answer):
    """Check if answer is meaningful and complete"""
    
    if not answer.strip():
        return False
        
    
    words = answer.split()
    fragmented_count = sum(1 for word in words if len(word) < 3 or 
                          (len(word) > 3 and not any(v in word for v in 'aeiouAEIOU')))
    
    
    if fragmented_count / len(words) > 0.3:
        return False
        
    
    if sum(c in string.punctuation for c in answer) / len(answer) > 0.3:
        return False
        
    return True

def get_legal_answer(query):
    
    section_match = re.search(
        r'(?:section|s\.?|sec\.?|article|art\.?)\s*(\d+[A-Z\-]*)|^(\d+[A-Z\-]*)$', 
        query, 
        re.IGNORECASE
    )
    section_num = section_match.group(1) or section_match.group(2) if section_match else None

    
    if section_num and section_num in section_dict:
        return section_dict[section_num]
    
    
    query_embedding = model.encode([query]).astype('float32')
    _, indices = index.search(query_embedding, 5) 
    top_indices = indices[0]
    
    
    context = ""
    for idx in top_indices:
        if idx < len(section_texts):
            context += section_texts[idx] + "\n\n"
    
    if not context:
        return "⚠️ Could not find relevant legal provisions."
    
    try:
        
        result = qa_model(question=query, context=context, max_answer_len=100)
        answer = result['answer']
        confidence = result['score']
        
        
        if not is_valid_answer(answer):
            
            return section_texts[top_indices[0]]
        
        
        if confidence < 0.3:
            return section_texts[top_indices[0]]
        
        
        if re.match(r'^\d+[A-Z\-]*$', answer.strip()) and answer in section_dict:
            return section_dict[answer]
        
        return answer
    except Exception as e:
        
        return section_texts[top_indices[0]]