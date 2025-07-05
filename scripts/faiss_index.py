import json
import faiss
import numpy as np
import pickle

with open('data/embeddings/ppc_embeddings_fixed.json', 'r') as f:
    data = json.load(f)

embeddings = []
texts = []  
section_nums = []  

for item in data:
    embeddings.append(item['embedding'])
    texts.append(item['text'])  
    section_nums.append(item['section_num'])  


embeddings = np.array(embeddings).astype('float32')
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, 'data/faiss_index.index')


with open('data/section_texts.pkl', 'wb') as f:
    pickle.dump(texts, f)  
    
with open('data/section_nums.pkl', 'wb') as f:  
    pickle.dump(section_nums, f)  

print("Rebuilt FAISS index with actual text content.")