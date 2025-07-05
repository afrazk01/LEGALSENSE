import json
import faiss
import numpy as np
import pickle

data = []
with open('data/embeddings/ppc_embeddings_fixed.json', 'r') as f:
    data = json.load(f)


embeddings = []
texts = []

for item in data:
    embeddings.append(item['embedding'])
    texts.append(item['text'])


embeddings = np.array(embeddings).astype('float32')

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)


faiss.write_index(index, 'data/faiss_index.index')
with open('data/section_texts.pkl', 'wb') as f:
    pickle.dump(texts, f)

print("FAISS index and section texts saved.")
