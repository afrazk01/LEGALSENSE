import json
import faiss
import numpy as np
import os
import pickle

with open('data/embeddings/ppc_embeddings.json', 'r') as f:
    data = json.load(f)


embeddings = []
texts = []

for item in data:
    embeddings.append(item['embedding'])
    texts.append(item['text'])

embeddings = np.array(embeddings).astype('float32')


