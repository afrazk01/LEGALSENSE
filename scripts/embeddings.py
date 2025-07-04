import json
from pathlib import Path
from sentence_transformers import SentenceTransformer


input_path = Path(__file__).parent.parent / "data/processed/ppc_section.json"

with open(input_path, 'r') as f:
    ppc_sections = json.load(f)


print('loading models...')
model = SentenceTransformer('all-MiniLM-L6-v2')


print('creating embeddings...')
