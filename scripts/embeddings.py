import json
from pathlib import Path
from sentence_transformers import SentenceTransformer


input_path = Path(__file__).parent.parent / "data/processed/ppc_section.json"
