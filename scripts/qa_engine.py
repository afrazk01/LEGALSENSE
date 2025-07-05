from transformers import pipeline
from scripts.retriever import search_similar_sections


qa_model = pipeline("question-answering", model = "deepset/roberta-base-squad2")

