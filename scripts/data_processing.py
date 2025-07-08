import pdfplumber
import json
import re
from pathlib import Path
import logging
import spacy
from collections import defaultdict


logging.basicConfig(
    filename='data_processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


nlp = spacy.load("en_core_web_sm", disable=["ner","parser"])


Structure_pattern = {
    'chapter': r'(CHAPTER\s+[IVXLCDM]+)\s*\n+([^\n]+)',
    'section': r'^\s*(\d+[A-Za-z\-]*)[\.\:\)]?\s+(.*?)(?=\n\s*\d+[A-Za-z\-]*[\.\:\)]|\nCHAPTER|\Z)',
    'subsection': r'^\s*\(([a-z\d]+)\)\s+(.*?)$'
}

