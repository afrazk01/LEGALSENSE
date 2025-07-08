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


def extract_text_from_pdf(pdf_path):
    full_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                try:
                    text = page.extract_text(layout = True)
                    full_text.append(text or "")
                    logging.info(f"processed page {page.page_number}")
                except Exception as e:
                    logging.error(f"Error processing page {page.page_number}: {str(e)}")
    except Exception as e:
        logging.exception(f"Failed to open PDF: {str(e)}")
    return "\n".join(full_text)



