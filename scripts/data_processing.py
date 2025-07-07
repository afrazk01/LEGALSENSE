import pdfplumber
import re
import json
from pathlib import Path
import spacy
import logging

logging.basicConfig(filename='data_processing.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


nlp = spacy.load("en_core_web_sm")


def extract_text_from_pdf(pdf_path):
    full_text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for i,page in enumerate(pdf.pages):
            try:
                tables = page.extract_tables()
                table_text = "\n\n[TABLE]\n" + "\n".join(
                    " | ".join(str(cell) for row in table for cell in row if cell)
                    for table in tables
                ) + "\n[END TABLE]\n\n"

                text = page.extract_text(
                    layout=True,
                    x_tolerance=2,
                    y_tolerance=2,
                    keep_blank_chars=True
                )
                full_text.append(table_text + text + "\n")
                logging.info(f"Processed page {i+1}")
            except Exception as e:
                logging.error(f"Error processing page {i+1}: {str(e)}")
                continue

    return "\n".join(full_text)


def clean_text(text):
    text = re.sub(r'Page \d+ of \d+', '', text)
    text = re.sub(r'===== Page \d+ =====', '', text)
    text = re.sub(r'\[\d+\]', '', text)

    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)

    text = re.sub(r'\s+', ' ', text)

    text = re.sub(r'(?<!\w)\d{1,2}(?!\w)', '', text)

    return text.strip()


def preprocess_ppc(text):

    text = re.sub(r'Page \d+ of \d+', '', text)
    text = re.sub(r'===== Page \d+ =====', '', text)
    pattern = r'(?<=\n)(\d+[A-Z\-]*\.)\s+(.*?)(?=\n\d+[A-Z\-]*\.|\Z)'
    matches = re.findall(pattern, text, flags=re.DOTALL)
    
    structured_data = {}
    clean_structured_data = {} 
    
    for section_num, content in matches:
        section_num = section_num.strip().replace('.', '')
        content = content.strip()
        
        if content:
         
            structured_data[section_num] = content
        
            cleaned_text = lemmatize_text(content)
            clean_structured_data[section_num] = cleaned_text
    
    return structured_data, clean_structured_data 

def save_ppc_json(original_data, clean_data): 
    
    orig_path = Path(__file__).parent.parent / "data/processed/ppc_section_original.json"
    orig_path.parent.mkdir(parents=True, exist_ok=True)
    with open(orig_path, 'w') as f:
        json.dump(original_data, f, indent=4)
    
    
    clean_path = Path(__file__).parent.parent / "data/processed/ppc_section_clean.json"
    with open(clean_path, 'w') as f:
        json.dump(clean_data, f, indent=4)

pdf_text = extract_text_from_pdf('scripts/Pakistan Penal Code.pdf')
original_ppc, clean_ppc = preprocess_ppc(pdf_text)  
save_ppc_json(original_ppc, clean_ppc)  