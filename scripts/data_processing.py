import pdfplumber
import re
import json
from pathlib import Path
import spacy

nlp = spacy.load("en_core_web_sm")


def extract_text_from_pdf(pdf_path):
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + '\n'
    return text


def lemmatize_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)


def preprocess_ppc(text):
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