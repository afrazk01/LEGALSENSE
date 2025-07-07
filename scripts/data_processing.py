import pdfplumber
import re
import json
from pathlib import Path
import logging
import spacy
from collections import defaultdict
import yaml

logging.basicConfig(
    filename='data_processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Legal document structure patterns
STRUCTURE_PATTERNS = {
    'chapter': r'(CHAPTER\s+[IVXLCDM]+)\s*\n([^\n]+)',
    'section': r'^\s*(\d+[A-Za-z\-]*[\.\)\]]?)\s*(.*?)(?=\n\s*\d+[A-Za-z\-]*[\.\)\]]?|\nCHAPTER|\Z)',
    'subsection': r'^\s*\(([a-z\d]+)\)\s*(.*?)$'
}

def extract_text_from_pdf(pdf_path):
    full_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                try:
                    text = page.extract_text(layout=True, x_tolerance=1, y_tolerance=1)
                    tables = page.extract_tables()
                    
                    table_text = ""
                    if tables:
                        table_text += "\n\n[TABLE START]\n"
                        for table in tables:
                            for row in table:
                                table_text += "| " + " | ".join(str(cell or "") for cell in row) + " |\n"
                        table_text += "[TABLE END]\n\n"
                    
                    full_text.append(table_text + (text or "") + "\n")
                    logging.info(f"Processed page {page.page_number}")
                except Exception as e:
                    logging.error(f"Error processing page {page.page_number}: {str(e)}")
    except Exception as e:
        logging.exception(f"Failed to open PDF: {str(e)}")
    return "\n".join(full_text)

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
    return text.strip()

def preserve_legal_terms(text):
    # Preserve important legal terms during normalization
    doc = nlp(text)
    preserved = []
    for token in doc:
        if token.ent_type_ in ['LAW', 'ORG'] or token.text.isupper():
            preserved.append(token.text)
        else:
            preserved.append(token.text.lower())
    return ' '.join(preserved)

def parse_legal_document(text):
    structured_data = defaultdict(lambda: {"title": "", "sections": defaultdict(dict)})
    cleaned_text = clean_text(text)
    
    chapters = re.findall(STRUCTURE_PATTERNS['chapter'], cleaned_text, re.IGNORECASE | re.MULTILINE)
    
    for chapter_code, chapter_title in chapters:
        chapter_key = f"{chapter_code.strip().replace(' ', '_')}"
        chapter_title = chapter_title.strip()
        structured_data[chapter_key]["title"] = chapter_title
        
        chapter_block_regex = rf'{re.escape(chapter_code)}\s*{re.escape(chapter_title)}(.*?)(?=\nCHAPTER\s+[IVXLCDM]+|\Z)'
        chapter_block = re.search(chapter_block_regex, cleaned_text, re.IGNORECASE | re.DOTALL)
        
        if chapter_block:
            chapter_content = chapter_block.group(1)
            sections = re.findall(STRUCTURE_PATTERNS['section'], chapter_content, re.MULTILINE | re.DOTALL)
            
            for section_num, section_text in sections:
                section_text = section_text.strip()
                if not section_text:
                    continue
                
                # Handle subsections
                subsections = re.findall(STRUCTURE_PATTERNS['subsection'], section_text, re.MULTILINE)
                if subsections:
                    structured_data[chapter_key]["sections"][section_num] = {
                        "main_text": "",
                        "subsections": {}
                    }
                    main_text, _, subs_text = section_text.partition('\n')
                    structured_data[chapter_key]["sections"][section_num]["main_text"] = preserve_legal_terms(main_text.strip())
                    
                    for sub_num, sub_text in subsections:
                        structured_data[chapter_key]["sections"][section_num]["subsections"][sub_num] = preserve_legal_terms(sub_text.strip())
                else:
                    structured_data[chapter_key]["sections"][section_num] = preserve_legal_terms(section_text)

    return dict(structured_data)

def save_legal_data(data, base_dir, filename):
    try:
        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        output_path = base_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        logging.info(f"Saved legal data to: {'data/processed/' + filename}")
        return True
    except Exception as e:
        logging.error(f"Error saving data: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        config = yaml.safe_load(open("config.yaml", "r"))
        pdf_path = config['pdf_path']
        output_dir = config['output_dir']
        
        logging.info(f"Processing: {pdf_path}")
        pdf_text = extract_text_from_pdf(pdf_path)
        structured_data = parse_legal_document(pdf_text)
        
        if save_legal_data(structured_data, output_dir, "ppc_structured.json"):
            logging.info("Processing completed successfully")
    except Exception as e:
        logging.exception("Fatal error during processing")