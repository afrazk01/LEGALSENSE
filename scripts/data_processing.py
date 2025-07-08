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


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
    text = re.sub(r'\d+\[\]\d*', '', text)  
    return text.strip()


def preserve_legalterm(text):
    doc = nlp(text)
    preserved = []
    
    for token in doc:
        if token.text.isupper() or token.text.istitle():
            preserved.append(token.text)
        else:
            preserved.append(token.text.lower())
    return ''.join(preserved)


def parse_legaldocument(text):
    
    structured_data = defaultdict(lambda: {'title': "", "sections": defaultdict(dict)})
    cleaned_text = clean_text(text)

    chapters = re.findall(Structure_pattern['chapter'], cleaned_text, re.IGNORECASE)

    for chapter_code, chapter_title in chapters:
        chapter_key = f"{chapter_code.strip().replace(' ', '_')}"
        chapter_title = chapter_title.strip()
        structured_data[chapter_key]["title"] = chapter_title

        chapter_block_regex = rf'{re.escape(chapter_code)}\s*{re.escape(chapter_title)}(.*?)(?=CHAPTER\s+[IVXLCDM]+|\Z)'
        chapter_block = re.search(chapter_block_regex, cleaned_text, re.IGNORECASE | re.DOTALL)

        if chapter_block:
            chapter_content = chapter_block.group(1)
            sections = re.findall(Structure_pattern['section'], chapter_content, re.MULTILINE | re.DOTALL)

            for section_num, section_text in sections:
                section_text = section_text.strip()
                if not section_text:
                    continue

                subsections = re.findall(Structure_pattern['subsection'], section_text, re.MULTILINE)
                if subsections:
                    structured_data[chapter_key]["sections"][section_num] = {
                        "main_text": "",
                        "subsections": {}
                    }
                    main_text, _, subs_text = section_text.partition('\n')
                    structured_data[chapter_key]["sections"][section_num]["main_text"] = preserve_legalterm(main_text.strip())
                    for sub_num, sub_text in subsections:
                        structured_data[chapter_key]["sections"][section_num]["subsections"][sub_num] = preserve_legalterm(sub_text.strip())
                else:
                    structured_data[chapter_key]["sections"][section_num] = preserve_legalterm(section_text)
    return dict(structured_data)




def flatten_sections(structured_data):
    flat_dict = {}
    for chapter, content in structured_data.items():
        for sec_num, sec_content in content['sections'].items():
            if isinstance(sec_content, dict) and "main_text" in sec_content:
                full_text = sec_content["main_text"] + "\n" + "\n".join(sec_content["subsections"].values())
            else:
                full_text = sec_content
            flat_dict[sec_num] = full_text.strip()
    return flat_dict


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    logging.info(f"Saved: {path}")