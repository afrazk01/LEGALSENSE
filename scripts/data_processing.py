import PyPDF2 , re
import json
from pathlib import Path
import spacy

nlp = spacy.load("en_core_web_sm")


def extract_text_from_pdf(pdf_path):
    text = ''
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text


def lemmatize_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)


def preprocess_ppc(text):
    text = re.sub(r'===== Page \d+ =====', '', text)

    section = re.split(r'\n(\d+[A-Z]*-*[\s\S]*?\.)', text)

    structured_data = {}
    current_section = None

    for item in section[1:]:
        if item.strip().isdigit() or re.match(r'\d+[A-Z]*-*', item.strip()):
            current_section = item.strip()
        elif current_section:
            cleaned_text = lemmatize_text(item.strip())
            structured_data[current_section] = cleaned_text
    return structured_data

def save_ppc_json(structured_ppc):
    output_path = Path(__file__).parent.parent / "data/processed/ppc_section.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving JSON to: {output_path.resolve()}")
    with open(output_path, 'w') as f:
        json.dump(structured_ppc, f, indent=4)


pdf_text = extract_text_from_pdf('scripts/Pakistan Penal Code.pdf')
structured_ppc = preprocess_ppc(pdf_text)
save_ppc_json(structured_ppc)



