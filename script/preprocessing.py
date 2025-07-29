import os
import fitz
from tqdm.auto import tqdm

pdf_path = "U:\\LEGALSENSE\\script\\Pakistan Penal Code.pdf"


def text_formatter(text: str) -> str:
    cleaned_text = text.replace("\n", "").strip()
    return cleaned_text


def open_and_read_pdf(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages_and_text = []
    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text()
        text = text_formatter(text=text)
        pages_and_text.append({"page_number": page_number - 4,
                               "page_char_count": len(text),
                               "page_word_count": len(text.split(" ")),
                               "page_sentence_count": len(text.split(". ")),
                               "page_token_count": len(text) / 4,
                               "text": text})
    return pages_and_text


pages_and_text = open_and_read_pdf(pdf_path=pdf_path)
pages_and_text[:2]