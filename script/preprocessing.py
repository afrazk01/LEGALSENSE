import os
import fitz
from tqdm.auto import tqdm

pdf_path = "U:\\LEGALSENSE\\script\\Pakistan Penal Code.pdf"


def text_formatter(text: str) -> str:
    cleaned_text = text.replace("\n", "").strip()
    return cleaned_text

