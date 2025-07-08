import pdfplumber

pdf_path = "scripts/Pakistan Penal Code.pdf"

with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages[:3]):
        text = page.extract_text()
        print(f"\n--- Page {i+1} ---\n{text}\n")
