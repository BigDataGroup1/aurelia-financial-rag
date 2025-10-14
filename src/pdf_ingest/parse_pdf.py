# src/pdf_ingest/parse_pdf.py
import fitz
import re
import os
from typing import List, Dict

def parse_pdf(pdf_path: str) -> List[Dict]:
    """Extract text from PDF with page numbers and headings."""
    doc = fitz.open(pdf_path)
    parsed = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        # Basic cleanup
        text = re.sub(r'\s+', ' ', text).strip()
        parsed.append({
            "page": page_num,
            "text": text
        })
    return parsed

if __name__ == "__main__":
    import json
    data = parse_pdf("data/raw_pdfs/fintbx.pdf")
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/fintbx_parsed.json", "w") as f:
        json.dump(data, f, indent=2)
    print("✅ Parsed PDF saved to data/processed/fintbx_parsed.json")
