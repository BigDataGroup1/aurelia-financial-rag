import os, json, re, fitz, subprocess, tempfile, camelot
from pathlib import Path
from tqdm import tqdm

# ===================== CONFIG =====================
PDF_PATH = Path("data/raw_pdfs/fintbx.pdf")
OUT_DIR = Path("data/parsed_chunks")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
def run_marker(pdf_path: Path, out_dir: Path):
    try:
        subprocess.run(["marker", str(pdf_path), str(out_dir)], check=True)
        print("✅ Marker conversion complete")
        return out_dir
    except Exception as e:
        print("⚠️ Marker failed:", e)
        return None


# ==================================================
def extract_text_pymupdf(pdf_path: Path):
    """Extracts text, images, and captions using PyMuPDF."""
    print("⚙️ Extracting text + images with PyMuPDF...")
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in tqdm(enumerate(doc), total=len(doc)):
        text = page.get_text("text")

        # Detect figure/table captions
        captions = []
        for line in text.split("\n"):
            if re.match(r"^(Figure|Table)\s+\d+", line.strip(), re.IGNORECASE):
                captions.append(line.strip())

        # Detect math (LaTeX or Unicode)
        formulas = re.findall(r"(\$.*?\$|[A-Za-z]\s?=\s?[0-9A-Za-z\+\-\*/\^]+)", text)

        # Extract embedded images
        images = []
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            img_ext = base_image["ext"]
            img_path = OUT_DIR / f"page_{i+1}_img_{xref}.{img_ext}"
            with open(img_path, "wb") as f:
                f.write(img_bytes)
            images.append(str(img_path))

        pages.append({
            "page": i + 1,
            "type": "text+image",
            "content": text,
            "captions": captions,
            "formulas": formulas,
            "images": images
        })

    return pages


# ==================================================
def extract_tables(pdf_path: Path):
    """Extracts tables using Camelot (may take time on large files)."""
    print("📊 Extracting tables with Camelot (this can take time)...")
    tables_data = []
    try:
        tables = camelot.read_pdf(str(pdf_path), pages="1-end", flavor="stream")
        for i, t in enumerate(tables):
            csv_path = OUT_DIR / f"table_{i+1}.csv"
            t.to_csv(csv_path)
            tables_data.append({
                "page": t.page,
                "type": "table",
                "content": t.df.to_dict(),
                "csv_path": str(csv_path)
            })
        print(f"✅ Extracted {len(tables_data)} tables.")
    except Exception as e:
        print("⚠️ Table extraction skipped:", e)
    return tables_data


# ==================================================
def merge_results(marker_md, pymupdf_pages, tables_data):
    """Combine Markdown + text + table chunks into a single JSONL."""
    chunks = []

    # Add Marker Markdown
    if marker_md and marker_md.exists():
        chunks.append({
            "page": None,
            "type": "marker_markdown",
            "content": marker_md.read_text(encoding="utf-8")
        })

    # Add PyMuPDF pages
    chunks.extend(pymupdf_pages)

    # Add tables
    chunks.extend(tables_data)

    # Save combined JSONL
    out_path = OUT_DIR / "fintbx_chunks_enhanced.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    print(f"✅ Saved {len(chunks)} chunks to {out_path}")
    return out_path


# ==================================================
def main():
    md_path = OUT_DIR / "fintbx_marker.md"

    # Step 1 — Run Marker
    marker_out = run_marker(PDF_PATH, md_path)

    # Step 2 — Extract text + images
    pymupdf_pages = extract_text_pymupdf(PDF_PATH)

    # Step 3 — Extract tables
    tables_data = extract_tables(PDF_PATH)

    # Step 4 — Merge and save
    merge_results(marker_out, pymupdf_pages, tables_data)


# ==================================================
if __name__ == "__main__":
    main()
