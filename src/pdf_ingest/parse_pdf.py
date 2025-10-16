# src/pdf_ingest/parse_pdf.py
import os
from pathlib import Path
import csv
from typing import List

# ---- Fast settings to avoid stalls on 2.57 ----
# Force no OCR and no picture description (VLM/API) for speed/stability.
os.environ["DOCLING_OCR_ENGINE"] = "none"
os.environ["DOCLING_PICTURE_DESCRIPTIONS"] = "none"
os.environ["DOCLING_ENABLE_PICTURE_DESCRIPTION"] = "0"
# -----------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PDF = PROJECT_ROOT / "data" / "raw_pdfs" / "fintbx.pdf"
OUT_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUT_PAGES = OUT_PROCESSED / "pages"
OUT_FIGS = OUT_PROCESSED / "figures"
OUT_PROCESSED.mkdir(parents=True, exist_ok=True)
OUT_PAGES.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)

COMBINED_MD = OUT_PROCESSED / "fintbx_parsed.md"
MANIFEST_CSV = OUT_PROCESSED / "manifest.csv"

def convert_with_docling(pdf_path: Path):
    """
    Docling 2.57 conversion using the simple converter call.
    We pass a path string; the converter returns a result object.
    """
    from docling.document_converter import DocumentConverter
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    if isinstance(result, (list, tuple)):
        result = result[0]
    return result

def save_docling_images(result, out_dir: Path) -> List[str]:
    saved: List[str] = []
    # Prefer built-in saver if present (2.57 usually has it)
    if hasattr(result, "save_resources"):
        try:
            result.save_resources(str(out_dir))
            for p in sorted(out_dir.glob("*")):
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    saved.append(p.name)
            return saved
        except Exception as e:
            print(f"[WARN] save_resources failed: {e}")

    # Fallback: walk document pages for images, if exposed
    try:
        doc = getattr(result, "document", None)
        if doc and hasattr(doc, "pages"):
            counter = 0
            for pi, page in enumerate(doc.pages):
                imgs = getattr(page, "images", []) or []
                for img in imgs:
                    counter += 1
                    name = f"page_{pi+1:03d}_img_{counter}.png"
                    path = out_dir / name
                    # try Pillow image-like
                    if hasattr(img, "save"):
                        img.save(path)
                        saved.append(name)
                    # try raw bytes
                    elif isinstance(img, (bytes, bytearray)):
                        path.write_bytes(img)
                        saved.append(name)
    except Exception as e:
        print(f"[WARN] Could not extract images via fallback: {e}")
    return saved

def render_page_markdown(result, page_idx: int) -> str:
    """
    Render one page to Markdown with broad compatibility across 2.57 variants.
    """
    # Some 2.5x builds expose a renderer:
    if hasattr(result, "render_page_markdown"):
        try:
            md = result.render_page_markdown(page_idx)
            return md.strip()
        except Exception:
            pass

    # Fallback: iterate structured document and assemble basic MD
    doc = getattr(result, "document", None)
    parts: List[str] = [f"<!-- Page: {page_idx+1} -->", f"### Page {page_idx+1}", ""]
    if not doc or not hasattr(doc, "pages"):
        return "\n".join(parts + ["_No structured content available on this page._"])

    page = None
    try:
        page = doc.pages[page_idx]
    except Exception:
        return "\n".join(parts + ["_Page out of range or unavailable._"])

    # Best-effort text export if provided
    for attr in ("to_markdown", "export_markdown", "export_text", "to_text"):
        if hasattr(page, attr):
            try:
                content = getattr(page, attr)()
                if content:
                    parts.append(content.strip())
                    return "\n\n".join(parts)
            except Exception:
                pass

    # Otherwise, walk blocks/spans if present
    blocks = getattr(page, "blocks", None) or []
    for blk in blocks:
        # block-level markdown if available
        if hasattr(blk, "to_markdown"):
            try:
                parts.append(blk.to_markdown().strip())
                continue
            except Exception:
                pass
        # else try text/span aggregation
        text = getattr(blk, "text", "") or ""
        if not text:
            spans = getattr(blk, "spans", None) or []
            if spans:
                text = " ".join([getattr(s, "text", "").strip() for s in spans if getattr(s, "text", "").strip()])
        if text:
            parts.append(text)

    return "\n\n".join([p for p in parts if p])

def stitch_combined_md(page_files: List[Path]) -> str:
    parts = ["# Financial Toolbox — Parsed (Docling 2.57, no OCR)",
             "_Source: fintbx.pdf_",
             ""]
    for pf in page_files:
        parts.append(pf.read_text(encoding="utf-8"))
        parts.append("")  # spacer
    return "\n".join(parts)

def main():
    if not RAW_PDF.exists():
        raise FileNotFoundError(f"Missing PDF at {RAW_PDF}. Put fintbx.pdf under data/raw_pdfs/")

    print(f"[Docling] Converting: {RAW_PDF}")
    result = convert_with_docling(RAW_PDF)

    saved_imgs = save_docling_images(result, OUT_FIGS)
    if saved_imgs:
        print(f"[Docling] Saved {len(saved_imgs)} images to {OUT_FIGS}")

    # Page count across variants
    num_pages = None
    for attr in ("num_pages",):
        if hasattr(result, attr):
            num_pages = getattr(result, attr)
            break
    if num_pages is None:
        doc = getattr(result, "document", None)
        if doc and hasattr(doc, "pages"):
            num_pages = len(doc.pages)
    if not num_pages:
        raise RuntimeError("Could not determine page count from Docling result.")

    # Optional: first run on a small range to confirm
    max_pages_env = os.environ.get("PARSE_MAX_PAGES")
    if max_pages_env:
        try:
            num_pages = min(num_pages, int(max_pages_env))
            print(f"[Docling] Limiting to first {num_pages} page(s) due to PARSE_MAX_PAGES.")
        except Exception:
            pass

    print(f"[Docling] Rendering {num_pages} pages to Markdown…")
    written_pages: List[Path] = []
    manifest_rows = []

    for i in range(num_pages):
        md = render_page_markdown(result, i)
        page_fname = OUT_PAGES / f"page_{i+1:03d}.md"
        page_fname.write_text(md, encoding="utf-8")
        written_pages.append(page_fname)
        manifest_rows.append({
            "page": i + 1,
            "path": str(page_fname.relative_to(PROJECT_ROOT)),
            "bytes": page_fname.stat().st_size,
        })

    combined = stitch_combined_md(written_pages)
    COMBINED_MD.write_text(combined, encoding="utf-8")
    print(f"[Docling] Wrote combined Markdown: {COMBINED_MD}")

    with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["page", "path", "bytes"])
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)
    print(f"[Docling] Wrote manifest: {MANIFEST_CSV}")

if __name__ == "__main__":
    main()
