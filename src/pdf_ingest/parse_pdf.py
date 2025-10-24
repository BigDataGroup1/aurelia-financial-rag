"""
Complete PDF Parser for Lab 1 - AURELIA Assignment
Extracts: text, code, tables, figures, formulas, captions
Preserves: reading order, structure, citations
OPTIMIZED: Writes pages immediately + skips problematic pages
HARD TIMEOUT: Per-page text+tables extracted via an internal subprocess mode
"""

# ---------------------------
# Minimal single-page worker
# ---------------------------
# We purposely put this BEFORE any logging/env/directory setup so that when the
# file is invoked with --single-page, it prints *only JSON* to stdout and exits.

import sys as _sys, json as _json

def _single_page_entrypoint_if_requested():
    """
    If invoked as:
        python parse_pdf.py --single-page <page_index_zero_based> --pdf <pdf_path>
    run a tiny, log-free extraction of text+tables for that single page and exit.
    """
    argv = _sys.argv
    if len(argv) >= 2 and argv[1] == "--single-page":
        try:
            # Parse args: --single-page <idx> --pdf <path>
            idx = int(argv[2])
            if len(argv) >= 5 and argv[3] == "--pdf":
                pdf_path = argv[4]
            else:
                print(_json.dumps({"ok": False, "err": "missing --pdf"}))
                _sys.exit(0)

            try:
                import pdfplumber
                with pdfplumber.open(pdf_path) as _pdf:
                    page = _pdf.pages[idx]

                    # Text
                    try:
                        text = page.extract_text() or ""
                    except Exception:
                        text = ""

                    # Tables: try strict (lines) then text strategies
                    tables = []
                    try:
                        strict = {
                            "vertical_strategy": "lines",
                            "horizontal_strategy": "lines",
                            "snap_tolerance": 3,
                            "join_tolerance": 3,
                        }
                        tables = page.extract_tables(table_settings=strict) or []
                        if not tables:
                            text_set = {"vertical_strategy": "text", "horizontal_strategy": "text"}
                            tables = page.extract_tables(table_settings=text_set) or []
                    except Exception:
                        tables = []

                    print(_json.dumps({"ok": True, "text": text, "tables": tables}))
            except Exception as e:
                print(_json.dumps({"ok": False, "err": str(e)}))
        except Exception as e:
            print(_json.dumps({"ok": False, "err": str(e)}))
        _sys.exit(0)

_single_page_entrypoint_if_requested()

# ---------------------------
# Normal module imports/setup
# ---------------------------
import os
import re
from pathlib import Path
import csv
from typing import List, Dict, Tuple
import logging
import json
import subprocess
import sys

from dotenv import load_dotenv

# Load env (harmless in Composer)
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Hard timeout for a single page's text+tables extraction (seconds).
PAGE_TIMEOUT_SEC = int(os.environ.get("PARSE_PAGE_TIMEOUT_SEC", "20"))

# ---------------------------
# Paths (unchanged semantics)
# ---------------------------
# Keep your original PROJECT_ROOT scheme so outputs/paths remain identical
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PDF = PROJECT_ROOT / "data" / "raw_pdfs" / "fintbx.pdf"
OUT_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUT_PAGES = OUT_PROCESSED / "pages"
OUT_FIGS = OUT_PROCESSED / "figures"
OUT_META = OUT_PROCESSED / "metadata"

OUT_PROCESSED.mkdir(parents=True, exist_ok=True)
OUT_PAGES.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)
OUT_META.mkdir(parents=True, exist_ok=True)

COMBINED_MD = OUT_PROCESSED / "fintbx_parsed.md"
MANIFEST_CSV = OUT_PROCESSED / "manifest.csv"

# ---------------------------
# Detectors (unchanged logic)
# ---------------------------

def detect_code_blocks(text: str) -> List[Tuple[int, int, str]]:
    """
    Detect code blocks in text (MATLAB patterns)
    Returns: List of (start_pos, end_pos, code_content)
    """
    code_blocks = []
    lines = text.split('\n')

    in_code = False
    code_start = 0
    code_lines = []

    for i, line in enumerate(lines):
        is_code = (
            line.strip().startswith('>>') or
            line.strip().startswith('%') or
            line.strip().startswith('function ') or
            re.match(r'^\s*(for|if|while|switch)\s+', line) or
            (line.strip() == 'end' and in_code) or
            (in_code and re.match(r'^\s+', line) and len(line.strip()) > 0)
        )

        if is_code and not in_code:
            in_code = True
            code_start = i
            code_lines = [line]
        elif is_code and in_code:
            code_lines.append(line)
        elif not is_code and in_code:
            in_code = False
            code_blocks.append((code_start, i - 1, '\n'.join(code_lines)))
            code_lines = []

    if in_code and code_lines:
        code_blocks.append((code_start, len(lines) - 1, '\n'.join(code_lines)))

    return code_blocks


def detect_formulas(text: str) -> List[str]:
    """
    Detect mathematical formulas/equations
    Returns: List of formula strings
    """
    formulas = []

    # Pattern 1: Variable = Expression
    pattern1 = r'[A-Z][a-zA-Z]*\s*=\s*[^=\n]{5,}'
    formulas.extend(re.findall(pattern1, text))

    # Pattern 2: Greek letters or math symbols
    math_symbols = r'[αβγδεζηθλμπρστφχψω∑∏∫√±≤≥≠≈]'
    if re.search(math_symbols, text):
        lines = text.split('\n')
        formulas.extend([line.strip() for line in lines if re.search(math_symbols, line)])

    # Pattern 3: Parentheses-heavy expressions
    pattern3 = r'\([^)]+\([^)]+\)[^)]*\)'
    formulas.extend(re.findall(pattern3, text))

    return list(set(formulas))[:10]  # unique + limit


def extract_figure_captions(text: str) -> List[Dict]:
    """
    Extract figure captions and references
    Returns: List of {caption, figure_num, context}
    """
    captions = []
    pattern = r'(Figure|Fig\.)\s+(\d+[-\d]*)[:\.]?\s*([^\n]+)'
    matches = re.finditer(pattern, text, re.IGNORECASE)

    for match in matches:
        captions.append({
            'type': 'figure',
            'number': match.group(2),
            'caption': match.group(3).strip(),
            'full_text': match.group(0)
        })

    return captions


def extract_table_captions(text: str) -> List[Dict]:
    """
    Extract table captions and references
    """
    captions = []
    pattern = r'Table\s+(\d+[-\d]*)[:\.]?\s*([^\n]+)'
    matches = re.finditer(pattern, text, re.IGNORECASE)

    for match in matches:
        captions.append({
            'type': 'table',
            'number': match.group(1),
            'caption': match.group(2).strip(),
            'full_text': match.group(0)
        })

    return captions


def format_page_as_markdown(page_data: Dict) -> str:
    """
    Convert page data to comprehensive markdown
    Preserves reading order and includes all extracted elements
    """
    page_num = page_data['page_num']
    text = page_data['text']
    tables = page_data['tables']
    images = page_data['images']
    code_blocks = page_data['code_blocks']
    formulas = page_data['formulas']
    figure_captions = page_data['figure_captions']
    table_captions = page_data['table_captions']

    parts = [
        f"<!-- Page: {page_num} -->",
        f"### Page {page_num}",
        ""
    ]

    # Metadata badges
    metadata = []
    if images:
        metadata.append(f"📷 {len(images)} image(s)")
    if tables:
        metadata.append(f"📊 {len(tables)} table(s)")
    if code_blocks:
        metadata.append(f"💻 {len(code_blocks)} code block(s)")
    if formulas:
        metadata.append(f"🔢 {len(formulas)} formula(s)")

    if metadata:
        parts.append(f"**Content:** {' | '.join(metadata)}")
        parts.append("")

    if figure_captions:
        parts.append("**Figures on this page:**")
        for cap in figure_captions:
            parts.append(f"- Figure {cap['number']}: {cap['caption']}")
        parts.append("")

    if images:
        parts.append("**Images:**")
        for img in images:
            parts.append(f"![Image {img['index']}](../figures/{img['filename']})")
            parts.append(f"*File: {img['filename']} ({img['size'] / 1024:.1f} KB)*")
            parts.append("")

    if text and text.strip():
        lines = text.split('\n')
        processed_lines = []

        current_line = 0
        for cb_start, cb_end, cb_content in code_blocks:
            while current_line < cb_start:
                processed_lines.append(lines[current_line])
                current_line += 1

            processed_lines.append("")
            processed_lines.append("```matlab")
            processed_lines.append(cb_content)
            processed_lines.append("```")
            processed_lines.append("")
            current_line = cb_end + 1

        while current_line < len(lines):
            processed_lines.append(lines[current_line])
            current_line += 1

        parts.append('\n'.join(processed_lines).strip())
        parts.append("")

    if formulas:
        parts.append("**Detected Formulas:**")
        for formula in formulas[:5]:
            parts.append(f"- `{formula.strip()}`")
        parts.append("")

    if tables:
        for idx, table in enumerate(tables, 1):
            if not table or len(table) == 0:
                continue
            has_content = any(any(cell and str(cell).strip() for cell in row) for row in table)
            if not has_content:
                continue

            caption = None
            for tc in table_captions:
                if str(idx) in tc['number']:
                    caption = tc['caption']
                    break

            if caption:
                parts.append(f"**Table {idx}: {caption}**")
            else:
                parts.append(f"**Table {idx} (Page {page_num}):**")
            parts.append("")

            header = table[0] if len(table) > 0 else []
            if header:
                header_cells = [str(cell or "").strip() for cell in header]
                parts.append("| " + " | ".join(header_cells) + " |")
                parts.append("|" + "|".join(["---"] * len(header_cells)) + "|")

                for row in table[1:]:
                    row_cells = [str(cell or "").strip() for cell in row]
                    parts.append("| " + " | ".join(row_cells) + " |")

                parts.append("")

    return "\n".join(parts)

# --------------------------------------------------------
# Subprocess wrapper (Airflow-safe, no multiprocessing)
# --------------------------------------------------------

def _extract_text_and_tables_via_subprocess(pdf_path_str: str, page_index: int, timeout_sec: int):
    """
    Re-invokes this file in --single-page mode to extract one page with a hard timeout.
    Returns (text, tables) or ("", []) on timeout/error. Prints no logs in worker mode.
    """
    try:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--single-page", str(page_index),
            "--pdf", pdf_path_str,
        ]
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
            text=True,
        )
        if res.returncode != 0:
            # Not fatal—treat as empty and continue
            return "", []
        try:
            msg = json.loads(res.stdout.strip() or "{}")
        except Exception:
            return "", []
        if msg.get("ok"):
            return msg.get("text", ""), msg.get("tables", [])
        return "", []
    except subprocess.TimeoutExpired:
        return "", []
    except Exception:
        return "", []

# ---------------------------
# Main extraction
# ---------------------------

def extract_with_pdfplumber(pdf_path: Path):
    """Main extraction using pdfplumber + PyMuPDF (images in parent; text/tables via timed subprocess)"""
    try:
        import pdfplumber  # noqa: F401
    except ImportError:
        logger.error("pdfplumber not installed. Run: pip install pdfplumber")
        raise

    # Try to import PyMuPDF for images
    try:
        import fitz
        has_pymupdf = True
        logger.info("PyMuPDF available for image extraction")
    except ImportError:
        has_pymupdf = False
        logger.warning("PyMuPDF not installed. Images will not be extracted. Run: pip install pymupdf")

    logger.info(f"Opening PDF: {pdf_path}")
    try:
        size_mb = pdf_path.stat().st_size / 1e6
        logger.info(f"PDF size: {size_mb:.2f} MB")
    except Exception:
        pass

    # Open PyMuPDF document ONCE for all pages (image extraction in parent)
    pymupdf_doc = None
    if has_pymupdf:
        try:
            pymupdf_doc = fitz.open(str(pdf_path))
        except Exception as e:
            logger.warning(f"Could not open PDF with PyMuPDF: {e}")
            has_pymupdf = False

    pages_data: List[Dict] = []

    # Open with pdfplumber for page count / iteration
    import pdfplumber as _pp
    with _pp.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        logger.info(f"Total pages in PDF: {total_pages}")

        # Page selection logic (unchanged)
        page_range_env = os.getenv("PARSE_PAGE_RANGE", "").strip()
        max_pages_env = os.environ.get("PARSE_MAX_PAGES")

        if page_range_env:
            start, end = map(int, page_range_env.split("-"))
            start = max(1, min(start, total_pages))
            end = min(end, total_pages)
            logger.info(f"Processing pages {start} to {end}")
            page_indices = range(start - 1, end)
        elif max_pages_env:
            cap = min(int(max_pages_env), total_pages)
            logger.info(f"Processing first {cap} pages")
            page_indices = range(cap)
        else:
            page_indices = range(total_pages)

        logger.info(f"\n{'=' * 70}")
        logger.info("Starting page-by-page extraction...")
        logger.info('-' * 70)

        for i in page_indices:
            page_num = i + 1
            try:
                # -------------------------------
                # Text + Tables via timed subprocess (no hangs)
                # -------------------------------
                text, tables = _extract_text_and_tables_via_subprocess(str(pdf_path), i, PAGE_TIMEOUT_SEC)
                if text == "" and tables == []:
                    logger.warning(f"⏱️ Page {page_num}: text/tables empty (timeout, worker error, or no content)")

                # -------------------------------
                # Images via PyMuPDF in parent
                # -------------------------------
                images = []
                if has_pymupdf and pymupdf_doc:
                    try:
                        pymupdf_page = pymupdf_doc[i]
                        image_list = pymupdf_page.get_images(full=True)
                        for img_index, img in enumerate(image_list):
                            try:
                                xref = img[0]
                                base_image = pymupdf_doc.extract_image(xref)
                                image_bytes = base_image["image"]
                                image_ext = base_image["ext"]

                                image_filename = f"page_{page_num:03d}_img_{img_index + 1}.{image_ext}"
                                image_path = OUT_FIGS / image_filename
                                image_path.write_bytes(image_bytes)

                                images.append({
                                    'filename': image_filename,
                                    'page': page_num,
                                    'index': img_index + 1,
                                    'size': len(image_bytes),
                                    'format': image_ext
                                })
                            except Exception:
                                pass
                    except Exception as e:
                        logger.debug(f"Image extraction failed for page {page_num}: {e}")

                # -------------------------------
                # Heuristics on the extracted text
                # -------------------------------
                try:
                    code_blocks = detect_code_blocks(text)
                except Exception:
                    code_blocks = []

                try:
                    formulas = detect_formulas(text)
                except Exception:
                    formulas = []

                try:
                    figure_captions = extract_figure_captions(text)
                    table_captions = extract_table_captions(text)
                except Exception:
                    figure_captions = []
                    table_captions = []

                # -------------------------------
                # Write per-page markdown immediately
                # -------------------------------
                page_data = {
                    'page_num': page_num,
                    'text': text,
                    'tables': tables,
                    'images': images,
                    'code_blocks': code_blocks,
                    'formulas': formulas,
                    'figure_captions': figure_captions,
                    'table_captions': table_captions,
                    'char_count': len(text)
                }

                md_content = format_page_as_markdown(page_data)
                page_fname = OUT_PAGES / f"page_{page_num:04d}.md"
                page_fname.write_text(md_content, encoding="utf-8")

                # Lightweight metadata in memory
                pages_data.append({
                    'page_num': page_num,
                    'char_count': len(text),
                    'images': len(images),
                    'tables': len(tables),
                    'code_blocks': len(code_blocks),
                    'formulas': len(formulas),
                    'figure_captions': len(figure_captions),
                    'table_captions': len(table_captions)
                })

                if (page_num) % 10 == 0 or len(page_indices) <= 20:
                    logger.info(
                        f"✓ Page {page_num}/{len(page_indices)}: "
                        f"{len(text)} chars, {len(tables)} tables, {len(images)} images, "
                        f"{len(code_blocks)} code blocks, {len(formulas)} formulas"
                    )

            except Exception as e:
                logger.error(f"❌ ERROR processing page {page_num}: {e}")
                # Always write a placeholder so combined output stays aligned
                page_fname = OUT_PAGES / f"page_{page_num:04d}.md"
                page_fname.write_text(
                    f"<!-- Page {page_num}: Error during extraction -->\n"
                    f"### Page {page_num}\n\n"
                    f"**⚠️ Extraction error on this page**\n",
                    encoding="utf-8"
                )
                pages_data.append({
                    'page_num': page_num,
                    'char_count': 0,
                    'images': 0,
                    'tables': 0,
                    'code_blocks': 0,
                    'formulas': 0,
                    'figure_captions': 0,
                    'table_captions': 0
                })
                continue

    # Close PyMuPDF document
    if pymupdf_doc:
        try:
            pymupdf_doc.close()
        except Exception:
            pass

    return pages_data

# ---------------------------
# Output writers (unchanged)
# ---------------------------

def save_metadata(pages_data: List[Dict]):
    """Save extraction metadata as JSON"""
    metadata = {
        'total_pages': len(pages_data),
        'total_images': sum(p['images'] for p in pages_data),
        'total_tables': sum(p['tables'] for p in pages_data),
        'total_code_blocks': sum(p['code_blocks'] for p in pages_data),
        'total_formulas': sum(p['formulas'] for p in pages_data),
        'total_chars': sum(p['char_count'] for p in pages_data),
        'pages': [
            {
                'page': p['page_num'],
                'chars': p['char_count'],
                'images': p['images'],
                'tables': p['tables'],
                'code_blocks': p['code_blocks'],
                'formulas': p['formulas'],
                'figure_captions': p['figure_captions'],
                'table_captions': p['table_captions']
            }
            for p in pages_data
        ]
    }

    metadata_file = OUT_META / "extraction_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✓ Saved metadata: {metadata_file}")

# ---------------------------
# Main
# ---------------------------

def main():
    logger.info("=" * 70)
    logger.info("COMPLETE PDF PARSER - Lab 1 Requirements")
    logger.info("=" * 70)

    if not RAW_PDF.exists():
        logger.error(f"PDF not found: {RAW_PDF}")
        logger.error(f"Expected location: {RAW_PDF}")
        return

    # Extract all content
    try:
        pages_data = extract_with_pdfplumber(RAW_PDF)
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    if not pages_data:
        logger.error("No pages extracted!")
        return

    logger.info(f"\n{'=' * 70}")
    logger.info("Writing output files...")
    logger.info('-' * 70)

    # Pages already written during extraction - just build manifest
    logger.info(f"✓ All {len(pages_data)} pages written during extraction")

    written_pages = sorted(OUT_PAGES.glob("page_*.md"))
    manifest_rows = []

    for page_meta in pages_data:
        page_num = page_meta['page_num']
        page_fname = OUT_PAGES / f"page_{page_num:04d}.md"

        if page_fname.exists():
            manifest_rows.append({
                "page": page_num,
                "path": str(page_fname.relative_to(PROJECT_ROOT)),
                "bytes": page_fname.stat().st_size,
                "images": page_meta['images'],
                "tables": page_meta['tables'],
                "code_blocks": page_meta['code_blocks']
            })

    logger.info(f"✓ Built manifest for {len(written_pages)} page files")

    # Combined markdown from disk
    combined_parts = [
        "# Financial Toolbox — Complete Extraction",
        "_Source: fintbx.pdf_",
        "",
        "**Extraction Summary:**",
        f"- Total Pages: {len(pages_data)}",
        f"- Total Characters: {sum(p['char_count'] for p in pages_data):,}",
        f"- Total Images: {sum(p['images'] for p in pages_data)}",
        f"- Total Tables: {sum(p['tables'] for p in pages_data)}",
        f"- Total Code Blocks: {sum(p['code_blocks'] for p in pages_data)}",
        f"- Total Formulas: {sum(p['formulas'] for p in pages_data)}",
        "",
        "---",
        ""
    ]

    logger.info("Building combined markdown from disk files...")
    for page_file in written_pages:
        try:
            combined_parts.append(page_file.read_text(encoding="utf-8"))
            combined_parts.append("")
        except Exception as e:
            logger.warning(f"Could not read {page_file.name}: {e}")

    combined = "\n".join(combined_parts)
    COMBINED_MD.write_text(combined, encoding="utf-8")
    logger.info(f"✓ Wrote combined markdown: {COMBINED_MD} ({len(combined)} chars)")

    # Write manifest
    with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["page", "path", "bytes", "images", "tables", "code_blocks"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    logger.info(f"✓ Wrote manifest: {MANIFEST_CSV}")

    # Save metadata
    save_metadata(pages_data)

    # Final summary
    logger.info(f"\n{'=' * 70}")
    logger.info("✓ EXTRACTION COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"📄 Pages: {len(pages_data)}")
    logger.info(f"📊 Tables: {sum(p['tables'] for p in pages_data)}")
    logger.info(f"📷 Images: {sum(p['images'] for p in pages_data)}")
    logger.info(f"💻 Code Blocks: {sum(p['code_blocks'] for p in pages_data)}")
    logger.info(f"🔢 Formulas: {sum(p['formulas'] for p in pages_data)}")
    logger.info(f"\n📁 Output: {OUT_PROCESSED}")
    logger.info(f"   - Pages: {OUT_PAGES}")
    logger.info(f"   - Figures: {OUT_FIGS}")
    logger.info(f"   - Metadata: {OUT_META}")
    logger.info(f"\nVerify output:")
    logger.info(f"   ls {OUT_PAGES}")
    logger.info(f"   cat {OUT_PAGES}/page_0001.md")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
