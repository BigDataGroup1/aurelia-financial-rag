"""
Complete PDF Parser for Lab 1 - AURELIA Assignment
Extracts: text, code, tables, figures, formulas, captions
Preserves: reading order, structure, citations
"""
import os
import re
from pathlib import Path
import csv
from typing import List, Dict, Tuple
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

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


def extract_images_pymupdf(pdf_path: Path, page_num: int, output_dir: Path) -> List[Dict]:
    """Extract images from a specific page using PyMuPDF"""
    try:
        import fitz
    except ImportError:
        logger.warning("PyMuPDF not installed. Images will not be extracted. Run: pip install pymupdf")
        return []
    
    doc = fitz.open(str(pdf_path))
    page = doc[page_num]
    image_list = page.get_images(full=True)
    
    extracted_images = []
    
    for img_index, img in enumerate(image_list):
        try:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Save image
            image_filename = f"page_{page_num + 1:03d}_img_{img_index + 1}.{image_ext}"
            image_path = output_dir / image_filename
            image_path.write_bytes(image_bytes)
            
            extracted_images.append({
                'filename': image_filename,
                'page': page_num + 1,
                'index': img_index + 1,
                'size': len(image_bytes),
                'format': image_ext
            })
            
        except Exception as e:
            logger.debug(f"Could not extract image {img_index} from page {page_num + 1}: {e}")
    
    doc.close()
    return extracted_images


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
        # MATLAB code indicators
        is_code = (
            line.strip().startswith('>>') or
            line.strip().startswith('%') or
            line.strip().startswith('function ') or
            re.match(r'^\s*(for|if|while|switch)\s+', line) or
            (line.strip() == 'end' and in_code) or
            (in_code and re.match(r'^\s+', line) and len(line.strip()) > 0)
        )
        
        if is_code and not in_code:
            # Start of code block
            in_code = True
            code_start = i
            code_lines = [line]
        elif is_code and in_code:
            # Continue code block
            code_lines.append(line)
        elif not is_code and in_code:
            # End of code block
            in_code = False
            code_blocks.append((code_start, i - 1, '\n'.join(code_lines)))
            code_lines = []
    
    # Handle code block at end of text
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
    
    return list(set(formulas))[:10]  # Return unique, limit to 10


def extract_figure_captions(text: str) -> List[Dict]:
    """
    Extract figure captions and references
    Returns: List of {caption, figure_num, context}
    """
    captions = []
    
    # Pattern: "Figure X: Caption" or "Fig. X: Caption"
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
    
    # Pattern: "Table X: Caption"
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


def extract_with_pdfplumber(pdf_path: Path):
    """Main extraction using pdfplumber"""
    try:
        import pdfplumber
    except ImportError:
        logger.error("pdfplumber not installed. Run: pip install pdfplumber")
        raise
    
    logger.info(f"Opening PDF: {pdf_path}")
    logger.info(f"PDF size: {pdf_path.stat().st_size / 1e6:.2f} MB")
    
    pages_data = []
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        logger.info(f"Total pages in PDF: {total_pages}")
        
        # Handle page limits
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
        
        for i in page_indices:
            page = pdf.pages[i]
            
            # Extract text
            text = page.extract_text() or ""
            
            # Extract tables with aggressive settings
            table_settings_strict = {
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "snap_tolerance": 3,
                "join_tolerance": 3,
            }
            
            tables = page.extract_tables(table_settings=table_settings_strict) or []
            
            # Try text-based if no tables found
            if not tables:
                table_settings_text = {
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                }
                tables = page.extract_tables(table_settings=table_settings_text) or []
            
            # Extract images using PyMuPDF
            images = extract_images_pymupdf(pdf_path, i, OUT_FIGS)
            
            # Detect code blocks
            code_blocks = detect_code_blocks(text)
            
            # Detect formulas
            formulas = detect_formulas(text)
            
            # Extract captions
            figure_captions = extract_figure_captions(text)
            table_captions = extract_table_captions(text)
            
            pages_data.append({
                'page_num': i + 1,
                'text': text,
                'tables': tables,
                'images': images,
                'code_blocks': code_blocks,
                'formulas': formulas,
                'figure_captions': figure_captions,
                'table_captions': table_captions,
                'char_count': len(text)
            })
            
            logger.info(
                f"✓ Page {i+1}/{len(page_indices)}: "
                f"{len(text)} chars, {len(tables)} tables, {len(images)} images, "
                f"{len(code_blocks)} code blocks, {len(formulas)} formulas"
            )
    
    return pages_data


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
    
    # Add metadata
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
    
    # Add figure captions at top for visibility
    if figure_captions:
        parts.append("**Figures on this page:**")
        for cap in figure_captions:
            parts.append(f"- Figure {cap['number']}: {cap['caption']}")
        parts.append("")
    
    # Add images with references
    if images:
        parts.append("**Images:**")
        for img in images:
            parts.append(f"![Image {img['index']}]({OUT_FIGS.name}/{img['filename']})")
            parts.append(f"*File: {img['filename']} ({img['size'] / 1024:.1f} KB)*")
            parts.append("")
    
    # Process text with code blocks
    if text and text.strip():
        lines = text.split('\n')
        processed_lines = []
        code_block_indices = {(cb[0], cb[1]) for cb in code_blocks}
        
        current_line = 0
        for cb_start, cb_end, cb_content in code_blocks:
            # Add text before code block
            while current_line < cb_start:
                processed_lines.append(lines[current_line])
                current_line += 1
            
            # Add code block
            processed_lines.append("")
            processed_lines.append("```matlab")
            processed_lines.append(cb_content)
            processed_lines.append("```")
            processed_lines.append("")
            
            current_line = cb_end + 1
        
        # Add remaining text
        while current_line < len(lines):
            processed_lines.append(lines[current_line])
            current_line += 1
        
        parts.append('\n'.join(processed_lines).strip())
        parts.append("")
    
    # Add detected formulas
    if formulas:
        parts.append("**Detected Formulas:**")
        for formula in formulas[:5]:  # Limit to 5
            parts.append(f"- `{formula.strip()}`")
        parts.append("")
    
    # Add tables
    if tables:
        for idx, table in enumerate(tables, 1):
            if not table or len(table) == 0:
                continue
            
            # Check if table has content
            has_content = any(any(cell and str(cell).strip() for cell in row) for row in table)
            if not has_content:
                continue
            
            # Find corresponding caption
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
            
            # Convert to markdown table
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


def save_metadata(pages_data: List[Dict]):
    """Save extraction metadata as JSON"""
    metadata = {
        'total_pages': len(pages_data),
        'total_images': sum(len(p['images']) for p in pages_data),
        'total_tables': sum(len(p['tables']) for p in pages_data),
        'total_code_blocks': sum(len(p['code_blocks']) for p in pages_data),
        'total_formulas': sum(len(p['formulas']) for p in pages_data),
        'total_chars': sum(p['char_count'] for p in pages_data),
        'pages': [
            {
                'page': p['page_num'],
                'chars': p['char_count'],
                'images': len(p['images']),
                'tables': len(p['tables']),
                'code_blocks': len(p['code_blocks']),
                'formulas': len(p['formulas']),
                'figure_captions': len(p['figure_captions']),
                'table_captions': len(p['table_captions'])
            }
            for p in pages_data
        ]
    }
    
    metadata_file = OUT_META / "extraction_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✓ Saved metadata: {metadata_file}")


def main():
    logger.info("="*70)
    logger.info("COMPLETE PDF PARSER - Lab 1 Requirements")
    logger.info("="*70)
    
    if not RAW_PDF.exists():
        logger.error(f"PDF not found: {RAW_PDF}")
        return
    
    # Extract all content
    try:
        pages_data = extract_with_pdfplumber(RAW_PDF)
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise
    
    if not pages_data:
        logger.error("No pages extracted!")
        return
    
    logger.info(f"\n{'='*70}")
    logger.info("Writing output files...")
    logger.info('-'*70)
    
    # Save individual page markdowns
    written_pages: List[Path] = []
    manifest_rows = []
    
    for page_data in pages_data:
        md_content = format_page_as_markdown(page_data)
        
        page_num = page_data['page_num']
        page_fname = OUT_PAGES / f"page_{page_num:03d}.md"
        page_fname.write_text(md_content, encoding="utf-8")
        
        written_pages.append(page_fname)
        manifest_rows.append({
            "page": page_num,
            "path": str(page_fname.relative_to(PROJECT_ROOT)),
            "bytes": page_fname.stat().st_size,
            "images": len(page_data['images']),
            "tables": len(page_data['tables']),
            "code_blocks": len(page_data['code_blocks'])
        })
    
    logger.info(f"✓ Wrote {len(written_pages)} page files")
    
    # Create combined markdown
    combined_parts = [
        "# Financial Toolbox — Complete Extraction",
        "_Source: fintbx.pdf_",
        "",
        "**Extraction Summary:**",
        f"- Total Pages: {len(pages_data)}",
        f"- Total Characters: {sum(p['char_count'] for p in pages_data):,}",
        f"- Total Images: {sum(len(p['images']) for p in pages_data)}",
        f"- Total Tables: {sum(len(p['tables']) for p in pages_data)}",
        f"- Total Code Blocks: {sum(len(p['code_blocks']) for p in pages_data)}",
        f"- Total Formulas: {sum(len(p['formulas']) for p in pages_data)}",
        "",
        "---",
        ""
    ]
    
    for page_file in written_pages:
        combined_parts.append(page_file.read_text(encoding="utf-8"))
        combined_parts.append("")
    
    combined = "\n".join(combined_parts)
    COMBINED_MD.write_text(combined, encoding="utf-8")
    logger.info(f"✓ Wrote combined markdown: {COMBINED_MD}")
    
    # Write manifest
    with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["page", "path", "bytes", "images", "tables", "code_blocks"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    
    logger.info(f"✓ Wrote manifest: {MANIFEST_CSV}")
    
    # Save metadata
    save_metadata(pages_data)
    
    # Final summary
    logger.info(f"\n{'='*70}")
    logger.info("✓ EXTRACTION COMPLETE!")
    logger.info("="*70)
    logger.info(f"📄 Pages: {len(pages_data)}")
    logger.info(f"📊 Tables: {sum(len(p['tables']) for p in pages_data)}")
    logger.info(f"📷 Images: {sum(len(p['images']) for p in pages_data)}")
    logger.info(f"💻 Code Blocks: {sum(len(p['code_blocks']) for p in pages_data)}")
    logger.info(f"🔢 Formulas: {sum(len(p['formulas']) for p in pages_data)}")
    logger.info(f"\n📁 Output: {OUT_PROCESSED}")
    logger.info(f"   - Pages: {OUT_PAGES}")
    logger.info(f"   - Figures: {OUT_FIGS}")
    logger.info(f"   - Metadata: {OUT_META}")
    logger.info("="*70)


if __name__ == "__main__":
    main()