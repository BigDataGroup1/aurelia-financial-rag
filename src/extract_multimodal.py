"""
AURELIA - Multi-Modal Element Extractor
========================================
Extracts and SAVES all images, tables, formulas, and code blocks
from your PDF as separate files you can view!
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
import re
import csv

import fitz  # PyMuPDF
from PIL import Image
import io


class MultiModalExtractor:
    """Extract and save all multi-modal elements from PDF"""
    
    def __init__(self, pdf_path: str, output_base: str = "data/extracted"):
        self.pdf_path = Path(pdf_path)
        self.doc = fitz.open(pdf_path)
        self.output_base = Path(output_base)
        
        # Create output directories
        self.images_dir = self.output_base / "images"
        self.tables_dir = self.output_base / "tables"
        self.formulas_dir = self.output_base / "formulas"
        self.code_dir = self.output_base / "code"
        
        for dir_path in [self.images_dir, self.tables_dir, self.formulas_dir, self.code_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Output directories created:")
        print(f"   - Images: {self.images_dir}")
        print(f"   - Tables: {self.tables_dir}")
        print(f"   - Formulas: {self.formulas_dir}")
        print(f"   - Code: {self.code_dir}")
    
    def extract_all_images(self) -> List[Dict]:
        """Extract and save all images from PDF"""
        print("\n" + "="*70)
        print("🖼️  EXTRACTING IMAGES")
        print("="*70 + "\n")
        
        extracted_images = []
        image_count = 0
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            image_list = page.get_images()
            
            print(f"📄 Page {page_num + 1}: Found {len(image_list)} images")
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = self.doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Generate filename
                    filename = f"page_{page_num+1:03d}_img_{img_index+1:02d}.{image_ext}"
                    filepath = self.images_dir / filename
                    
                    # Save image
                    with open(filepath, "wb") as f:
                        f.write(image_bytes)
                    
                    # Get image info
                    try:
                        pil_image = Image.open(io.BytesIO(image_bytes))
                        width, height = pil_image.size
                    except:
                        width, height = 0, 0
                    
                    # Try to find caption
                    caption = self._find_figure_caption(page, img, page_num)
                    
                    extracted_images.append({
                        'page': page_num + 1,
                        'filename': filename,
                        'filepath': str(filepath),
                        'caption': caption,
                        'width': width,
                        'height': height,
                        'format': image_ext
                    })
                    
                    image_count += 1
                    print(f"   ✓ Saved: {filename} ({width}x{height})")
                    if caption:
                        print(f"      Caption: {caption[:60]}...")
                
                except Exception as e:
                    print(f"   ⚠️  Error extracting image {img_index}: {e}")
        
        print(f"\n✅ Total images extracted: {image_count}")
        
        # Save metadata
        with open(self.images_dir / "images_metadata.json", 'w') as f:
            json.dump(extracted_images, f, indent=2)
        
        return extracted_images
    
    def _find_figure_caption(self, page, img, page_num) -> str:
        """Try to find caption for figure"""
        try:
            # Get image position
            img_rects = page.get_image_rects(img[0])
            if not img_rects:
                return None
            
            bbox = img_rects[0]
            
            # Search below the image for caption
            search_rect = fitz.Rect(
                bbox.x0, 
                bbox.y1, 
                bbox.x1, 
                min(bbox.y1 + 100, page.rect.height)
            )
            
            text = page.get_text("text", clip=search_rect)
            
            # Look for figure caption pattern
            patterns = [
                r'Figure\s+\d+[.:]\s*([^\n]+)',
                r'Fig\.\s*\d+[.:]\s*([^\n]+)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(0).strip()
            
            return None
        except:
            return None
    
    def extract_all_tables(self) -> List[Dict]:
        """Extract and save all tables from PDF"""
        print("\n" + "="*70)
        print("📊 EXTRACTING TABLES")
        print("="*70 + "\n")
        
        extracted_tables = []
        table_count = 0
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            
            try:
                tables = page.find_tables()
                
                if tables:
                    print(f"📄 Page {page_num + 1}: Found {len(tables)} tables")
                
                for table_index, table in enumerate(tables):
                    try:
                        # Extract table data
                        table_data = table.extract()
                        
                        if not table_data or len(table_data) == 0:
                            continue
                        
                        # Generate filename
                        filename_csv = f"page_{page_num+1:03d}_table_{table_index+1:02d}.csv"
                        filename_txt = f"page_{page_num+1:03d}_table_{table_index+1:02d}.txt"
                        
                        filepath_csv = self.tables_dir / filename_csv
                        filepath_txt = self.tables_dir / filename_txt
                        
                        # Save as CSV
                        with open(filepath_csv, 'w', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerows(table_data)
                        
                        # Save as formatted text
                        with open(filepath_txt, 'w', encoding='utf-8') as f:
                            for row in table_data:
                                f.write(' | '.join(str(cell) if cell else '' for cell in row))
                                f.write('\n')
                        
                        # Try to find caption
                        caption = self._find_table_caption(page, table.bbox)
                        
                        extracted_tables.append({
                            'page': page_num + 1,
                            'filename_csv': filename_csv,
                            'filename_txt': filename_txt,
                            'filepath_csv': str(filepath_csv),
                            'filepath_txt': str(filepath_txt),
                            'caption': caption,
                            'rows': len(table_data),
                            'cols': len(table_data[0]) if table_data else 0
                        })
                        
                        table_count += 1
                        print(f"   ✓ Saved: {filename_csv} ({len(table_data)} rows)")
                        if caption:
                            print(f"      Caption: {caption[:60]}...")
                    
                    except Exception as e:
                        print(f"   ⚠️  Error extracting table {table_index}: {e}")
            
            except Exception as e:
                print(f"   ⚠️  Error processing page {page_num + 1}: {e}")
        
        print(f"\n✅ Total tables extracted: {table_count}")
        
        # Save metadata
        with open(self.tables_dir / "tables_metadata.json", 'w') as f:
            json.dump(extracted_tables, f, indent=2)
        
        return extracted_tables
    
    def _find_table_caption(self, page, bbox) -> str:
        """Try to find caption for table"""
        try:
            # Search above the table for caption
            search_rect = fitz.Rect(
                bbox[0],
                max(bbox[1] - 100, 0),
                bbox[2],
                bbox[1]
            )
            
            text = page.get_text("text", clip=search_rect)
            
            # Look for table caption pattern
            patterns = [
                r'Table\s+\d+[.:]\s*([^\n]+)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(0).strip()
            
            return None
        except:
            return None
    
    def extract_all_formulas(self) -> List[Dict]:
        """Extract and save all formulas from PDF"""
        print("\n" + "="*70)
        print("🔢 EXTRACTING FORMULAS")
        print("="*70 + "\n")
        
        extracted_formulas = []
        formula_count = 0
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            text = page.get_text()
            
            # Patterns for formulas
            formula_patterns = [
                (r'\$\$(.+?)\$\$', 'latex_display'),
                (r'\$(.+?)\$', 'latex_inline'),
                (r'\\begin\{equation\}(.+?)\\end\{equation\}', 'latex_equation'),
                (r'\\begin\{align\}(.+?)\\end\{align\}', 'latex_align'),
                (r'\b([A-Z][a-zA-Z]*)\s*=\s*([^.;]+)', 'simple_equation'),
            ]
            
            page_formulas = []
            
            for pattern, formula_type in formula_patterns:
                matches = re.finditer(pattern, text, re.DOTALL)
                
                for match in matches:
                    formula_text = match.group(0).strip()
                    
                    # Skip if too short or too long (likely false positive)
                    if len(formula_text) < 3 or len(formula_text) > 500:
                        continue
                    
                    # Get context
                    start = max(0, match.start() - 150)
                    end = min(len(text), match.end() + 150)
                    context = text[start:end].strip()
                    
                    page_formulas.append({
                        'formula': formula_text,
                        'type': formula_type,
                        'context': context
                    })
            
            if page_formulas:
                print(f"📄 Page {page_num + 1}: Found {len(page_formulas)} formulas")
                
                # Save all formulas from this page
                filename = f"page_{page_num+1:03d}_formulas.txt"
                filepath = self.formulas_dir / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"FORMULAS FROM PAGE {page_num + 1}\n")
                    f.write("=" * 70 + "\n\n")
                    
                    for i, formula_info in enumerate(page_formulas, 1):
                        f.write(f"Formula {i} [{formula_info['type']}]:\n")
                        f.write("-" * 70 + "\n")
                        f.write(formula_info['formula'])
                        f.write("\n\n")
                        f.write("Context:\n")
                        f.write(formula_info['context'])
                        f.write("\n\n" + "=" * 70 + "\n\n")
                
                extracted_formulas.append({
                    'page': page_num + 1,
                    'filename': filename,
                    'filepath': str(filepath),
                    'count': len(page_formulas),
                    'formulas': page_formulas
                })
                
                formula_count += len(page_formulas)
                print(f"   ✓ Saved: {filename}")
        
        print(f"\n✅ Total formulas extracted: {formula_count}")
        
        # Save metadata
        with open(self.formulas_dir / "formulas_metadata.json", 'w') as f:
            json.dump(extracted_formulas, f, indent=2)
        
        return extracted_formulas
    
    def extract_all_code_blocks(self) -> List[Dict]:
        """Extract and save all code blocks from PDF"""
        print("\n" + "="*70)
        print("💻 EXTRACTING CODE BLOCKS")
        print("="*70 + "\n")
        
        extracted_code = []
        code_count = 0
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            page_code_blocks = []
            
            for block in blocks:
                if "lines" not in block:
                    continue
                
                # Check if block contains code (monospace font)
                block_text = ""
                is_code = False
                
                for line in block["lines"]:
                    for span in line["spans"]:
                        font_name = span["font"]
                        if "Courier" in font_name or "Mono" in font_name or "Code" in font_name:
                            is_code = True
                        block_text += span["text"] + "\n"
                
                if is_code and len(block_text.strip()) > 20:
                    page_code_blocks.append(block_text.strip())
            
            if page_code_blocks:
                print(f"📄 Page {page_num + 1}: Found {len(page_code_blocks)} code blocks")
                
                # Save all code blocks from this page
                filename = f"page_{page_num+1:03d}_code.txt"
                filepath = self.code_dir / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"CODE BLOCKS FROM PAGE {page_num + 1}\n")
                    f.write("=" * 70 + "\n\n")
                    
                    for i, code_text in enumerate(page_code_blocks, 1):
                        f.write(f"Code Block {i}:\n")
                        f.write("-" * 70 + "\n")
                        f.write(code_text)
                        f.write("\n\n" + "=" * 70 + "\n\n")
                
                extracted_code.append({
                    'page': page_num + 1,
                    'filename': filename,
                    'filepath': str(filepath),
                    'count': len(page_code_blocks)
                })
                
                code_count += len(page_code_blocks)
                print(f"   ✓ Saved: {filename}")
        
        print(f"\n✅ Total code blocks extracted: {code_count}")
        
        # Save metadata
        with open(self.code_dir / "code_metadata.json", 'w') as f:
            json.dump(extracted_code, f, indent=2)
        
        return extracted_code
    
    def create_index_html(self, images, tables, formulas, code_blocks):
        """Create an HTML index to browse all extracted elements"""
        print("\n📄 Creating HTML viewer...")
        
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>AURELIA - Extracted Multi-Modal Elements</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #2c3e50; }
        h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }
        .section { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .item { margin: 15px 0; padding: 10px; background: #ecf0f1; border-radius: 4px; }
        .caption { font-style: italic; color: #7f8c8d; margin-top: 5px; }
        img { max-width: 100%; height: auto; border: 1px solid #bdc3c7; margin-top: 10px; }
        table { border-collapse: collapse; width: 100%; margin-top: 10px; }
        th, td { border: 1px solid #bdc3c7; padding: 8px; text-align: left; }
        th { background: #3498db; color: white; }
        pre { background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 4px; overflow-x: auto; }
        .stats { background: #3498db; color: white; padding: 10px; border-radius: 4px; display: inline-block; margin-right: 10px; }
    </style>
</head>
<body>
    <h1>🎯 AURELIA - Extracted Multi-Modal Elements</h1>
    
    <div style="margin: 20px 0;">
        <span class="stats">📊 Images: {num_images}</span>
        <span class="stats">📋 Tables: {num_tables}</span>
        <span class="stats">🔢 Formulas: {num_formulas}</span>
        <span class="stats">💻 Code: {num_code}</span>
    </div>
""".format(
            num_images=len(images),
            num_tables=len(tables),
            num_formulas=sum(f['count'] for f in formulas),
            num_code=sum(c['count'] for c in code_blocks)
        )
        
        # Images section
        html += '<div class="section"><h2>🖼️ Images</h2>\n'
        for img in images:
            html += f'<div class="item">'
            html += f'<strong>Page {img["page"]}</strong> - {img["filename"]}<br>'
            if img["caption"]:
                html += f'<div class="caption">{img["caption"]}</div>'
            html += f'<img src="{img["filepath"]}" alt="{img["filename"]}">'
            html += f'</div>\n'
        html += '</div>\n'
        
        # Tables section
        html += '<div class="section"><h2>📊 Tables</h2>\n'
        for tbl in tables:
            html += f'<div class="item">'
            html += f'<strong>Page {tbl["page"]}</strong> - {tbl["rows"]} rows × {tbl["cols"]} columns<br>'
            if tbl["caption"]:
                html += f'<div class="caption">{tbl["caption"]}</div>'
            html += f'<a href="{tbl["filepath_csv"]}">Download CSV</a> | '
            html += f'<a href="{tbl["filepath_txt"]}">Download TXT</a>'
            html += f'</div>\n'
        html += '</div>\n'
        
        # Formulas section
        html += '<div class="section"><h2>🔢 Formulas</h2>\n'
        for formula in formulas:
            html += f'<div class="item">'
            html += f'<strong>Page {formula["page"]}</strong> - {formula["count"]} formulas<br>'
            html += f'<a href="{formula["filepath"]}">View All Formulas</a>'
            html += f'</div>\n'
        html += '</div>\n'
        
        # Code section
        html += '<div class="section"><h2>💻 Code Blocks</h2>\n'
        for code in code_blocks:
            html += f'<div class="item">'
            html += f'<strong>Page {code["page"]}</strong> - {code["count"]} code blocks<br>'
            html += f'<a href="{code["filepath"]}">View All Code</a>'
            html += f'</div>\n'
        html += '</div>\n'
        
        html += '</body></html>'
        
        # Save HTML
        html_path = self.output_base / "index.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✅ HTML viewer created: {html_path}")
        print(f"\n💡 Open this file in your browser to view all extracted elements!")
        
        return html_path
    
    def extract_all(self):
        """Extract everything"""
        print("\n" + "="*70)
        print("🚀 EXTRACTING ALL MULTI-MODAL ELEMENTS")
        print("="*70)
        
        images = self.extract_all_images()
        tables = self.extract_all_tables()
        formulas = self.extract_all_formulas()
        code_blocks = self.extract_all_code_blocks()
        
        # Create HTML viewer
        html_path = self.create_index_html(images, tables, formulas, code_blocks)
        
        print("\n" + "="*70)
        print("✅ EXTRACTION COMPLETE!")
        print("="*70)
        print(f"\n📁 All elements saved to: {self.output_base}")
        print(f"\n🌐 To view everything, open: {html_path}")
        print(f"\n   On Mac: open {html_path}")
        print(f"   Or just double-click the file in Finder")


def main():
    """Run extraction"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract all multi-modal elements from PDF")
    parser.add_argument('--pdf', default='data/raw_pdfs/fintbx.pdf', help='PDF file path')
    parser.add_argument('--output', default='data/extracted', help='Output directory')
    
    args = parser.parse_args()
    
    extractor = MultiModalExtractor(args.pdf, args.output)
    extractor.extract_all()


if __name__ == "__main__":
    main()