"""
Simple viewer for all extracted content
Creates an HTML page you can open in your browser
"""

import os
from pathlib import Path
import base64

def create_html_viewer():
    """Create HTML viewer for all extracted content"""
    
    base_dir = Path("data/extracted")
    
    # Collect all files
    images = sorted(base_dir.glob("images/*.png"))
    formulas = sorted(base_dir.glob("formulas/*.txt"))
    code = sorted(base_dir.glob("code/*.txt"))
    
    print(f"Found:")
    print(f"  - {len(images)} images")
    print(f"  - {len(formulas)} formula files")
    print(f"  - {len(code)} code files")
    
    # Create HTML (using f-string to avoid format conflicts)
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>AURELIA - Extracted Content Viewer</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 40px;
            background: #3498db;
            color: white;
            padding: 10px;
            border-radius: 5px;
        }}
        .stats {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        .stats span {{
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }}
        .item {{
            background: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .item h3 {{
            color: #2c3e50;
            margin-top: 0;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin: 10px 0;
        }}
        pre {{
            background: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
        }}
        .formula {{
            background: #fff9e6;
            padding: 10px;
            margin: 5px 0;
            border-left: 4px solid #f39c12;
            font-family: monospace;
        }}
        .nav {{
            position: sticky;
            top: 0;
            background: white;
            padding: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            z-index: 100;
            margin-bottom: 20px;
        }}
        .nav a {{
            margin: 0 15px;
            color: #3498db;
            text-decoration: none;
            font-weight: bold;
        }}
        .nav a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <h1>🎯 AURELIA - Extracted Multi-Modal Content</h1>
    
    <div class="nav">
        <a href="#images">📷 Images</a>
        <a href="#formulas">🔢 Formulas</a>
        <a href="#code">💻 Code</a>
    </div>
    
    <div class="stats">
        <h3>📊 Extraction Summary</h3>
        <p>✅ Images extracted: <span>{len(images)}</span></p>
        <p>✅ Formula files: <span>{len(formulas)}</span></p>
        <p>✅ Code files: <span>{len(code)}</span></p>
    </div>
"""
    
    # Add images
    html += '\n<h2 id="images">📷 Extracted Images</h2>\n'
    for img_path in images:
        # Encode image as base64
        with open(img_path, 'rb') as f:
            img_data = base64.b64encode(f.read()).decode()
        
        html += f"""
        <div class="item">
            <h3>{img_path.name}</h3>
            <img src="data:image/png;base64,{img_data}" alt="{img_path.name}">
        </div>
        """
    
    # Add formulas
    html += '\n<h2 id="formulas">🔢 Extracted Formulas</h2>\n'
    for formula_path in formulas[:20]:  # Show first 20
        with open(formula_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        html += f"""
        <div class="item">
            <h3>{formula_path.name}</h3>
            <div class="formula">{content.replace('<', '&lt;').replace('>', '&gt;')}</div>
        </div>
        """
    
    if len(formulas) > 20:
        html += f'<p><em>... and {len(formulas) - 20} more formula files</em></p>'
    
    # Add code
    html += '\n<h2 id="code">💻 Extracted Code</h2>\n'
    for code_path in code[:20]:  # Show first 20
        with open(code_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        html += f"""
        <div class="item">
            <h3>{code_path.name}</h3>
            <pre>{content.replace('<', '&lt;').replace('>', '&gt;')}</pre>
        </div>
        """
    
    if len(code) > 20:
        html += f'<p><em>... and {len(code) - 20} more code files</em></p>'
    
    html += """
</body>
</html>
"""
    
    # Save HTML
    output_path = Path("data/extracted/viewer.html")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n✅ HTML viewer created: {output_path}")
    print(f"\n🌐 To view, run:")
    print(f"   open {output_path}")
    print(f"\nOr double-click the file in Finder!")
    
    return output_path


if __name__ == "__main__":
    create_html_viewer()