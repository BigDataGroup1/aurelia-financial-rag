"""
Project AURELIA — Lab 1 Hybrid Version
Docling (→ Markdown, on GPU) + PyMuPDF (→ figures + captions)
Chunk → Embed → Store (Pinecone)
"""

import os, re, json, tempfile, shutil
from pathlib import Path
from dotenv import load_dotenv
import fitz  # PyMuPDF

from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PipelineOptions  # <-- for GPU device
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.settings import settings
from docling.document_converter import PdfFormatOption

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# ───────────────────────────────────────────────
# 1️⃣  Load Environment
# ───────────────────────────────────────────────
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

OPENAI_KEY   = os.getenv("OPENAI_API_KEY")
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME   = os.getenv("PINECONE_INDEX", "aurelia-fintbx")
NAMESPACE    = os.getenv("PINECONE_NAMESPACE", "fintbx")

PDF_PATH   = os.getenv("AURELIA_PDF_PATH", "data/raw_pdfs/fintbx.pdf")
WORK_DIR   = Path(os.getenv("AURELIA_WORK_DIR", "data/processed"))
FIG_DIR    = WORK_DIR / "figures"
JSONL_OUT  = WORK_DIR / "fintbx_hybrid_pages.jsonl"
CHUNKS_OUT = WORK_DIR / "fintbx_hybrid_chunks.jsonl"

WORK_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ───────────────────────────────────────────────
# 2️⃣  Helper Functions — Figures + Captions
# ───────────────────────────────────────────────
CAPTION_HINT = re.compile(r"^(Figure|Table)\s+\d+[:\.]", re.IGNORECASE)

def extract_figures(pdf_path: str):
    """Extract all images + captions from PDF."""
    doc = fitz.open(pdf_path)
    page_map = {}
    for i, page in enumerate(doc):
        imgs_meta = []
        for j, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n - pix.alpha > 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            out = FIG_DIR / f"page{i+1:03d}_img{j+1:02d}.png"
            pix.save(out)
            imgs_meta.append({"path": str(out), "xref": xref})
        # detect captions
        captions = [ln for ln in page.get_text("text").splitlines() if CAPTION_HINT.match(ln.strip())]
        page_map[i + 1] = {"images": imgs_meta, "captions": captions}
    return page_map

# ───────────────────────────────────────────────
# 3️⃣  Docling → Markdown (GPU enabled)
# ───────────────────────────────────────────────
def convert_pdf_to_markdown(pdf_path: str) -> str:
    # Select CUDA (your RTX 3060). You can also use AcceleratorDevice.AUTO.
    accelerator_options = AcceleratorOptions(
        num_threads=8,                 # tune if you like
        device=AcceleratorDevice.CUDA  # ← GPU
    )

    # PDF pipeline options (enable OCR/table structure if you want to stress GPU paths)
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = accelerator_options
    # Optional toggles:
    pipeline_options.do_ocr = False                 # skip OCR (often the slowest)
    pipeline_options.do_table_structure = False     # skip table structure recovery
    pipeline_options.page_range = (1, 5)            # process first 5 pages only
    settings.debug.profile_pipeline_timings = True  # print per-stage timings
    # Wire the PDF options into the converter
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # (Optional) timing/profiling output
    settings.debug.profile_pipeline_timings = True

    # Convert → Markdown
    tmp_dir = tempfile.mkdtemp()
    md_path = os.path.join(tmp_dir, "fintbx.md")

    print("⚙️ Docling: running PDF pipeline on CUDA…")

    result = converter.convert(pdf_path)
    md_text = result.document.export_to_markdown()

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    shutil.copy(md_path, WORK_DIR / "fintbx_docling.md")
    return md_text





# ───────────────────────────────────────────────
# 4️⃣  Merge Markdown + Figures → JSONL
# ───────────────────────────────────────────────
def merge_to_jsonl(markdown_text: str, fig_map: dict, out_path: Path):
    sections = markdown_text.split("\n## ")
    records = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        # guess page via figure caption keywords
        page_guess = None
        for pg, meta in fig_map.items():
            for cap in meta["captions"]:
                if cap.split(":")[0].lower() in sec.lower():
                    page_guess = pg
                    break
            if page_guess:
                break
        rec = {
            "page": page_guess,
            "section": sec.splitlines()[0][:120],
            "markdown": "## " + sec if not sec.startswith("##") else sec,
            "figures": fig_map.get(page_guess, {}).get("images", []),
            "captions": fig_map.get(page_guess, {}).get("captions", []),
            "source": os.path.basename(PDF_PATH),
        }
        records.append(rec)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✅ Hybrid records created → {out_path} ({len(records)} sections)")
    return records

# ───────────────────────────────────────────────
# 5️⃣  Chunking + Embeddings + Upload
# ───────────────────────────────────────────────
def chunk_and_embed(records):
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")])
    char_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=120)

    client = OpenAI(api_key=OPENAI_KEY)
    pc = Pinecone(api_key=PINECONE_KEY)

    print("🔍 Ensuring Pinecone index exists...")
    sample = client.embeddings.create(model="text-embedding-3-large", input=["sample"]).data[0].embedding
    dim = len(sample)
    existing = [x["name"] for x in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV),
        )
        print(f"✅ Created index {INDEX_NAME}")
    index = pc.Index(INDEX_NAME)

    total_chunks = 0
    for rec in records:
        parts = md_splitter.split_text(rec["markdown"])
        for p in parts:
            texts = char_splitter.split_text(p.page_content)
            for i, t in enumerate(texts):
                if len(t) < 200:
                    continue
                emb = client.embeddings.create(model="text-embedding-3-large", input=[t]).data[0].embedding
                vec_id = f"{rec['page'] or 0:03d}_{i:03d}"
                meta = {"page": rec["page"], "section": rec["section"], "source": rec["source"]}
                index.upsert(
                    vectors=[{"id": vec_id, "values": emb, "metadata": meta}],
                    namespace=NAMESPACE,
                )
                total_chunks += 1
    print(f"✅ Embedded and uploaded {total_chunks} chunks to '{INDEX_NAME}' (ns={NAMESPACE})")

# ───────────────────────────────────────────────
# 6️⃣  Main Pipeline
# ───────────────────────────────────────────────
if __name__ == "__main__":
    print("🚦 Starting AURELIA Hybrid Pipeline...")
    figs = extract_figures(PDF_PATH)
    md_text = convert_pdf_to_markdown(PDF_PATH)  # now uses GPU for Docling
    recs = merge_to_jsonl(md_text, figs, JSONL_OUT)
    chunk_and_embed(recs)
    print("🎉 Hybrid Lab 1 complete — Figures + Markdown embedded successfully.")
