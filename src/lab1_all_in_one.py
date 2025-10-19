"""
Project AURELIA — Lab 1 (All-in-One)
Parse → Chunk → Embed → Store (Pinecone)
"""

import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv
import fitz  # PyMuPDF
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# ========== 1️⃣ Load Environment ==========
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("PINECONE_INDEX", "aurelia-fintbx")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "fintbx")

PDF_PATH = os.getenv("AURELIA_PDF_PATH", "data/raw_pdfs/fintbx.pdf")
WORK_DIR = Path(os.getenv("AURELIA_WORK_DIR", "data/processed"))
FIG_DIR = WORK_DIR / "figures"
PAGES_JSONL = WORK_DIR / "fintbx_pages.jsonl"
CHUNKS_JSONL = WORK_DIR / "fintbx_chunks.jsonl"

WORK_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ========== 2️⃣ PDF Parsing ==========
def parse_pdf(pdf_path: str, out_path: Path):
    doc = fitz.open(pdf_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        # crude section title detection like '2.3 Sharpe Ratio'
        header = None
        for ln in text.splitlines()[:8]:
            if re.match(r"^\\d+(?:\\.\\d+)*\\s+.+", ln.strip()):
                header = ln.strip()
                break
        rec = {
            "page": i + 1,
            "section": header or "",
            "markdown": text.strip(),
            "source": os.path.basename(pdf_path),
        }
        records.append(rec)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\\n")
    print(f"✅ Parsed {len(records)} pages → {out_path}")
    return records

# ========== 3️⃣ Chunking ==========
def chunk_pages(records, out_path: Path):
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
        ("#", "h1"), ("##", "h2"), ("###", "h3")
    ])
    char_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=120)

    chunks = []
    for rec in records:
        md_text = rec["markdown"]
        md_docs = md_splitter.split_text(md_text)
        for d in md_docs:
            parts = char_splitter.split_text(d.page_content)
            for idx, t in enumerate(parts):
                if len(t) < 200:
                    continue
                chunks.append({
                    "id": f"page{rec['page']:03d}_chunk{idx:03d}",
                    "text": t,
                    "metadata": {
                        "page": rec["page"],
                        "section": rec["section"],
                        "source": rec["source"]
                    }
                })
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\\n")
    print(f"✅ Created {len(chunks)} chunks → {out_path}")
    return chunks

# ========== 4️⃣ Embedding + Pinecone Upload ==========
def embed_and_store(chunks):
    client = OpenAI(api_key=OPENAI_KEY)
    pc = Pinecone(api_key=PINECONE_KEY)

    # create index if not exist
    print("🔍 Checking/creating Pinecone index…")
    sample_embedding = client.embeddings.create(model="text-embedding-3-large", input=["test"]).data[0].embedding
    dim = len(sample_embedding)
    existing = [i["name"] for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(name=INDEX_NAME, dimension=dim, metric="cosine", spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV))
        print(f"✅ Created index: {INDEX_NAME}")

    index = pc.Index(INDEX_NAME)

    print(f"🚀 Uploading {len(chunks)} embeddings to Pinecone...")
    BATCH = 50
    buf = []
    for i, c in enumerate(chunks):
        emb = client.embeddings.create(model="text-embedding-3-large", input=[c["text"]]).data[0].embedding
        buf.append({"id": c["id"], "values": emb, "metadata": c["metadata"]})
        if len(buf) >= BATCH:
            index.upsert(vectors=buf, namespace=NAMESPACE)
            buf = []
    if buf:
        index.upsert(vectors=buf, namespace=NAMESPACE)

    print(f"✅ Uploaded all chunks to index '{INDEX_NAME}' (namespace='{NAMESPACE}')")

# ========== 5️⃣ Main Runner ==========
if __name__ == "__main__":
    print("🚦 Starting Lab 1 Pipeline...")
    records = parse_pdf(PDF_PATH, PAGES_JSONL)
    chunks = chunk_pages(records, CHUNKS_JSONL)
    embed_and_store(chunks)
    print("🎉 Lab 1 complete: PDF parsed, chunks embedded, Pinecone updated.")
