from dotenv import load_dotenv
import os

# Load .env from project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

print("✅ OPENAI key exists:", bool(os.getenv("OPENAI_API_KEY")))
print("✅ PINECONE key exists:", bool(os.getenv("PINECONE_API_KEY")))
print("📄 PDF path:", os.getenv("AURELIA_PDF_PATH"))
