import os
import pickle
from pathlib import Path
import faiss
import numpy as np
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

MODEL = SentenceTransformer(EMBED_MODEL)
UPLOADS_DIR = Path("uploads")
OUT_DIR = Path("saved_index")
OUT_DIR.mkdir(exist_ok=True)

def load_pdf_text(path: Path) -> str:
    doc = fitz.open(path)
    pages = []
    for p in doc:
        text = p.get_text().strip()
        if text:
            pages.append(text)
    return "\n".join(pages)

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150):
    start = 0
    L = len(text)
    while start < L:
        end = min(L, start + chunk_size)
        yield text[start:end]
        start += chunk_size - overlap

def collect_documents():
    """Collect all PDF and TXT documents in uploads/ and return text chunks and metadata."""
    texts, metas = [], []

    if not UPLOADS_DIR.exists():
        print(f"Uploads directory {UPLOADS_DIR} does not exist. Create it and add files.")
        return texts, metas

    for f in UPLOADS_DIR.iterdir():
        if f.suffix.lower() not in (".pdf", ".txt"):
            continue
        try:
            if f.suffix.lower() == ".pdf":
                full = load_pdf_text(f)
            else:
                full = f.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Failed to read {f.name}: {e}")
            continue

        for i, chunk in enumerate(chunk_text(full)):
            texts.append(chunk)
            metas.append({"source": f.name, "chunk_id": i})

    return texts, metas

def build_index():
    texts, metas = collect_documents()
    if not texts:
        print("No documents found in uploads/. Add PDFs or .txt files and run again.")
        return

    print(f"Embedding {len(texts)} text chunks with {EMBED_MODEL} ...")
    embeddings = MODEL.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, OUT_DIR / "faiss.index")
    with open(OUT_DIR / "metadata.pkl", "wb") as fh:
        pickle.dump({"texts": texts, "metas": metas}, fh)

    print("Saved index to", OUT_DIR)

if __name__ == "__main__":
    build_index()
