import os
import pickle
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
INDEX_DIR = Path("saved_index")

_model = SentenceTransformer(EMBED_MODEL)

def load_store():
    idx_path = INDEX_DIR / "faiss.index"
    meta_path = INDEX_DIR / "metadata.pkl"
    if not idx_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Index not found. Run ingest.py first.")
    index = faiss.read_index(str(idx_path))
    meta = pickle.load(open(meta_path, "rb"))
    return index, meta

def query(text, k=4):
    index, meta = load_store()
    emb = _model.encode([text], convert_to_numpy=True).astype("float32")
    D, I = index.search(emb, k)
    results = []
    for idx in I[0]:
        results.append({
            "text": meta["texts"][idx],
            "meta": meta["metas"][idx]
        })
    return results
