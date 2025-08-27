import os
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv
from vector_store import query as query_index
from pathlib import Path

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not set in .env")

genai.configure(api_key=GOOGLE_API_KEY)
GEN_MODEL = "gemini-1.5-flash"
model = genai.GenerativeModel(GEN_MODEL)

# Set template and static folders relative to this file
app = Flask(__name__, template_folder="templates", static_folder="static")

def build_context_from_docs(user_question, k=4):
    try:
        docs = query_index(user_question, k=k)
    except FileNotFoundError:
        return "", []
    contexts, sources = [], []
    for d in docs:
        contexts.append(d["text"].strip())
        sources.append(f'{d["meta"]["source"]}#chunk{d["meta"]["chunk_id"]}')
    return "\n\n---\n\n".join(contexts), sources

def make_prompt(question, contexts):
    return f"""
You are an experienced college-level teaching assistant.
Answer clearly and step-by-step.

Goals:
1) Explain simply in structured steps.
2) Provide a short example or analogy.
3) End with one quick quiz question.
4) If no CONTEXTS, answer from your knowledge but note that.

CONTEXTS:
{contexts}

QUESTION:
{question}
"""

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(force=True)
    message = payload.get("message", "").strip()
    if not message:
        return jsonify({"ok": False, "answer": "Please type a question."})
    contexts, sources = build_context_from_docs(message, k=4)
    prompt = make_prompt(message, contexts)
    try:
        response = model.generate_content(prompt)
        answer = response.text
    except Exception as e:
        answer = f"Model error: {e}"
    return jsonify({"ok": True, "answer": answer, "sources": sources})

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file part"})
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"ok": False, "error": "No selected file"})
    dest = UPLOADS_DIR / f.filename
    f.save(dest)
    os.system("python ingest.py")
    return jsonify({"ok": True, "filename": f.filename})

# Production-ready: use PORT from environment, host 0.0.0.0
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
