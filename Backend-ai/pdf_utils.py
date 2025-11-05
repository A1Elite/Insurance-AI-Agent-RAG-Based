# pdf_utils.py
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Extract text from PDF and split into clauses
def extract_clauses_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    clauses = [clause.strip() for clause in full_text.split('\n') if len(clause.strip()) > 30]
    return clauses

# Embed clauses using SentenceTransformer
def embed_clauses(clauses):
    return model.encode(clauses)

# Build FAISS index from embeddings
def build_faiss_index(embeddings):
    embeddings_np = np.array(embeddings)
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)
    return index
# Query FAISS index for similar clauses
def query_faiss(query_text, clauses, index):
    query_embedding = model.encode([query_text])
    distances, indices = index.search(np.array(query_embedding), 5)
    return [clauses[i] for i in indices[0]]
