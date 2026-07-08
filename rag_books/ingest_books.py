#!/usr/bin/env python3
"""
ingest_books.py - PDF ingestion for the local RAG chatbot

Drop your PDFs into ./books (sub-folders become "categories"), then run:

    python ingest_books.py --clear

Produces:
    ./vectorstore        ChromaDB vector index (semantic search)
    ./bm25_corpus.json   chunk corpus for BM25 keyword search

Features:
- Code-fence-aware chunking (1200 chars / 300 overlap) so code examples stay intact
- Per-chunk metadata: book_title, category, page, chunk_index/total_chunks
- --clear flag to wipe and rebuild the vectorstore
"""
import os
import sys
import json
import warnings
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

warnings.filterwarnings("ignore")

# ==================== CONFIG ====================
BASE_DIR          = Path(__file__).resolve().parent
BOOKS_ROOT        = BASE_DIR / "books"
PERSIST_DIR       = BASE_DIR / "vectorstore"
BM25_CORPUS_PATH  = BASE_DIR / "bm25_corpus.json"
EMBED_MODEL       = "nomic-embed-text"
OLLAMA_BASE_URL   = "http://127.0.0.1:11434"
COLLECTION_NAME   = "technical_books_rag"

CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 300
BATCH_SIZE    = 100

# ==================== CLEAR ====================
if "--clear" in sys.argv:
    import shutil
    shutil.rmtree(PERSIST_DIR, ignore_errors=True)
    print("Cleared existing vectorstore.\n")

print("Starting ingestion...\n")

if not BOOKS_ROOT.exists():
    print(f"No books folder found. Create {BOOKS_ROOT} and add your PDFs (sub-folders become categories).")
    sys.exit(1)

# ==================== SETUP ====================
embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

vectorstore = Chroma(
    persist_directory=str(PERSIST_DIR),
    collection_name=COLLECTION_NAME,
    embedding_function=None,
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[
        "\n```",    # don't split inside code fences
        "```\n",
        "\n\n\n",
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ],
)

all_pages = []
processed = skipped = total_pages = 0

print(f"Scanning: {BOOKS_ROOT}\n")

for root, dirs, files in os.walk(BOOKS_ROOT):
    category = Path(root).name
    for file in sorted(files):
        if not file.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(root, file)
        try:
            loader = PyMuPDFLoader(pdf_path)
            docs = loader.load()

            for doc in docs:
                book_title = Path(file).stem.replace("_", " ").replace("-", " ").strip()
                page_num = doc.metadata.get("page") or doc.metadata.get("page_number")
                doc.metadata.update({
                    "book_title": book_title,
                    "file_name": file,
                    "source": pdf_path,
                    "category": category,
                    "page": page_num,
                })

            all_pages.extend(docs)
            processed += 1
            total_pages += len(docs)
            print(f"  [{processed:3d}] {file[:55]:<55} -> {len(docs):4d} pages")

        except Exception as e:
            skipped += 1
            print(f"  SKIP {file}: {e}")

print(f"\nLoaded {len(all_pages)} pages from {processed} PDFs (skipped {skipped})\n")

if not all_pages:
    print("No PDFs found. Add PDFs to the books/ folder.")
    sys.exit(1)

# ==================== CHUNK ====================
print("Chunking documents...")
raw_chunks = splitter.split_documents(all_pages)

# Annotate with chunk position per source file
from collections import defaultdict
file_chunk_counts: dict = defaultdict(int)
for chunk in raw_chunks:
    src = chunk.metadata.get("source", "")
    file_chunk_counts[src] += 1

file_chunk_index: dict = defaultdict(int)
chunks = []
for chunk in raw_chunks:
    src = chunk.metadata.get("source", "")
    idx = file_chunk_index[src]
    file_chunk_index[src] += 1
    chunk.metadata["chunk_index"] = idx
    chunk.metadata["total_chunks"] = file_chunk_counts[src]
    chunks.append(chunk)

print(f"Created {len(chunks)} chunks (avg {len(chunks)//max(processed,1)} per PDF)\n")

# ==================== BM25 CORPUS ====================
print(f"Saving BM25 corpus to {BM25_CORPUS_PATH}...")
bm25_corpus = [
    {"text": c.page_content, "metadata": c.metadata}
    for c in chunks
]
with open(BM25_CORPUS_PATH, "w", encoding="utf-8") as f:
    json.dump(bm25_corpus, f, ensure_ascii=False)
print(f"Saved {len(bm25_corpus)} entries.\n")

# ==================== EMBED + STORE ====================
print(f"Embedding and storing in batches (batch_size={BATCH_SIZE})...\n")

for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i : i + BATCH_SIZE]
    texts = [d.page_content for d in batch]
    metadatas = [d.metadata for d in batch]

    batch_embeddings = embeddings.embed_documents(texts)

    vectorstore.add_texts(
        texts=texts,
        metadatas=metadatas,
        embeddings=batch_embeddings,
    )
    print(f"  Batch {i // BATCH_SIZE + 1:3d} ({len(batch)} chunks)")

print("\nIngestion complete!")
print(f"Total chunks in vectorstore: {vectorstore._collection.count()}")
print("\nRun: python rag_chatbot.py   (or python rag_ui.py for the web UI)")
