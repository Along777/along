# Local RAG Chatbot for Your PDF Library

Turn a folder of PDFs (mine was 148 technical books from Humble Bundle — ~48,800 pages)
into a private, citation-grounded chatbot that runs **entirely on your own machine**.
No cloud APIs, no cost, no data leaving your laptop.

**[Read the full technical write-up →](index.html)** (architecture, design decisions, and the debugging journal)

## Features

- **Hybrid retrieval** — ChromaDB vector search + BM25 keyword search, merged with Reciprocal Rank Fusion
- **Query rewriting** — vague follow-ups ("how do I do that?") become standalone search queries
- **Citation guardrails** — answers cite only books that were actually retrieved; a deterministic
  post-check redacts anything else
- **Persistent memory** — conversations survive restarts (SQLite-backed LangGraph checkpoints)
- **Full trace logs** — every turn logs the rewritten query, retrieved chunks, raw answer, and
  verification result, so quality is debuggable from evidence
- **Two interfaces** — terminal chat and a Gradio web UI

## Stack

| Tool | Role |
|---|---|
| [Ollama](https://ollama.com) | runs the LLM (`qwen2.5:3b`) and embeddings (`nomic-embed-text`) locally |
| ChromaDB | persistent vector store |
| rank_bm25 | in-memory keyword search |
| LangGraph | pipeline orchestration + conversation memory |
| PyMuPDF | PDF text extraction |
| Gradio | web chat UI |

## Setup

**1. Install [Ollama](https://ollama.com) and pull the models:**

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

**2. Install Python dependencies** (Python 3.10+):

```bash
pip install -r requirements.txt
```

**3. Add your PDFs** to a `books/` folder next to the scripts.
Sub-folders become categories in citations:

```
books/
├── Machine Learning/
│   ├── some-ml-book.pdf
│   └── another-book.pdf
└── Python/
    └── python-book.pdf
```

**4. Ingest** (builds the vector store + BM25 corpus — takes a while for large libraries):

```bash
python ingest_books.py --clear
```

**5. Chat:**

```bash
python rag_chatbot.py     # terminal
python rag_ui.py          # web UI at http://localhost:7860
```

## How it works

```
question → rewrite_query → retrieve (vector + BM25 → RRF) → generate → verify_citations → answer
```

Each stage exists to contain a specific failure mode of a small local model — the
[write-up](index.html) covers the reasoning and the bugs that shaped the design.

## Notes

- Your `books/`, `vectorstore/`, chat logs, and memory DB are gitignored — this repo ships
  only the code, so your library stays yours.
- Model is configurable in `rag_chatbot.py` (`LLM_MODEL`). Larger models like `qwen2.5:7b`
  follow the citation rules even more reliably if your hardware allows.
