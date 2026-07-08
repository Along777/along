#!/usr/bin/env python3
"""
rag_chatbot.py - Local RAG chatbot over your own PDF library
============================================================
Features:
- Hybrid retrieval: BM25 keyword search + ChromaDB vector search merged via RRF
- Query rewriting so vague/follow-up questions retrieve well
- Citation verification: redacts book titles that were not actually retrieved
- Persistent conversation memory (SQLite) + full per-turn trace logs

Pipeline: rewrite_query -> retrieve -> generate -> verify_citations -> END

Setup:
    python ingest_books.py --clear   (first time or after adding books)
    python rag_chatbot.py            (CLI)  /  python rag_ui.py  (web UI)
"""

import json
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import TypedDict, List, Annotated, Optional, Set

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages

from rank_bm25 import BM25Okapi

warnings.filterwarnings("ignore")

# ==================== CONFIG ====================
BASE_DIR         = Path(__file__).resolve().parent
PERSIST_DIR      = BASE_DIR / "vectorstore"
BM25_CORPUS_PATH = BASE_DIR / "bm25_corpus.json"
CHECKPOINT_DB    = BASE_DIR / "chat_memory.sqlite"
CHAT_LOG_DIR     = BASE_DIR / "chat_logs"
EMBED_MODEL      = "nomic-embed-text"
LLM_MODEL        = "qwen2.5:3b"
OLLAMA_BASE_URL  = "http://127.0.0.1:11434"
RETRIEVER_K      = 12   # per source before RRF merge
FINAL_K          = 10   # after RRF

CHAT_LOG_DIR.mkdir(exist_ok=True)

if not PERSIST_DIR.exists() or not BM25_CORPUS_PATH.exists():
    print("Vectorstore or BM25 corpus not found.")
    print("Add PDFs to ./books and run:  python ingest_books.py --clear")
    sys.exit(1)


def log_turn(thread_id: str, result: dict, question: str) -> None:
    """Write a full per-node trace of one turn through the graph to the thread's log file."""
    log_path = CHAT_LOG_DIR / f"{thread_id}.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rewritten_query = result.get("rewritten_query") or "(unchanged - no prior history)"
    retrieved_docs  = result.get("retrieved_docs") or []
    raw_answer      = result.get("raw_answer", "")
    citation_flags  = result.get("citation_flags") or []
    final_answer    = result.get("answer", "")

    lines = [f"[{timestamp}] ==== Turn ===="]
    lines.append(f"Question: {question}")
    lines.append(f"Rewritten query: {rewritten_query}")

    lines.append(f"Retrieved ({len(retrieved_docs)} chunks):")
    if retrieved_docs:
        for doc in retrieved_docs:
            title    = doc.metadata.get("book_title", "Unknown Book")
            page     = doc.metadata.get("page")
            category = doc.metadata.get("category", "")
            chunk_i  = doc.metadata.get("chunk_index")
            total_c  = doc.metadata.get("total_chunks")
            loc = f"chunk {chunk_i + 1}/{total_c}" if chunk_i is not None and total_c is not None else ""
            lines.append(f"  - {title} (p. {page}) [{category}] {loc}".rstrip())
    else:
        lines.append("  (none)")

    lines.append(f"Raw answer: {raw_answer}")

    if citation_flags:
        lines.append(f"Citation check: redacted unverified title(s): {citation_flags}")
    else:
        lines.append("Citation check: no titles redacted")

    lines.append(f"Final answer: {final_answer}")
    lines.append("")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

print("Loading embeddings and vector store...")

embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

vectorstore = Chroma(
    persist_directory=str(PERSIST_DIR),
    embedding_function=embeddings,
)

print("Loading BM25 corpus...")
with open(BM25_CORPUS_PATH, encoding="utf-8") as f:
    bm25_data = json.load(f)

bm25_texts     = [d["text"] for d in bm25_data]
bm25_metadatas = [d["metadata"] for d in bm25_data]
bm25_index     = BM25Okapi([t.lower().split() for t in bm25_texts])

llm = ChatOllama(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.1,
    num_predict=2048,
)

print("Vector store, BM25, and LLM loaded.\n")


# ==================== HELPERS ====================
def reciprocal_rank_fusion(
    list1: List[Document], list2: List[Document], k: int = 60
) -> List[Document]:
    scores: dict = {}
    doc_map: dict = {}

    for rank, doc in enumerate(list1):
        key = doc.page_content[:120]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc

    for rank, doc in enumerate(list2):
        key = doc.page_content[:120]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc

    ranked = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[k] for k in ranked]


def format_docs(docs: List[Document]) -> tuple[str, set]:
    """Return (formatted context string, set of allowed book titles)."""
    if not docs:
        return "No relevant documents were retrieved from your book collection.", set()

    book_titles: Set[str] = set()
    formatted_chunks = []

    for doc in docs:
        title    = doc.metadata.get("book_title", "Unknown Book")
        page     = doc.metadata.get("page") or doc.metadata.get("page_number")
        category = doc.metadata.get("category", "")
        chunk_i  = doc.metadata.get("chunk_index")
        total_c  = doc.metadata.get("total_chunks")

        book_titles.add(title)

        if page is not None and category:
            source_line = f"Source: {title} (p. {page}) [{category}]"
        elif page is not None:
            source_line = f"Source: {title} (p. {page})"
        elif category:
            source_line = f"Source: {title} [{category}]"
        else:
            source_line = f"Source: {title}"

        if chunk_i is not None and total_c is not None:
            source_line += f" chunk {chunk_i + 1}/{total_c}"

        content = doc.page_content.strip()
        if len(content) > 900:
            content = content[:900] + " ... [truncated]"

        formatted_chunks.append(f"{source_line}\n{content}")

    sources_header = (
        "Available Sources in this retrieval (ONLY cite titles from this exact list):\n"
        + "\n".join(f"- {t}" for t in sorted(book_titles))
    )
    context_body = "\n\n---\n\n".join(formatted_chunks)
    context_str = f"{sources_header}\n\n---\n\nContext excerpts from your books:\n{context_body}"

    return context_str, book_titles


# ==================== STATE ====================
class GraphState(TypedDict):
    messages:        Annotated[List[BaseMessage], add_messages]
    context:         Optional[str]
    rewritten_query: Optional[str]
    answer:          Optional[str]
    allowed_titles:  Optional[set]              # set of book titles from retrieved docs
    retrieved_docs:  Optional[List[Document]]   # raw merged docs, for tracing/logging
    raw_answer:      Optional[str]              # LLM answer before citation verification
    citation_flags:  Optional[list]             # titles redacted by verify_citations_node, if any


# ==================== NODES ====================
def rewrite_query_node(state: GraphState) -> dict:
    messages = state["messages"]
    if not messages:
        return {"rewritten_query": ""}

    last_human = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human = msg.content
            break

    if not last_human:
        return {"rewritten_query": ""}

    if len(messages) <= 1:
        return {"rewritten_query": last_human}

    print("Rewriting query for better retrieval...")

    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at rewriting user questions into optimized search queries
for a personal library of technical books covering many topics (programming, AI tools, data science, business, and more).

Your goal: clear, specific, standalone search query that retrieves the most relevant chunks.
- Expand abbreviations (EDA -> exploratory data analysis)
- Keep the key terms the user actually used
- Make the query self-contained (no pronouns like "these", "it", "them")
- Do NOT add topics, languages, or technologies the user did not mention
  (e.g. do not append "Python" or "machine learning" to a question that isn't about them)
- Write in plain natural language with normal spaces — do NOT join words with hyphens or underscores

Output ONLY the rewritten search query, as a short plain-language phrase (under 20 words). No explanations."""),
        ("human", "Conversation:\n{history}\n\nLatest question: {question}\n\nRewritten query:"),
    ])

    history_str = ""
    for msg in messages[-6:]:
        if isinstance(msg, HumanMessage):
            history_str += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_str += f"Assistant: {msg.content[:200]}...\n"

    response = (rewrite_prompt | llm).invoke({
        "history": history_str.strip(),
        "question": last_human,
    })

    rewritten = response.content.strip().strip('"').strip("'")

    # Sanity check: reject malformed rewrites (excessive hyphenation, run-on length)
    # and fall back to the original question rather than searching with garbage.
    if rewritten.count("-") > 3 or len(rewritten) > 200 or not rewritten:
        print(f"  Rewrite looked malformed, falling back to original question: {rewritten[:80]}")
        rewritten = last_human

    print(f"  -> {rewritten[:100]}")
    return {"rewritten_query": rewritten}


def retrieve_node(state: GraphState) -> dict:
    messages = state["messages"]
    if not messages:
        return {"context": "No question found.", "allowed_titles": set()}

    query = state.get("rewritten_query") or ""
    if not query:
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                query = msg.content
                break

    print(f"Retrieving (hybrid, final_k={FINAL_K})...")

    # Vector search
    vector_docs = vectorstore.similarity_search(query, k=RETRIEVER_K)

    # BM25 search
    tokenized = query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized)
    top_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:RETRIEVER_K]
    bm25_docs = [
        Document(page_content=bm25_texts[i], metadata=bm25_metadatas[i])
        for i in top_idx
    ]

    # RRF merge
    merged = reciprocal_rank_fusion(vector_docs, bm25_docs)[:FINAL_K]

    context_str, allowed_titles = format_docs(merged)
    return {"context": context_str, "allowed_titles": allowed_titles, "retrieved_docs": merged}


def generate_node(state: GraphState) -> dict:
    context      = state.get("context", "")
    all_messages = state["messages"]

    if all_messages and isinstance(all_messages[-1], HumanMessage):
        current_question = all_messages[-1].content
        history          = all_messages[:-1]
    else:
        current_question = ""
        history          = all_messages

    system_content = f"""You are a helpful technical assistant. Answer questions using ONLY the book excerpts provided below.

RULES:

1. Base your answer entirely on the "Context excerpts from your books" below. Do not use outside knowledge.

2. Only mention book titles that appear in the "Available Sources in this retrieval" list.

3. Not every excerpt is relevant. Use ONLY the excerpts that actually relate to the question, and silently ignore the rest. Never force an irrelevant excerpt into the answer just because it was retrieved.

4. The context below DOES contain relevant excerpts. Read them carefully and give a thorough answer. Do not refuse unless the excerpts truly contain zero relevant content after careful reading.

5. If and only if EVERY excerpt is completely unrelated to the question, respond with EXACTLY this sentence and NOTHING ELSE:
   "I don't have information about that in the books I have access to."
   Never append this sentence to an answer — either answer, or output only this sentence.

Context excerpts from your books:
{context}
"""

    chat_messages: List[BaseMessage] = (
        [SystemMessage(content=system_content)] + history + [HumanMessage(content=current_question)]
    )

    print("Generating answer...")
    response = llm.invoke(chat_messages)
    answer   = response.content.strip()

    return {
        "messages":   [AIMessage(content=answer)],
        "answer":     answer,
        "raw_answer": answer,
    }


def verify_citations_node(state: GraphState) -> dict:
    """Deterministic post-processing on the generated answer.

    1. Strip a contradictory refusal sentence the model sometimes appends to a real answer.
    2. Redact book titles cited in the prose that were NOT in the retrieved sources.

    Lesson from the trace logs: an earlier version treated ANY quoted string as a potential
    book title and replaced the whole answer with a refusal. Every flag it ever raised was
    a false positive (quoted phrases like "You're absolutely right" or "an AI chatbot"),
    destroying good answers. Now a quote only counts as a citation when it appears in
    citation context (near words like book/titled/guide), and we redact just the title
    instead of nuking the answer.
    """
    answer   = state.get("answer", "")
    allowed  = state.get("allowed_titles") or set()
    original = answer

    refusal = "I don't have information about that in the books I have access to."

    # ---- 1. Remove a refusal sentence glued onto a substantive answer ----
    # (Model sometimes ends a full tutorial with the refusal line. Handle curly apostrophes.)
    refusal_pat = r"I don[’']t have information about that in the books I have access to\.?"
    without_refusal = re.sub(refusal_pat, "", answer).strip()
    if without_refusal != answer.strip() and len(without_refusal) > 200:
        answer = without_refusal
        print("  Citation check: removed contradictory refusal sentence from a real answer")

    # ---- 2. Citation check (only when we have an allow-list to check against) ----
    citation_flags: List[str] = []
    if allowed:
        # Ignore code: quoted string literals inside code are never citations.
        prose_only = re.sub(r'```.*?```', ' ', answer, flags=re.DOTALL)
        prose_only = re.sub(r'`[^`]*`', ' ', prose_only)

        # A quote is a book citation only in citation context — e.g. 'the book "X"',
        # 'a guide titled "X"'. Plain quoted phrases in prose are NOT citations.
        cite_pattern = r'(?:book|books|titled|title|guide|primer|reference|bundle)\s+["“]([^"”]{10,100})["”]'
        candidates = re.findall(cite_pattern, prose_only, flags=re.IGNORECASE)

        for cand in candidates:
            clean = cand.strip().strip(",.;:").lower()
            if not any(clean in t.lower() or t.lower() in clean for t in allowed):
                citation_flags.append(cand)

        # Redact just the offending titles — keep the rest of the answer intact.
        for bad in citation_flags:
            for q1, q2 in (('"', '"'), ("“", "”")):
                answer = answer.replace(f"{q1}{bad}{q2}", "[title removed — not in retrieved sources]")

        if citation_flags:
            print(f"  Citation check: redacted unverified title(s): {citation_flags}")

    if answer != original:
        return {
            "messages":       [AIMessage(content=answer)],
            "answer":         answer,
            "citation_flags": citation_flags,
        }

    return {"citation_flags": []}


# ==================== GRAPH ====================
workflow = StateGraph(GraphState)
workflow.add_node("rewrite_query",     rewrite_query_node)
workflow.add_node("retrieve",          retrieve_node)
workflow.add_node("generate",          generate_node)
workflow.add_node("verify_citations",  verify_citations_node)

workflow.add_edge(START,              "rewrite_query")
workflow.add_edge("rewrite_query",    "retrieve")
workflow.add_edge("retrieve",         "generate")
workflow.add_edge("generate",         "verify_citations")
workflow.add_edge("verify_citations", END)

# Persistent SQLite-backed memory: conversation history survives restarts.
_checkpointer_cm = SqliteSaver.from_conn_string(str(CHECKPOINT_DB))
memory = _checkpointer_cm.__enter__()
app    = workflow.compile(checkpointer=memory)

print("LangGraph RAG ready (hybrid search + citation verification + persistent memory).\n")


# ==================== CLI LOOP ====================
if __name__ == "__main__":
    thread_id     = "technical_books_session"
    thread_config = {"configurable": {"thread_id": thread_id}}
    print("Ask questions about your technical books ('quit' to exit):\n")
    print(f"Chat log: {CHAT_LOG_DIR / (thread_id + '.log')}\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if question.lower() in ["quit", "exit", "q", "/q"]:
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            result       = app.invoke({"messages": [HumanMessage(content=question)]}, config=thread_config)
            final_answer = result["messages"][-1].content
            print(f"\nAssistant:\n{final_answer}\n")
            log_turn(thread_id, result, question)
        except Exception as e:
            print(f"\n[Error] {e}\nCheck Ollama is running and bm25_corpus.json exists.\n")
