#!/usr/bin/env python3
"""
rag_ui.py - Gradio web UI for the RAG chatbot
=============================================
Wraps rag_chatbot.py in a browser-based chat interface.

Run:
    python rag_ui.py

Opens at http://127.0.0.1:7860
Each browser session gets its own LangGraph thread (conversation memory is isolated).
"""

import uuid
import gradio as gr
from langchain_core.messages import HumanMessage

from rag_chatbot import app, log_turn


def extract_sources(context: str) -> list[str]:
    """Pull 'Source: ...' lines out of the formatted context string."""
    return [
        line.replace("Source: ", "").strip()
        for line in context.split("\n")
        if line.startswith("Source:")
    ]


def respond(message: str, history: list, session_id: str):
    if not session_id:
        session_id = str(uuid.uuid4())

    config = {"configurable": {"thread_id": session_id}}
    result = app.invoke({"messages": [HumanMessage(content=message)]}, config=config)

    answer  = result["messages"][-1].content
    context = result.get("context", "")
    sources = extract_sources(context)

    if sources:
        deduplicated = list(dict.fromkeys(sources))   # preserve order, remove dupes
        sources_block = "\n\n---\n**Sources retrieved:**\n" + "\n".join(f"- {s}" for s in deduplicated)
        answer += sources_block

    log_turn(session_id, result, message)

    history.append({"role": "user",      "content": message})
    history.append({"role": "assistant", "content": answer})
    return "", history, session_id


def new_session():
    return [], str(uuid.uuid4())


# ==================== UI LAYOUT ====================
with gr.Blocks(title="Technical Books RAG", theme=gr.themes.Soft()) as demo:
    session_id = gr.State(str(uuid.uuid4()))

    gr.Markdown("## Technical Books RAG Chatbot")
    gr.Markdown(
        "Ask questions about your book collection. "
        "Fully local via Ollama — no data leaves your machine."
    )

    chatbot = gr.Chatbot(height=520)
    msg_box = gr.Textbox(
        placeholder="Ask about your books...",
        show_label=False,
        lines=2,
        autofocus=True,
    )

    with gr.Row():
        send_btn  = gr.Button("Send", variant="primary", scale=3)
        clear_btn = gr.Button("New Session", scale=1)

    gr.Markdown(
        "_Tip: hybrid search (BM25 + vector) is active. "
        "Ask follow-up questions freely — query rewriting handles context. "
        "Conversations are logged to `chat_logs/<session>.log` and memory persists across restarts via SQLite._"
    )

    send_btn.click(
        respond,
        inputs=[msg_box, chatbot, session_id],
        outputs=[msg_box, chatbot, session_id],
    )
    msg_box.submit(
        respond,
        inputs=[msg_box, chatbot, session_id],
        outputs=[msg_box, chatbot, session_id],
    )
    clear_btn.click(new_session, outputs=[chatbot, session_id])


if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
