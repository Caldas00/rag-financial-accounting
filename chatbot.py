"""
RAG - simple web chatbot for testing

Finds the most relevant chunks in ChromaDB and sends them, together with the
question, to an AI model through an API (OpenAI by default).

USAGE:
    python rag_creator.py --index   # once, to build the database
    python chatbot.py               # opens http://127.0.0.1:7860
"""

import os
import sys
import warnings

# Keep the terminal quiet: no download/loading progress bars or library warnings
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

from rag_creator import RAG, RAGConfig

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ============================================================
# CONFIG (via .env)
# ============================================================

API_KEY = os.getenv("AI_KEY")
MODEL = os.getenv("AI_MODEL", "gpt-6-luna")
# Reasoning models only (GPT-6, GPT-5.x, o-series). Set AI_REASONING_EFFORT= (empty) for other models
REASONING_EFFORT = os.getenv("AI_REASONING_EFFORT", "low")
BASE_URL = os.getenv("AI_BASE_URL") or None  # optional: any OpenAI-compatible API
MAX_HISTORY_MESSAGES = 6  # previous messages sent to the model

SYSTEM_PROMPT = """You are a study assistant for a Financial Accounting course.
Answer using ONLY the context excerpts provided below. If the answer is not in the
context, say you could not find it in the course materials.
Always answer in English, even when the question or the context is in Portuguese
(translate the content as needed). Be clear and concise, show formulas
and journal entries (debit/credit) when relevant, and cite the source and page
you used, e.g. (4. Inventories.pdf, p. 3).

CONTEXT:
{context}"""

if not API_KEY:
    sys.exit("❌ AI_KEY is missing. Copy .env.example to .env and add your API key.")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
rag = RAG(RAGConfig(), verbose=False)

if rag.collection.count() == 0:
    sys.exit("❌ The database is empty. Run first: python rag_creator.py --index")


def _text(content) -> str:
    """Gradio may store a message as a string or as a list of content parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content or "")


def chat(message: str, history: list):
    hits = rag.search(message)
    context = "\n\n---\n\n".join(hit["text"] for hit in hits)

    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
    for msg in history[-MAX_HISTORY_MESSAGES:]:
        messages.append({"role": msg["role"], "content": _text(msg["content"])})
    messages.append({"role": "user", "content": message})

    sources = "\n".join(
        f"- {hit['source']}, p. {hit['page']} ({hit['score']:.0%})" for hit in hits
    )
    footer = f"\n\n<details><summary>📚 Retrieved sources</summary>\n\n{sources}\n</details>"

    try:
        extra = {"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else {}
        stream = client.chat.completions.create(model=MODEL, messages=messages, stream=True, **extra)
        answer = ""
        for event in stream:
            if event.choices and event.choices[0].delta.content:
                answer += event.choices[0].delta.content
                yield answer
        yield answer + footer
    except Exception as exc:
        yield f"❌ API error: {exc}"


demo = gr.ChatInterface(
    fn=chat,
    title="📊 Financial Accounting — RAG Chatbot",
    description=f"Answers based on the course materials. Model: `{MODEL}`",
    examples=[
        "What is the balance sheet?",
        "How is days sales outstanding calculated?",
        "When prices rise, which inventory method gives the lowest COGS?",
    ],
)

if __name__ == "__main__":
    demo.launch(quiet=True, prevent_thread_lock=True)
    print(f"💬 Chatbot running at {demo.local_url}  (Ctrl+C to stop)", flush=True)
    try:
        demo.block_thread()
    except KeyboardInterrupt:
        pass
