# RAG — Financial Accounting

A small **RAG** (Retrieval-Augmented Generation) system over Financial Accounting
course notes, with a simple web chatbot to test it.

You ask a question → the most relevant passages from the notes are retrieved →
an AI model answers **using only those passages** and cites the source PDF and page.

## Quick start

Requires **Python 3.10+** and an API key (OpenAI by default).

```bash
git clone git@github.com:Caldas00/rag-financial-accounting.git
cd rag-financial-accounting

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then put your API key in .env (AI_KEY=...)

python rag_creator.py --index      # build the vector database (~1 min the first time)
python chatbot.py                  # open http://127.0.0.1:7860
```

The first run downloads the embedding model (~470 MB), which is then cached.

## How it works

```
documents/*.txt ──► rag_creator.py ──► chroma_db/  (vector database)
                     split into chunks,
                     compute embeddings
                                            │
question ──► chatbot.py ──► top 4 chunks ───┘
                  │
                  └──► AI API (question + chunks) ──► answer with sources
```

1. **Chunking** — `documents/tudo.txt` is split at every label line. Each label
   starts a chunk (one slide/page of the course):

   ```
   [TEMA: Financial Accounting, FONTE: 4. Inventories.pdf, página 3]
   # Slide title
   - content...
   ```

   The source (`FONTE`) and page (`página`) are stored as metadata, and the label is
   kept in the chunk text so the model can cite it. English labels
   (`SOURCE: ..., page N`) also work.

2. **Embeddings** — each chunk is turned into a vector with
   [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small),
   a small multilingual model (works well for Portuguese + English). It runs on
   your machine, on CPU, and is free.

3. **Storage** — vectors are stored in [ChromaDB](https://www.trychroma.com/) in
   `chroma_db/` (not committed, rebuild it with `--index`).

4. **Answering** — `chatbot.py` embeds the question, fetches the 4 closest chunks,
   and sends them to the AI model with instructions to answer only from that context.
   The retrieved sources are shown under each answer.

## Commands

| Command | What it does |
| --- | --- |
| `python rag_creator.py --index` | Index all `.txt` files in `documents/` (re-indexing a file replaces its old chunks) |
| `python rag_creator.py --reset --index` | Delete the database and rebuild from scratch |
| `python rag_creator.py --test "what is VAT?"` | Show the chunks retrieved for a question (no AI call) |
| `python rag_creator.py --stats` | Chunks per source |
| `python chatbot.py` | Start the web chatbot |

## Adding documents

Put more `.txt` files (UTF-8) in `documents/`, using the label format above, and
run `python rag_creator.py --index` again. Keep each chunk reasonably short (one
slide/topic): the embedding model only reads the first ~512 tokens of each chunk.

## Configuration

In `.env`:

| Variable | Default | Description |
| --- | --- | --- |
| `AI_KEY` | — | API key (required) |
| `AI_MODEL` | `gpt-4.1-mini` | Model name |
| `AI_BASE_URL` | OpenAI | Any OpenAI-compatible API, e.g. Groq, OpenRouter, Gemini |

Retrieval settings (embedding model, number of chunks) are in `RAGConfig` in
`rag_creator.py`. If you change the embedding model, rebuild with `--reset --index`.

## Project structure

```
├── rag_creator.py     # chunking, embeddings, ChromaDB indexing and search
├── chatbot.py         # Gradio web chatbot (calls the AI API)
├── documents/
│   └── tudo.txt       # course notes, one labelled chunk per slide
├── requirements.txt
└── .env.example
```

## Known limitations

- `5. Taxes and Gov acc.pdf` and `8. Leases.pdf` only have a one-page summary, so
  questions on those topics get less detail. Operating leases, for example, are not covered.
- Retrieval uses only the current question, so a vague follow-up like "and the
  other one?" may retrieve the wrong passages. Ask complete questions.
