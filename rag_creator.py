"""
RAG - vector database builder (ChromaDB)

Reads the .txt files in 'documents/', splits them into chunks (one per
label line such as [TEMA: ..., FONTE: file.pdf, página N]) and stores the
embeddings in 'chroma_db/'.

USAGE:
    python rag_creator.py --index                    # Index documents
    python rag_creator.py --reset --index            # Wipe the database and index from scratch
    python rag_creator.py --test "what is VAT?"      # Test retrieval (no LLM)
    python rag_creator.py --stats                    # Show statistics
"""

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIG
# ============================================================

@dataclass
class RAGConfig:
    # Multilingual embedding model (the documents are in Portuguese/English)
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-small"

    # ChromaDB
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    COLLECTION_NAME: str = "financial_accounting"

    # Documents
    DOCUMENTS_DIR: str = "./documents"

    # Number of chunks retrieved per question
    N_RESULTS: int = 4


# Label at the start of each chunk, e.g. [TEMA: X, FONTE: Y.pdf, página 3]
LABEL_PATTERN = re.compile(r'^\[(?P<label>[^\]]+)\][^\n]*\n?', re.MULTILINE)
# Source runs until ", página N" / ", page N" (file names may contain commas)
SOURCE_PATTERN = re.compile(r"(?:fonte|source)\s*:\s*(.+?)\s*(?:,\s*(?:p[áa]gina|page)\b|$)", re.IGNORECASE)
PAGE_PATTERN = re.compile(r"(?:p[áa]gina|page)\s*(\d+)", re.IGNORECASE)
# Citation leftovers when text is copied from ChatGPT
CITATION_PATTERN = re.compile(r'[ \t]*\[oai_citation:[^\]]*\]\([^)]*\)')

# ============================================================
# RAG
# ============================================================

class RAG:
    """Indexing and search over the vector database."""

    def __init__(self, config: RAGConfig | None = None):
        self.config = config or RAGConfig()

        print("📊 Loading embedding model...")
        self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)

        print("💾 Opening ChromaDB...")
        self.chroma_client = chromadb.PersistentClient(
            path=self.config.CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"✅ Ready! ({self.collection.count()} chunks in the database)\n")

    # --------------------------------------------------------
    # Embeddings (E5 models require the "query: " / "passage: " prefixes)
    # --------------------------------------------------------

    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        return self.embedding_model.encode(
            [f"passage: {t}" for t in texts],
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        ).tolist()

    def embed_query(self, query: str) -> List[float]:
        return self.embedding_model.encode(
            f"query: {query}", normalize_embeddings=True
        ).tolist()

    # --------------------------------------------------------
    # Indexing
    # --------------------------------------------------------

    def parse_document(self, text: str, file_name: str) -> List[Dict[str, Any]]:
        """One chunk per label. The label is kept in the text so the LLM can see the source."""
        text = CITATION_PATTERN.sub('', text)
        matches = list(LABEL_PATTERN.finditer(text))
        chunks: List[Dict[str, Any]] = []

        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            body = text[match.end():end].strip()
            if not body:
                continue

            label = match.group('label').strip()
            source = SOURCE_PATTERN.search(label)
            page = PAGE_PATTERN.search(label)

            chunks.append({
                'text': f"[{label}]\n{body}",
                'metadata': {
                    'file': file_name,  # originating .txt file (used for re-indexing)
                    'source': source.group(1).strip() if source else file_name,
                    'page': int(page.group(1)) if page else 0,
                },
            })

        return chunks

    def index_documents(self) -> None:
        """Index every .txt in the documents folder (re-indexes files already present)."""
        documents_dir = Path(self.config.DOCUMENTS_DIR)
        print(f"📂 Looking for documents in: {documents_dir}")

        txt_files = sorted(documents_dir.glob("*.txt"))
        if not txt_files:
            print(f"❌ No .txt files found in '{documents_dir}'")
            return

        total_chunks = 0
        for txt_file in txt_files:
            print(f"📖 Processing: {txt_file.name}")
            content = txt_file.read_text(encoding='utf-8')

            # Remove this file's old chunks so it can be re-indexed
            previous = self.collection.get(where={"file": txt_file.name}, include=[])
            if previous["ids"]:
                self.collection.delete(ids=previous["ids"])
                print(f"   └─ Removed {len(previous['ids'])} old chunks")

            chunks = self.parse_document(content, txt_file.name)
            if not chunks:
                print("   ⚠️  No valid chunks - check the label format!")
                continue

            texts = [c['text'] for c in chunks]
            self.collection.add(
                ids=[f"{txt_file.stem}_{i}" for i in range(len(chunks))],
                embeddings=self.embed_passages(texts),
                documents=texts,
                metadatas=[c['metadata'] for c in chunks],
            )
            total_chunks += len(chunks)
            print(f"   ✓ {len(chunks)} chunks indexed\n")

        print(f"✅ Indexing complete! Total: {total_chunks} chunks")
        print(f"💾 Database saved to: {self.config.CHROMA_PERSIST_DIR}")

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    def search(self, query: str, n_results: int | None = None) -> List[Dict[str, Any]]:
        """Return the most relevant chunks: [{'text', 'source', 'page', 'score'}]."""
        results = self.collection.query(
            query_embeddings=[self.embed_query(query)],
            n_results=n_results or self.config.N_RESULTS,
            include=['documents', 'metadatas', 'distances'],
        )
        return [
            {
                'text': doc,
                'source': meta.get('source', 'unknown'),
                'page': meta.get('page', 0),
                'score': 1 - dist,
            }
            for doc, meta, dist in zip(
                results['documents'][0], results['metadatas'][0], results['distances'][0]
            )
        ]

    def test_retrieval(self, query: str) -> None:
        print(f"🔍 Question: '{query}'\n")
        for i, hit in enumerate(self.search(query), 1):
            preview = hit['text'].split('\n', 1)[-1][:150].replace('\n', ' ')
            print(f"📄 {i}. {hit['source']} (p. {hit['page']}) — relevance {hit['score']:.2%}")
            print(f"   {preview}...\n")

    def stats(self) -> None:
        count = self.collection.count()
        print(f"{'=' * 60}\n📊 STATISTICS\n{'=' * 60}")
        print(f"Total chunks: {count}")
        if count:
            metas = self.collection.get(include=['metadatas'])['metadatas']
            per_source: Dict[str, int] = {}
            for meta in metas:
                src = meta.get('source', 'unknown')
                per_source[src] = per_source.get(src, 0) + 1
            print("\nSources:")
            for src, n in sorted(per_source.items()):
                print(f"  • {src} ({n})")
        print(f"{'=' * 60}\n")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Build and test the RAG vector database")
    parser.add_argument('--index', action='store_true', help='Index documents')
    parser.add_argument('--test', type=str, metavar='QUESTION', help='Test retrieval with a question')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--reset', action='store_true', help='Wipe the database before starting')
    parser.add_argument('--docs-dir', type=str, default=RAGConfig.DOCUMENTS_DIR,
                        help='Documents folder (default: ./documents)')
    args = parser.parse_args()

    if not (args.index or args.test or args.stats or args.reset):
        parser.print_help()
        return

    config = RAGConfig(DOCUMENTS_DIR=args.docs_dir)

    if args.reset:
        # Deleting the whole folder also handles databases created by other ChromaDB versions
        shutil.rmtree(config.CHROMA_PERSIST_DIR, ignore_errors=True)
        print("♻️  Database wiped.\n")

    rag = RAG(config)

    if args.index:
        rag.index_documents()
        rag.stats()
    if args.test:
        rag.test_retrieval(args.test)
    if args.stats and not args.index:
        rag.stats()


if __name__ == "__main__":
    main()
