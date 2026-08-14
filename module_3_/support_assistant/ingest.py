"""Ingest Zepto policy docs into a ChromaDB collection.

Responsibilities:
- Load every ``docs/doc_*.txt`` file in the order they are numbered.
- Chunk each document (per-document is fine — the docs are short).
- Embed with a local ``sentence-transformers`` model.
- Persist vectors into a ChromaDB collection (cosine similarity).

The module is runnable as a script:

    python ingest.py

It writes to a local ``chroma_store/`` directory inside the project
folder and is idempotent — re-running upserts the same chunks.

No network calls and no API keys are used.
"""

from __future__ import annotations

import glob
import os
from typing import List, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_store")
COLLECTION_NAME = "zepto_policies"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Lazy-loaded so importing the module is cheap.
_embed_model: SentenceTransformer | None = None
_chroma_client: chromadb.PersistentClient | None = None


# ---------------------------------------------------------------------------
# Lazy initializers
# ---------------------------------------------------------------------------

def get_embed_model() -> SentenceTransformer:
    """Load (and cache) the local sentence-transformers model."""

    global _embed_model
    if _embed_model is None:
        try:
            _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        except Exception:
            _embed_model = SentenceTransformer(EMBED_MODEL_NAME, local_files_only=True)
    return _embed_model


def get_chroma_client() -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client rooted at CHROMA_DIR."""

    global _chroma_client
    if _chroma_client is None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False, allow_reset=False),
        )
    return _chroma_client


def get_collection():
    """Return the (existing or freshly-created) policy collection."""

    client = get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Loading + chunking
# ---------------------------------------------------------------------------

def _doc_sort_key(path: str) -> int:
    """Sort by the trailing integer in the filename: doc_01.txt -> 1."""

    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    try:
        return int(stem.split("_")[-1])
    except (ValueError, IndexError):
        return 1_000_000


def load_documents() -> List[Tuple[str, str]]:
    """Read every ``docs/doc_NN.txt`` file in numerical order.

    Returns a list of ``(doc_id, text)`` tuples. The doc_id is the
    filename without extension (e.g. ``doc_01``).
    """

    pattern = os.path.join(DOCS_DIR, "doc_*.txt")
    paths = sorted(glob.glob(pattern), key=_doc_sort_key)
    docs: List[Tuple[str, str]] = []
    for p in paths:
        doc_id = os.path.splitext(os.path.basename(p))[0]
        with open(p, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            continue
        docs.append((doc_id, text))
    return docs


def chunk_document(doc_id: str, text: str) -> List[Tuple[str, str]]:
    """Split a document into chunks.

    The corpus docs are short (a single paragraph each), so per-document
    is a reasonable chunk strategy. The COMPLETE contents of each document
    (doc_01.txt through doc_08.txt) are preserved verbatim in the chunk — no
    first-line / title stripping, no body truncation, no reformatting. Each
    chunk is tagged with a stable id of the form `<doc_id>::chunk0`
    (e.g. `doc_01::chunk0`).
    """

    chunk_text = text.strip()
    if not chunk_text:
        return []
    return [(f"{doc_id}::chunk0", chunk_text)]


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def build_index(force: bool = False) -> int:
    """Embed and upsert every chunk. Returns the number of chunks indexed.

    ``force=True`` deletes the collection first, which is useful for
    local development; in production we'd rather upsert in place.
    """

    client = get_chroma_client()
    if force:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = get_collection()
    docs = load_documents()

    ids: List[str] = []
    texts: List[str] = []
    metadatas: List[dict] = []
    for doc_id, text in docs:
        for chunk_id, chunk_text in chunk_document(doc_id, text):
            ids.append(chunk_id)
            texts.append(chunk_text)
            # Splitting the chunk id on "::" lets downstream code tell
            # the doc id apart from the chunk offset without parsing.
            metadatas.append({"doc_id": doc_id})

    if not texts:
        return 0

    model = get_embed_model()
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    return len(texts)


def retrieve(query: str, top_k: int = 3) -> List[dict]:
    """Embed ``query`` and return the top-k chunks by cosine similarity.

    Each returned dict has ``id``, ``text``, ``doc_id``, and ``score``
    keys. ``score`` is the cosine distance (lower = more similar)
    exposed by ChromaDB; we also expose ``similarity = 1 - score`` for
    convenience.
    """

    collection = get_collection()
    model = get_embed_model()
    q_emb = model.encode([query], normalize_embeddings=True).tolist()

    res = collection.query(
        query_embeddings=q_emb,
        n_results=min(top_k, collection.count() or 1),
    )

    out: List[dict] = []
    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    for i, doc_id, text, meta, dist in zip(
        range(len(ids)), ids, docs, metas, dists
    ):
        out.append({
            "id": doc_id,
            "text": text,
            "doc_id": (meta or {}).get("doc_id", doc_id.split("::")[0]),
            "score": float(dist),
            "similarity": max(0.0, 1.0 - float(dist)),
            "rank": i,
        })
    return out


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    n = build_index(force=True)
    print(f"Indexed {n} chunks into '{COLLECTION_NAME}' at {CHROMA_DIR}")
