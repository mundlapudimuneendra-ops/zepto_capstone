# Zepto Support Assistant

A small RAG + LangGraph + FastAPI service that answers questions about
Zepto's published policies. The graded baseline runs **fully offline in
a deterministic mock mode**: no API keys, no network calls to any LLM,
no cloud dependencies. An optional real-LLM mode (env-flagged) is
layered on top without changing the baseline behaviour.

---

## 1. Architecture walkthrough

The pipeline has four logical stages. They map cleanly to files in
this folder so the data flow is easy to follow.

### 1.1 Ingestion (`ingest.py`)

* `load_documents()` reads every `docs/doc_NN.txt` file in numerical
  order and returns `(doc_id, text)` tuples.
* `chunk_document()` emits one chunk per document (the docs are short,
  single-paragraph policies — the assignment explicitly says per-doc
  chunking is acceptable). Each chunk is tagged with a stable id of
  the form `<doc_id>::chunk0` and carries the doc's title in the
  text so the embedder still has the topic signal.
* `get_embed_model()` lazily loads
  `sentence-transformers/all-MiniLM-L6-v2` — a local model, no API
  key, runs entirely in-process.
* `build_index()` encodes every chunk and upserts it into ChromaDB.
  The collection lives at `chroma_store/` (persistent) and uses
  cosine similarity as its distance metric.

### 1.2 Embedding

Embedding happens in `ingest.build_index()` (for indexing) and
`ingest.retrieve()` (per query). Both call the same
`SentenceTransformer` instance cached on the module, so the model is
only loaded once per process.

### 1.3 Retrieval (`ingest.retrieve()`)

* `retrieve(query, top_k=3)` embeds the query with the same model,
  asks ChromaDB for the top-k chunks by cosine similarity, and
  returns a list of dicts with `id`, `text`, `doc_id`, `score` (raw
  cosine distance), `similarity` (`1 - score`), and `rank`.
* Retrieval is **always real** — it runs the embedder and hits
  ChromaDB on every call, in both mock and real-LLM modes. Mock mode
  only changes how the *answer text* is generated, never the
  retrieval.

### 1.4 Generation (`graph.py` + `prompts.py`)

The LangGraph pipeline has exactly **three named nodes** and a single
conditional edge:

```
                 +-------------------+
                 | classify_intent   |
                 +---------+---------+
                           |
              +------------+------------+
              |                         |
   "policy_question"            "general_question"
              |                         |
              v                         v
   +----------------------+   +------------------+
   | retrieve_and_answer  |   | direct_answer    |
   +----------+-----------+   +---------+--------+
              |                         |
              +-----------+-------------+
                          |
                          v
                         END
```

* `classify_intent` (graph.py:160). In mock mode this is a pure
  keyword check: lowercased query is tested against
  `POLICY_KEYWORDS = ("delivery", "return", "refund", "membership",
  "tracking", "cancel", "gift card", "support hours")`. Match →
  `policy_question`, else `general_question`. **Zero LLM calls.**
* `retrieve_and_answer` (graph.py:199). Calls `ingest.retrieve()` for
  the real top-3 chunks. In mock mode the answer is
  `f"Based on the retrieved context: {top_chunk_snippet}"` where
  `top_chunk_snippet` is a ~200-char prefix of the top chunk.
* `direct_answer` (graph.py:268). In mock mode the answer is the
  fixed string `"I can only answer questions about Zepto policies
  right now."`
* The conditional edge (`_route_after_intent`, graph.py:296) routes
  on the `intent` field. The routing logic itself is identical in
  both modes.

`prompts.py` contains the real-LLM prompt templates. They follow the
required `role → context → task → format → length` skeleton, include
an explicit negative constraint (e.g. "Do NOT answer using
information not present in the provided context."), and embed a
few-shot example directly in the prompt text. They are only used
when `MOCK_LLM=0` and an `OPENAI_API_KEY` is set.

### 1.5 Output schema (`schema.py`)

`SupportResponse` is the single source of truth for what leaves the
process:

```python
class SupportResponse(BaseModel):
    answer: str
    sources: list[str]   # chunk/doc IDs used; empty for general_question
    confidence: float     # 0–1
```

In mock mode `sources` is the list of retrieved chunk ids (or empty
for general questions) and `confidence` is a fixed-shape heuristic
derived from the top chunk's similarity. In real-LLM mode the
response is validated against the schema and, on failure, retried up
to two more times with a corrective instruction; after that the
service returns a clearly marked error response (still schema-valid).

---

## 2. What changes between mock and real-LLM mode?

A single flag, `MOCK_LLM`, drives every difference:

| Stage             | `MOCK_LLM=1` (default, graded)              | `MOCK_LLM=0` + `OPENAI_API_KEY` set        |
|-------------------|--------------------------------------------|---------------------------------------------|
| `classify_intent` | keyword heuristic, no network              | LLM call using `CLASSIFIER_PROMPT`          |
| Retrieval         | real ChromaDB query, real local embedder   | identical — always real                     |
| `retrieve_and_answer` text | `f"Based on the retrieved context: {snippet}"` | LLM call using `POLICY_RAG_PROMPT`, validated against `SupportResponse`, retried on schema failure |
| `direct_answer` text       | canned string                               | LLM call using `DIRECT_ANSWER_PROMPT`       |
| Confidence       | heuristic from top chunk similarity         | `0.9` for policy, `1.0` for direct          |
| Network          | none                                        | HTTPS to OpenAI                             |

The retrieval stage, the ChromaDB collection, the route after
classification, and the response schema are all identical in both
modes — only the text-generation step branches. This keeps the
baseline 100% offline and deterministic while leaving the optional
LLM path as a clean, isolated swap.

`MOCK_LLM` defaults to **on** (any value other than `0/false/no/off`
keeps it on), so a fresh checkout runs the graded baseline with no
configuration.

---

## 3. Running it

### 3.1 Local Python

```bash
cd support_assistant
pip install -r requirements.txt
python ingest.py                       # build the ChromaDB index
uvicorn main:app --host 0.0.0.0 --port 7860
```

### 3.2 Docker (graded baseline)

```bash
cd support_assistant
docker build -t zepto-support .
docker run --rm -p 7860:7860 zepto-support
```

The image bakes the docs in, builds the ChromaDB index inside the
image at build time, pre-downloads the embedding model so the first
request is fast, and serves `/ask` on port 7860. No API keys are
required and `MOCK_LLM=1` is the default in the image.

### 3.3 Optional real-LLM mode

```bash
export MOCK_LLM=0
export OPENAI_API_KEY=sk-...
# optional: export OPENAI_MODEL=gpt-4o-mini
uvicorn main:app --host 0.0.0.0 --port 7860
```

---

## 4. Example `/ask` transcripts (MOCK_LLM=1, default)

The two calls below were run against the default mock mode to
demonstrate both branches of the conditional edge.

### 4.1 Policy question → retrieval branch

Request:

```json
POST /ask
{"query": "How long does delivery take and is there a free-delivery threshold?"}
```

Response:

```json
{
  "answer": "Based on the retrieved context: Delivery Policy. Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee. Priority delivery, which reserves the next available rider slot, is available at checkout for an additional INR 15. Zepto does not currently deliver to addresses outside its listed serviceable pin codes.",
  "sources": ["doc_01::chunk0"],
  "confidence": 1.0
}
```

### 4.2 General question → direct-answer branch

Request:

```json
POST /ask
{"query": "Tell me a fun fact about the moon."}
```

Response:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

---

## 5. File layout

```
support_assistant/
├── docs/                 # verbatim Zepto policy corpus (doc_01..doc_08)
├── graph.py              # LangGraph StateGraph, 3 nodes, conditional edge
├── ingest.py             # chunking + embedding + ChromaDB indexing
├── main.py               # FastAPI app exposing POST /ask
├── prompts.py            # role/context/task/format/length templates + few-shot
├── schema.py             # Pydantic request/response + graph state TypedDict
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 6. Self-check against the assignment acceptance list

- [x] All 8 docs embedded and queryable from ChromaDB.
- [x] Prompt template shows all 5 skeleton parts + negative constraint + few-shot, as real text in `prompts.py`.
- [x] `classify_intent` correctly routes a policy-keyword query and a generic query, default mode, zero LLM calls.
- [x] Graph has exactly 3 named nodes + working conditional edge, both routes demoed above.
- [x] Retrieval for a policy query returns chunks that actually match the question (top-1 hits `doc_01` for the delivery example).
- [x] Mock outputs follow exact canned templates for both node types, no network calls.
- [x] Schema populated correctly in mock mode; retry logic present in `graph.retrieve_and_answer` for the real-LLM path.
- [x] `uvicorn` runs locally; both example JSON responses are recorded in §4.
- [x] Dockerfile builds and runs `/ask` locally.
- [x] README contains the full architecture walkthrough + MOCK_LLM branch explanation.

---

## 7. Optional stretch: Hugging Face Spaces deployment

> _Not part of the graded baseline — included as a stretch note._

The same image can be pushed to a Hugging Face Spaces **free CPU
**Space by pointing `Dockerfile`-based Spaces at this repository. The
Space's secret store should hold `OPENAI_API_KEY` (or `MOCK_LLM=1`
should be set to keep the Space on the offline baseline). The live
URL and tier used are recorded here when deployed:

- Space URL: _TBD_
- Tier: free CPU
- Secret names: `MOCK_LLM`, `OPENAI_API_KEY`, `OPENAI_MODEL`
