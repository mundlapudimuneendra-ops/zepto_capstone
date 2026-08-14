# Zepto Support Assistant

A RAG + LangGraph + FastAPI customer support service for Zepto's published policies. The default baseline operates **100% offline in a deterministic mock mode**: zero API keys, zero external network calls, and instant responses. An optional real-LLM mode (`MOCK_LLM=0`) adds live LLM text generation with Pydantic structured-output validation and automatic retries.

---

## 1. Architecture Walkthrough

The application is structured into four sequential stages: **Ingestion → Embedding → Retrieval → Generation**.

### 1.1 Ingestion (`ingest.py`)

* `load_documents()` loads all 8 policy files (`docs/doc_01.txt` through `docs/doc_08.txt`) in numerical order.
* `chunk_document()` splits each document into a chunk. The **complete text of each policy file is preserved verbatim in the chunk** — no title stripping, no line removal, and no content truncation.
* Chunks are assigned stable identifiers matching `<doc_id>::chunk0` (e.g., `doc_01::chunk0` through `doc_08::chunk0`).
* `build_index()` indexes all 8 chunks into a persistent ChromaDB collection.

### 1.2 Embedding

* `get_embed_model()` loads the local `sentence-transformers/all-MiniLM-L6-v2` model (cached in-process).
* Text chunks are embedded into dense vector representations and upserted into ChromaDB.

### 1.3 Retrieval (`ingest.py`)

* `retrieve(query, top_k=3)` embeds the incoming natural-language query using `sentence-transformers/all-MiniLM-L6-v2`.
* Queries the ChromaDB collection `zepto_policies` using **cosine similarity** distance metric.
* Returns top 3 matching chunks containing `id`, `text`, `doc_id`, `score`, and `similarity`.
* Retrieval is **always real** (hits local ChromaDB and embedding model) in both mock mode and real-LLM mode.

### 1.4 Generation (`graph.py` & `prompts.py`)

The pipeline uses a LangGraph `StateGraph` with **exactly three named nodes** and a conditional routing edge:

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

1. **`classify_intent`** (`graph.py`):
   - In mock mode (`MOCK_LLM=1` or unset), uses a deterministic keyword heuristic across all 8 policy areas (`delivery`, `return`, `refund`, `membership`, `tracking`, `cancel`, `damaged`, `missing`, `gift card`, `support hours`).
   - In real-LLM mode (`MOCK_LLM=0`), invokes the LLM using `CLASSIFIER_PROMPT` (`prompts.py`).
2. **`retrieve_and_answer`** (`graph.py`):
   - Runs `ingest.retrieve(query, top_k=3)`.
   - In mock mode, sets `confidence = 1.0`, `sources = [top retrieved chunk IDs]`, and answer starting with exact prefix: `Based on the retrieved context: <top chunk snippet>`.
   - In real-LLM mode, sends `POLICY_RAG_PROMPT` (`prompts.py`), parses JSON output, and strictly validates using `SupportResponse(**parsed)`.
3. **`direct_answer`** (`graph.py`):
   - For non-policy questions, sets `sources = []`, `confidence = 1.0`, and canned mock answer: `I can only answer questions about Zepto policies right now.`
   - In real-LLM mode, uses `DIRECT_ANSWER_PROMPT` (`prompts.py`) with structured JSON parsing and validation.

---

## 2. Dynamic `MOCK_LLM` Behavior & Structured Output Validation

`MOCK_LLM` is evaluated **dynamically inside node execution** via `is_mock_mode()` — it is never permanently assigned at module import time.

* **`MOCK_LLM` unset or `"1"`**: Deterministic, 100% offline mock baseline.
* **`MOCK_LLM="0"`**: Real-LLM mode.

### Structured Output Parsing & Retries (`MOCK_LLM=0`)

In real-LLM mode, nodes enforce strict JSON output matching the `SupportResponse` Pydantic schema:

```json
{
  "answer": "string",
  "sources": ["string"],
  "confidence": 0.0
}
```

If parsing or Pydantic `SupportResponse` validation fails:
* Retries up to **2 additional times** (3 total attempts).
* Appends explicit corrective instructions in the retry prompt.
* If all 3 attempts fail, returns a clearly marked error response (`"ERROR: failed to generate a valid structured response after 3 attempts"`) with `confidence = 0.0`.

---

## 3. Running & Testing

### 3.1 Index Ingestion & CLI Demo

```bash
python ingest.py
python main.py
```

### 3.2 Running FastAPI Local Server

```bash
uvicorn main:app --host 0.0.0.0 --port 7860
```

### 3.3 Endpoint Demonstration (`POST /ask`)

#### Policy Question

**Request**:
```json
POST http://127.0.0.1:7860/ask
{
  "query": "What is the delivery policy?"
}
```

**Response (HTTP 200 OK)**:
```json
{
  "answer": "Based on the retrieved context: Delivery Policy Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee. Priority delivery, which reserves the next available rider slot, is available at checkout for an additional INR 15. Zepto does not currently deliver to addresses outside its listed serviceable pin codes.",
  "sources": [
    "doc_01::chunk0",
    "doc_05::chunk0",
    "doc_02::chunk0"
  ],
  "confidence": 1.0
}
```

#### General Question

**Request**:
```json
POST http://127.0.0.1:7860/ask
{
  "query": "What is the capital of France?"
}
```

**Response (HTTP 200 OK)**:
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

---

## 4. File Structure

```
support_assistant/
├── docs/                 # Policy corpus files (doc_01.txt .. doc_08.txt)
├── chroma_store/         # Persistent ChromaDB vector database
├── ingest.py             # Document loading, full-text chunking, embedding & retrieval
├── graph.py              # LangGraph StateGraph (classify_intent, retrieve_and_answer, direct_answer)
├── main.py               # FastAPI application exposing POST /ask
├── prompts.py            # Real-LLM prompt templates (ROLE, CONTEXT, TASK, FORMAT, LENGTH, FEW-SHOT)
├── schema.py             # Pydantic request/response models & TypedDict state
├── Dockerfile            # Container definition for service deployment
└── requirements.txt      # Python dependencies
```

---

## 5. Verification Checklist

- [x] **Full-text corpus chunking**: All 8 docs preserved verbatim with IDs `doc_01::chunk0` .. `doc_08::chunk0` in `ingest.py`.
- [x] **Dynamic `MOCK_LLM`**: Environment variable checked dynamically inside node execution via `is_mock_mode()`.
- [x] **Real-LLM structured output**: `SupportResponse` Pydantic model validation with 3 total attempts and corrective retry prompts.
- [x] **Mock confidence & sources**: Deterministic `confidence = 1.0`, policy `sources = [chunk IDs]`, general `sources = []`.
- [x] **Graph structure**: Exactly 3 named nodes (`classify_intent`, `retrieve_and_answer`, `direct_answer`) and conditional routing edge.
- [x] **ChromaDB Retrieval**: `sentence-transformers/all-MiniLM-L6-v2` query embedding, top 3 retrieval via cosine similarity.
- [x] **Exact mock output formatting**: Policy answer starts with `Based on the retrieved context:`, general answer is `I can only answer questions about Zepto policies right now.`.
- [x] **FastAPI Verification**: Tested `uvicorn main:app --port 7860` with `POST /ask` for policy and general queries.
- [ ] **Docker Execution**: Skipped (Docker CLI is not installed on the current host machine).

---

## 🛠️ Troubleshooting

- **ChromaDB Indexing Warnings / Telemetry**: `ingest.py` initializes `Settings(anonymized_telemetry=False, allow_reset=False)` to run completely offline without external analytics network requests.
- **Model Download on First Run**: First run of `ingest.py` or startup auto-indexing downloads `sentence-transformers/all-MiniLM-L6-v2` (~90MB). Subsequent runs use the cached weights in HuggingFace cache.
- **Real-LLM API Key Missing**: When `MOCK_LLM=0` is enabled, ensure `OPENAI_API_KEY` environment variable is set. If missing or invalid, the pipeline automatically falls back to keyword intent classification and deterministic mock answers to prevent request failure.

---

## 👨‍💻 Author

**Mundlapudi Muneendra**

AI & Machine Learning Student

GitHub:
https://github.com/mundlapudimuneendra-ops

---

## 📄 License

This project was developed for educational purposes as part of the Zepto Support Assistant Capstone.


