# Self-RAG — Diabetes Clinical Knowledge Assistant

A multi-strategy Retrieval-Augmented Generation (RAG) system built on LangChain + Groq + Pinecone.
The system routes each user query to the best retrieval strategy automatically and self-corrects
when the retrieved information is not relevant or when the generated answer is not grounded.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│         rag_retrieval_pipeline.py   │  ← Main entry point
│              answer(query)          │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│           rag_adaptive.py           │  ← Step 1: Route the query
│              router(query)          │
└──────┬──────────────┬───────────────┘
       │              │
  "no_retrieval"   "single" or "rag_fusion"
       │              │
       ▼              ▼
┌────────────┐  ┌─────────────────────────────────────┐
│no_rag_     │  │  "single"         "rag_fusion"       │
│retrieval.py│  │  query_pinecone   rag_fusion.py      │
│            │  │  (1 search)       (5 queries + RRF)  │
│LLM answers │  └──────────────────┬──────────────────┘
│from general│                     │
│knowledge   │            list of chunks (Document / dict)
└─────┬──────┘                     │
      │                            ▼
      │              ┌─────────────────────────────────┐
      │              │          rag_self.py             │  ← Step 2: Self-RAG
      │              │  is_rel(query, chunks)           │
      │              │  LLM grades each chunk: relevant?│
      │              └──────────┬──────────────────────┘
      │                         │
      │               relevant chunks found?
      │                  NO ──────────────────────────────────────────┐
      │                  YES                                          │
      │                         ▼                                     ▼
      │              ┌─────────────────────┐            ┌────────────────────────┐
      │              │  generate_response  │            │   rag_corrective.py    │
      │              │  (rag_self.py)      │            │   corrective_isrel()   │
      │              │  Build context from │            │   Retry 0: wider k=20  │
      │              │  rel chunks → LLM  │            │   Retry 1: rewrite     │
      │              └──────────┬──────────┘            │   query clinically     │
      │                         │                       │   Retry 2: escalate    │
      │                         ▼                       └────────────────────────┘
      │              ┌─────────────────────┐
      │              │     rag_self.py      │  ← Step 3: Grounding check
      │              │   sup_check()        │
      │              │   Is answer grounded │
      │              │   in the chunks?     │
      │              └──────────┬──────────┘
      │                         │
      │               fully supported?
      │                  YES ─────────────────────────────┐
      │                  NO                               │
      │                         ▼                        │
      │              ┌─────────────────────┐             │
      │              │  rag_corrective.py  │             │
      │              │  corrective_issup() │             │
      │              │  Retry 0: surgical  │             │
      │              │  fix (remove bad    │             │
      │              │  claims)            │             │
      │              │  Retry 1-2: full    │             │
      │              │  regeneration       │             │
      │              └──────────┬──────────┘             │
      │                         │ (loop max 3x)          │
      └─────────────────────────┴───────────────────────►│
                                                         ▼
                                             Final Answer returned
                                             as dict with metadata
```

---

## File-by-File Explanation

### Entry Point

| File | What it does |
|------|-------------|
| `rag_retrieval_pipeline.py` | **Main orchestrator.** Calls the router, picks a retrieval strategy, runs Self-RAG relevance check, generates the answer, runs grounding check, and applies corrective actions if needed. Returns a structured dict with the answer and metadata. |

### RAG Strategies

| File | What it does |
|------|-------------|
| `rag_adaptive.py` | **Router.** Sends the query to the LLM which classifies it as `no_retrieval`, `single`, or `rag_fusion`. Returns a JSON dict with strategy, complexity, and reason. |
| `no_rag_retrieval.py` | **Direct LLM answer.** Used when the router says `no_retrieval`. Answers from general knowledge — no Pinecone call at all. |
| `rag_fusion.py` | **RAG Fusion retrieval.** Generates 4 query variants → retrieves top-5 chunks per variant (5 queries × 5 chunks = up to 25 chunks) → applies **Reciprocal Rank Fusion (RRF)** to score and deduplicate → returns top-5 best chunks as `list[dict]`. |

### Self-RAG (Reflection & Grounding)

| File | What it does |
|------|-------------|
| `rag_self.py` | **Three functions:** `is_rel()` — LLM judges each chunk for relevance, returns only relevant ones as `list[dict]`. `generate_response()` — builds context from relevant chunks and calls LLM. `sup_check()` — LLM checks if every claim in the answer is backed by a source chunk. Returns `fully / partially / not_supported` plus a list of unsupported claims. |

### Corrective RAG (Self-Healing)

| File | What it does |
|------|-------------|
| `rag_corrective.py` | **Two corrective functions:** `corrective_isrel()` — fires when `is_rel()` finds no relevant chunks. Retry 0: widens search to k=20. Retry 1: rewrites query using clinical terminology. Retry 2: escalates. `corrective_issup()` — fires when `sup_check()` finds unsupported claims. Retry 0: surgical fix (remove only bad sentences). Retry 1-2: full regeneration with stricter grounding prompt. |

### Ingestion Pipeline

| File | What it does |
|------|-------------|
| `main_ingestion.py` | Loads `Diabetes_RAG_Practice.pdf`, splits into 1000-char chunks with 200-char overlap, adds metadata (`source`, `chunk_id`, `subject`), writes to Pinecone index `diabetes`. Run once to populate the vector store. |

### Core Modules

| File | What it does |
|------|-------------|
| `core/llm.py` | Initialises Groq LLM (`openai/gpt-oss-120b`, temperature=0). Shared by all modules. |
| `core/embeddings.py` | Loads HuggingFace `all-MiniLM-L6-v2` (384-dim). Used by vectorstore. |
| `core/vectorstore.py` | `write_embeddings_to_pinecone()` — stores chunks. `query_pinecone(query, k=5)` — similarity search. |
| `core/textsplitter.py` | `RecursiveCharacterTextSplitter` with chunk_size=1000, overlap=200. |
| `ingestion/pdfload.py` | Wraps `PyPDFLoader` to load any PDF into LangChain `Document` objects. |

---

## How to Run

```bash
# 1. Install dependencies
uv sync

# 2. Ingest the PDF into Pinecone (run once)
python main_ingestion.py

# 3. Run the pipeline
python rag_retrieval_pipeline.py
```

---

## Sample Test Scenarios

### Lane 1 — Simple (no retrieval needed)
These route to `no_retrieval`. The LLM answers from general knowledge. Pinecone is **not called**.

| # | Query |
|---|-------|
| 1 | What does HbA1c stand for? |
| 2 | What does GLP-1 stand for? |
| 3 | What is insulin? |
| 4 | What does BMI stand for? |
| 5 | What is the pancreas? |

**Expected:** `strategy: no_retrieval`, answer from LLM general knowledge, `chunks_used: 0`

---

### Lane 2 — Single retrieval (Self-RAG pipeline)
These use clinical language that maps directly to PDF content. One Pinecone search finds the answer.
`is_rel` filters chunks → `generate_response` → `sup_check` verifies grounding.

| # | Query |
|---|-------|
| 6 | How many adults globally have diabetes according to the IDF 2023 Atlas? |
| 7 | What is the projected number of people with diabetes by 2045? |
| 8 | What percentage of diabetes cases does Type 2 diabetes account for? |
| 9 | What HLA alleles are associated with Type 1 diabetes genetic risk? |
| 10 | What is LADA and how is it different from Type 1 diabetes? |
| 11 | What are the risk factors for developing Type 2 diabetes? |

**Expected:** `strategy: single`, one Pinecone search, `is_rel` and `sup_check` both pass clean

---

### Lane 3 — RAG Fusion (informal / vague language)
These use everyday language that won't match clinical terms in the PDF directly.
RAG Fusion generates 4 clinical variants and fuses results with RRF to bridge the vocabulary gap.

| # | Query |
|---|-------|
| 41 | Which diabetes drug is best for losing weight? |
| 42 | Which pill helps the heart in diabetics? |
| 43 | What injection do diabetics take once a week? |
| 44 | What is the diabetes eye problem called? |
| 45 | Can diabetes damage your kidneys? |
| 46 | What happens to your feet in diabetes? |
| 47 | Which diabetes drug works on the kidneys to lower sugar? |
| 48 | What is the sugar level test that shows 3 months average? |

**Expected:** `strategy: rag_fusion`, 5 query variants generated, RRF reranking applied, top-5 chunks sent to LLM

---

## Data Flow Summary

```
PDF
 └─► PyPDFLoader ──► chunk (1000 chars) ──► Pinecone (index: diabetes)
                                                   │
User Query ──► Adaptive Router                     │
                    │                              │
          ┌─────────┴────────┐                     │
     no_retrieval         single / rag_fusion ◄────┘
          │                    │
       LLM only            is_rel() ──► corrective_isrel() if empty
                               │
                        generate_response()
                               │
                          sup_check() ──► corrective_issup() if not grounded
                               │
                         Final Answer
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Groq — `openai/gpt-oss-120b` |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (384-dim) |
| Vector Store | Pinecone (index: `diabetes`, metric: cosine) |
| Framework | LangChain + LangChain-Community |
| PDF Loader | PyPDFLoader (`pypdf`) |
| Text Splitter | RecursiveCharacterTextSplitter (1000 chars, 200 overlap) |
| Web Search | Tavily (available for future CRAG extension) |
| Runtime | Python 3.10, managed with `uv` |
