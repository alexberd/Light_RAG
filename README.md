# Light RAG (Qdrant)

Small retrieval pipeline: chunk UTF-8 text, embed with a local sentence-transformer model (default **BAAI/bge-m3**, 1024 dimensions, multilingual) or OpenAI, store vectors and text in **Qdrant**, and query by semantic similarity.

## Requirements

- **Python 3.12+**
- **Qdrant** reachable over HTTP (default `http://localhost:6333`). Run Qdrant however you prefer (binary, Docker, cloud). Install from https://github.com/qdrant/qdrant/releases/download/v1.17.0/qdrant-x86_64-pc-windows-msvc.zip

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

Copy `.env.example` to `.env` and adjust if needed.

## Configuration

Environment variables are documented in `.env.example`. Important defaults:

| Variable | Default | Purpose |
|----------|---------|---------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant REST URL |
| `QDRANT_COLLECTION` | `chunks` | Collection name |
| `EMBEDDING_PROVIDER` | `local` | `local` (Hugging Face model) or `openai` |
| `LOCAL_EMBEDDING_MODEL` | `BAAI/bge-m3` | Local embedding model (see below) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `512` / `50` | Token-aware chunking |
| `RETRIEVE_LIMIT` | `3` | Top chunks returned per query |

With `EMBEDDING_PROVIDER=local`, the first run downloads the model into the Hugging Face cache (on Windows you may see a symlink cache warning; it is safe to ignore or set `HF_HUB_DISABLE_SYMLINKS_WARNING=1`).

### English vs multilingual

- **`BAAI/bge-m3`** — **default**: **multilingual** dense embeddings (e.g. Greek alongside other languages), **1024** dimensions. The app applies the recommended query vs document prefixes for BGE-M3 on search vs ingest.
- **`BAAI/bge-large-en-v1.5`** — strong for **English**-only text; set `LOCAL_EMBEDDING_MODEL` accordingly. Same vector size as BGE-M3.

Re-ingest documents after switching models; do not mix two different embedders in one collection.

## Usage

**Ingest** a UTF-8 text file. The document id defaults to the **filename without extension** (override with `--doc-id`):

```bash
python -m rag ingest path\to\document.txt
```

**Query** (semantic search over stored chunks):

```bash
python -m rag query "your search phrase"
```

**Delete Qdrant Database** 
```bash
curl -X DELETE "http://localhost:6333/collections/chunks"
```

Optional: limit results to one ingested document:

```bash
python -m rag query "your phrase" --doc-id mydoc
```

The CLI prints each hit with a similarity **score** and the **chunk text** stored in Qdrant (payload includes `doc_id`, `chunk_index`, and `content`).

## OpenAI embeddings

Set `EMBEDDING_PROVIDER=openai`, provide `OPENAI_API_KEY`, and optionally `OPENAI_EMBEDDING_MODEL` (default `text-embedding-3-large`). The project expects **1024-dimensional** vectors to match the Qdrant collection configuration.
