# ContractLens

> AI-powered contract intelligence platform for document analysis, semantic search, RAG-based Q&A, and citation-grounded insights.

🚀 **Live Demo:** https://contractlens-snowy.vercel.app/


# ContractLens

ContractLens is an intelligent contract analysis project. Its current MVP provides document upload and extraction, chunking, open-source embeddings, PostgreSQL/pgvector persistence, document-scoped semantic search, grounded question answering, and structured contract analysis with source citations for PDF, DOCX, and TXT contracts.

## Project structure

```text
app/
  api/routes/        HTTP endpoints
  core/              Settings and logging
  schemas/           Request/response models
  services/          Extraction, document store, chunking, embeddings, retrieval, LLM, RAG
scripts/
  init_db.py         PostgreSQL/pgvector schema initializer
  backfill_documents.py   Legacy upload metadata backfill
tests/               Chunking, retrieval, and upload API tests
evaluation/          Synthetic contracts, offline service harness, and metrics
docker-compose.yml   Local pgvector database service
data/uploads/        Uploaded source files (created at runtime)
```

The upload response includes text segments with page numbers when available (PDF), and a document label for formats without reliable page boundaries. Chunk IDs and source locators are retained through the embedding and pgvector pipeline. Stages 3 and 4 retrieve only the requested document's chunks, ask a configured LLM to answer or extract facts from that context, and map provider-returned source labels back to retrieved chunk metadata.

## Run locally

Python 3.10+ is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
docker compose up -d db
python -m scripts.init_db
uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs` for the interactive API documentation. Health check: `GET /health`. Upload a document with `POST /api/documents` as multipart form field `file`.

## Configuration

See `.env.example`. Change the placeholder `POSTGRES_PASSWORD` and matching password in `DATABASE_URL`. The Compose service uses the pgvector PostgreSQL 17 image, defaults to host port 5433 to avoid a common local PostgreSQL conflict, and persists data in a named volume. `python -m scripts.init_db` creates the extension, document metadata/text table, chunk table, and cosine HNSW index without dropping existing tables. For legacy uploads still present in `UPLOAD_DIR`, run `python -m scripts.backfill_documents` after schema initialization to create their metadata and extracted-text rows; this leaves vector rows untouched.

`MAX_UPLOAD_SIZE_MB` limits uploads; `UPLOAD_DIR` stores original files; `EMBEDDING_MODEL` selects a FastEmbed supported model; `EMBEDDING_DIMENSIONS` must match that model's output dimension and the schema; `CHUNK_SIZE_CHARS` / `CHUNK_OVERLAP_CHARS` control chunking (overlap must be less than half the chunk size); and `RETRIEVAL_TOP_K` sets the default search result count. The default `BAAI/bge-small-en-v1.5` produces 384-dimensional vectors and its model files are downloaded on first processing if not cached. If you change the model dimension after creating the table, migrate/recreate the `document_chunks.embedding` column to match before processing documents.

For question answering and analysis, set `LLM_API_KEY`; `LLM_BASE_URL` defaults to OpenAI's API and can point to an OpenAI-compatible endpoint; `LLM_MODEL` chooses the provider model; and `LLM_TIMEOUT_SECONDS` bounds request time. `RAG_MIN_SIMILARITY` controls which retrieved chunks are considered relevant. `ANALYSIS_TOP_K` sets the number of candidates retrieved for each analysis topic. The API key is read from the environment and never returned in API responses or logs. Retrieved contract context is sent to the configured LLM endpoint. The provider uses the Chat Completions-compatible HTTP interface without adding an LLM framework dependency. See the [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create) for the default endpoint contract.

## API behavior

- Accepted extensions: `.pdf`, `.docx`, `.txt`.
- Rejects empty files, unsupported formats, oversized uploads, and documents with no extractable text.
- Stores accepted original files under a generated ID, never the user-supplied filename.
- Returns extracted text, page/section segments, document ID, original filename, and byte size.
- `POST /api/documents/{document_id}/process` re-reads the stored upload, chunks each extracted source segment, embeds every chunk, and replaces that document's stored chunks.
- `POST /api/documents/{document_id}/search` accepts `{"query":"...", "top_k":5}` and returns semantically nearest chunks, similarity scores, and source/page locators. Search is scoped to the requested document.
- `POST /api/documents/{document_id}/ask` accepts `{"question":"What is the termination notice period?"}` and returns a grounded answer plus citations with chunk ID, source locator, page number, and similarity. It requires an uploaded and processed document. If retrieval does not find sufficiently relevant context, it returns a fixed `not enough information` answer without calling the LLM.
- `POST /api/documents/{document_id}/analyze` accepts an empty JSON object and returns structured findings for contract type, parties, dates, renewal and notice terms, payment, termination, confidentiality, intellectual property, non-compete/non-solicitation, governing law, disputes, obligations, deadlines, other clauses, and attention items.
- Extraction errors are logged server-side and returned as safe HTTP errors.

Example analysis request:

```http
POST /api/documents/00000000-0000-0000-0000-000000000000/analyze
Content-Type: application/json
{}
```

Example response (abbreviated):

```json
{
  "document_id": "00000000-0000-0000-0000-000000000000",
  "status": "complete",
  "contract_type": {
    "value": "Professional Services Agreement",
    "basis": "explicit",
    "sources": [
      {
        "chunk_id": "retrieved-chunk-id",
        "source_locator": "Page 1",
        "page_number": 1,
        "similarity": 0.82
      }
    ]
  },
  "parties": [],
  "effective_date": null,
  "expiration_date": null,
  "renewal_terms": [],
  "notice_periods": [],
  "payment_terms": [],
  "termination_conditions": [],
  "confidentiality_obligations": [],
  "intellectual_property_provisions": [],
  "non_compete_non_solicitation": [],
  "governing_law": null,
  "dispute_resolution": [],
  "major_obligations": [],
  "important_deadlines": [],
  "other_important_clauses": [],
  "attention_items": []
}
```

Each populated field is a finding with `value`, `basis` (`explicit` or `inference`), and one or more `sources`. Source labels emitted by the provider are mapped to chunks retrieved for this document; findings with missing or invalid labels are omitted. `status` is `insufficient_context` when relevant evidence or citable findings are unavailable. Analysis facts are informational summaries, not legal advice or legal conclusions.

The RAG system prompt treats contract text as untrusted data, rejects instructions embedded in the document, prohibits outside knowledge, and requests a structured answer with source labels. Returned source labels are mapped only to chunks retrieved for the requested document; invalid labels are discarded, and an answer with no valid citations is replaced with the safe not-found response. An LLM response cannot be guaranteed factually correct, so the source excerpts should be reviewed against the contract.

PDF pages are one-indexed. DOCX and TXT extraction returns a single `Document` segment because those formats do not provide stable page boundaries in this implementation. This is text extraction only; it does not claim legal interpretation or analysis.

## Tests

Run `pytest`. The suite checks source-aware chunking and overlap, retrieval isolation, grounded answers, structured contract analysis, citation mapping, error cases, and the existing health/docs/TXT upload path. RAG and analysis tests use deterministic mock providers and do not need an external LLM key. A PostgreSQL instance is required for live document processing and retrieval; unit tests use fake stores so they run without a database.

## Evaluation

Run the repeatable synthetic evaluation with:

```powershell
python -m scripts.evaluate
```

The small `evaluation/data/synthetic_contracts.json` dataset contains two fictional contracts, explicit expected facts, retrieval cases, answerable RAG questions, and deliberately missing information. The runner calls the existing `retrieve_chunks`, `answer_question`, and `analyze_contract` services using an in-memory document-filtered vector store and deterministic signed lexical feature vectors. It reports Recall@K, mean reciprocal rank, document isolation, expected-answer pass rate, unsupported-answer count, citation validity, analysis fact/citation pass rate, missing-field handling, and total passed/failed evaluation assertions. Citation checks require the chunk to have been retrieved for the requested document and verify its locator, page, and similarity metadata.

The RAG and analysis providers are deterministic mocks. The analysis mock is seeded from the dataset's gold facts, so the reported analysis pass rate measures retrieval, schema validation, citation mapping, and missing-field handling; it does not measure an external LLM's extraction accuracy. The offline lexical vectors are a repeatable retrieval proxy, not BGE or live pgvector performance. Use production embeddings and human-reviewed expected answers for a meaningful evaluation of live model quality. The separate test suite also checks that invented/foreign citations are rejected and unsupported missing-field claims are suppressed.

## Frontend

The Next.js App Router frontend lives in `frontend/` and uses TypeScript, Tailwind CSS, and a small typed API client. It provides an upload workspace, extracted-text/source view, document processing status, structured analysis, and contract Q&A with citation metadata. Document history is loaded from PostgreSQL through `GET /api/documents`. Selecting an existing document loads durable metadata and extracted text using `GET /api/documents/{document_id}` and `/text`; browser storage is not the source of truth.

Start the backend from the project root after configuring the database, embedding model, and LLM as described above:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
cd frontend
Copy-Item .env.example .env.local
npm install
npm run dev
```

Open `http://localhost:3000`. `NEXT_PUBLIC_API_URL` in `frontend/.env.local` sets the FastAPI base URL; `NEXT_PUBLIC_MAX_UPLOAD_SIZE_MB` optionally adjusts the client-side size check. The backend CORS allowlist is `CORS_ORIGINS` (comma-separated exact origins). No backend secrets are sent to the browser. Frontend checks: `npm run typecheck`, `npm test`, and `npm run build` from `frontend/`.

## Production readiness and deployment

Architecture: `User → Next.js (Vercel) → FastAPI (Render) → Managed PostgreSQL 17 + pgvector`, with FastAPI calling the configured LLM provider and generating BGE embeddings. The Render Blueprint is `render.yaml`; it provisions a private managed PostgreSQL database, initializes `vector` and both tables through `scripts.init_db`, runs the API with Uvicorn, and configures `/health` as its liveness endpoint. The database extension is initialized with `CREATE EXTENSION IF NOT EXISTS vector`; schema setup is repeatable and does not drop or reset existing data. Render Blueprint requires a connected Git repository. Set `CORS_ORIGINS` to the exact deployed Vercel origin and set `LLM_API_KEY` in Render's secret environment settings. The Blueprint uses paid Render compute/database plans; review provider pricing before creating resources.

Create the Vercel project from the same repository and set its root directory to `frontend/`. Set `NEXT_PUBLIC_API_URL` to the deployed Render API origin and `NEXT_PUBLIC_MAX_UPLOAD_SIZE_MB` consistently with the backend. Set these variables in Vercel before building. Configure the backend's `CORS_ORIGINS` to the Vercel production origin. Backend production settings require `DATABASE_URL`, `LLM_API_KEY`, explicit HTTPS CORS origins, and `ENVIRONMENT=production`.

Production uploads and extracted contract text are stored in PostgreSQL; the upload directory is only a compatibility copy and is not used to retrieve documents. Lifecycle state (`uploaded`, `processing`, `processed`, or `failed`), chunk counts, and embedding model metadata are durable. `GET /health` is a liveness check; `GET /ready` checks database connectivity.

This MVP does not implement user authentication or per-user document authorization. Document history is a shared workspace, so the configured API must not be exposed for confidential contract use until authentication, ownership checks, and appropriate retention controls are added. Public deployment is suitable only for a synthetic portfolio demo in its current form.

Important endpoints: `POST /api/documents` upload; `GET /api/documents` history; `GET /api/documents/{id}` metadata/status; `GET /api/documents/{id}/text` extracted source text; `POST /api/documents/{id}/process`; `POST /api/documents/{id}/ask`; and `POST /api/documents/{id}/analyze`. `/docs` exposes the OpenAPI documentation.

## Dependencies

FastAPI, Uvicorn, Pydantic Settings, python-multipart, pypdf, python-docx, FastEmbed, psycopg 3, pgvector, and HTTPX power the backend. The frontend dependencies are managed in `frontend/package.json`. PostgreSQL and the pgvector extension are provided for local development by Docker Compose. PDF page locators are available for text-based PDFs; DOCX and TXT page numbers are unavailable, and scanned PDFs require OCR. Retrieval can miss a relevant provision, so missing/empty analysis fields should not be treated as proof that a clause is absent from the full contract. The similarity threshold can reject useful context when configured too high; tune it for your documents and embedding model. Attention items are extracted provisions, not a numerical risk score or legal judgment.
