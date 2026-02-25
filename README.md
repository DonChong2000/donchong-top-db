# donchong-top-db

PostgreSQL + pgvector setup for RAG with Gemini chunking + embeddings.

## Included

- `docker-compose.yml` with `pgvector/pgvector:pg16`
- Schema bootstrap in `init/001_schema.sql`
- `data/` folder for raw markdown files to ingest
- Node.js ingestion script at `scripts/rag-ingest.mjs`

## 1) Start database

```bash
docker compose up -d
```

## 2) Configure env

Copy `.env.example` to `.env` and fill values:

```bash
cp .env.example .env
```

Required variables:

- `AI_GATEWAY_API_KEY`
- `DATABASE_URL`

## 3) Install dependencies

```bash
pnpm install
```

## 4) Add markdown files

Put source markdown files inside:

```bash
./data
```

Example:

```bash
mkdir -p data/notes
cp ../some-file.md data/notes/some-file.md
```

## 5) Ingest into RAG DB

```bash
pnpm run ingest
```

What ingestion does:

- Reads all `data/**/*.md`
- Uses **Gemini Flash latest** (`gemini-2.5-flash`) to chunk content using prompt-based chunking guidelines
- Uses **Gemini embedding model** (`gemini-embedding-001`) for vectors
- Skips files whose content hash matches the last ingested version
- Saves:
  - `documents.source` = source file path
  - `document_chunks.metadata.source` and `document_chunks.metadata.basename`
  - `document_chunks.embedding` as `vector(3072)`

## Delete all entries from one file

Command format:

```bash
node scripts/rag-ingest.mjs "Delete <filename>"
```

Examples:

```bash
node scripts/rag-ingest.mjs "Delete some-file.md"
node scripts/rag-ingest.mjs "Delete data/notes/some-file.md"
```

The script deletes from `documents`; related `document_chunks` are deleted via `ON DELETE CASCADE`.

## Optional model overrides

You can override defaults in `.env`:

- `CHUNK_MODEL` (default: `gemini-2.5-flash`)
- `EMBEDDING_MODEL` (default: `gemini-embedding-001`)
