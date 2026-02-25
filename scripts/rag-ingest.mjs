import { readFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { createHash, randomUUID } from 'node:crypto';
import { fileURLToPath } from 'node:url';

import dotenv from 'dotenv';
import { glob } from 'glob';
import pg from 'pg';
import { createGateway, embed, generateText, jsonSchema, Output } from 'ai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectDir = path.resolve(__dirname, '..');

dotenv.config({ path: path.join(projectDir, '.env') });

const API_KEY = process.env.AI_GATEWAY_API_KEY;
const DATABASE_URL = process.env.DATABASE_URL;
const CHUNK_MODEL = process.env.CHUNK_MODEL ?? 'google/gemini-3-flash';
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL ?? 'google/gemini-embedding-001';

if (!API_KEY) {
  throw new Error('Missing AI_GATEWAY_API_KEY. Add it to donchong-top-db/.env');
}

if (!DATABASE_URL) {
  throw new Error('Missing DATABASE_URL. Add it to donchong-top-db/.env');
}

const { Client } = pg;
const gateway = createGateway({ apiKey: API_KEY });
const db = new Client({ connectionString: DATABASE_URL });

const chunkSchema = jsonSchema({
  type: 'object',
  properties: {
    chunks: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          chunk_index: { type: 'integer' },
          title: { type: 'string' },
          content: { type: 'string' }
        },
        required: ['content']
      }
    }
  },
  required: ['chunks']
});

function normalizeVector(values) {
  return `[${values.join(',')}]`;
}

function hashMarkdown(markdown) {
  return createHash('sha256').update(markdown).digest('hex');
}

function parseDeleteCommand(rawArg) {
  if (!rawArg) return null;
  const trimmed = rawArg.trim();
if (trimmed.toLowerCase().startsWith('delete ')) {
    return trimmed.slice(7).trim();
  }
  return null;
}

async function chunkMarkdown(markdown, sourceFile) {
  const prompt = `You are an expert RAG chunking assistant.
Chunk the markdown into semantically coherent chunks for retrieval.

Chunking guidelines:
1. Keep each chunk focused on one idea or one subsection.
2. Prefer chunks roughly 300-900 tokens; split long sections.
3. Preserve important headings and list context in chunk text.
4. Keep code blocks intact when possible.
5. Avoid tiny fragments unless they contain key facts.
6. Complete or supplement the context if needed.
7. Return valid JSON only.

Required output schema:
{
  "chunks": [
    {
      "chunk_index": 0,
      "title": "optional short heading",
      "content": "chunk text"
    }
  ]
}

Source filename: ${sourceFile}
Markdown:
"""
${markdown}
"""`;

  const { output: parsed } = await generateText({
    model: gateway(CHUNK_MODEL),
    prompt,
    output: Output.object({ schema: chunkSchema }),
    temperature: 0.1
  });

  if (!parsed?.chunks || !Array.isArray(parsed.chunks) || parsed.chunks.length === 0) {
    throw new Error(`No chunks returned by model for ${sourceFile}`);
  }

  return parsed.chunks
    .map((chunk, index) => ({
      chunk_index: Number.isInteger(chunk.chunk_index) ? chunk.chunk_index : index,
      title: typeof chunk.title === 'string' ? chunk.title : null,
      content: String(chunk.content ?? '').trim()
    }))
    .filter((chunk) => chunk.content.length > 0)
    .sort((a, b) => a.chunk_index - b.chunk_index);
}

async function embedText(text) {
  const { embedding } = await embed({
    model: gateway.embeddingModel(EMBEDDING_MODEL),
    value: text
  });

  if (!embedding || embedding.length === 0) {
    throw new Error('Embedding API returned empty vector');
  }

  return embedding;
}

async function deleteByFilename(inputName) {
  const result = await db.query(
    `DELETE FROM documents
     WHERE source = $1 OR source LIKE ('%/' || $1)`,
    [inputName]
  );

  console.log(`Deleted ${result.rowCount} document record(s) for "${inputName}".`);
}

async function ingestOneFile(filePath) {
  const markdown = await readFile(filePath, 'utf8');
  const relativeSourcePath = path.relative(projectDir, filePath).replaceAll('\\', '/');
  const basename = path.basename(filePath);
  const contentHash = hashMarkdown(markdown);

  const existing = await db.query(
    'SELECT metadata FROM documents WHERE source = $1 LIMIT 1',
    [relativeSourcePath]
  );
  const existingHash = existing.rows[0]?.metadata?.hash ?? null;

  if (existingHash === contentHash) {
    console.log(`Skipping ${relativeSourcePath} (unchanged).`);
    return;
  }

  console.log(`Chunking ${relativeSourcePath}...`);
  const chunks = await chunkMarkdown(markdown, relativeSourcePath);

  await db.query('BEGIN');
  try {
    await db.query('DELETE FROM documents WHERE source = $1', [relativeSourcePath]);

    const documentId = randomUUID();

    await db.query(
      `INSERT INTO documents (id, source, title, metadata)
       VALUES ($1, $2, $3, $4::jsonb)`,
      [
        documentId,
        relativeSourcePath,
        basename,
        JSON.stringify({ basename, hash: contentHash })
      ]
    );

    for (const chunk of chunks) {
      const embedding = await embedText(chunk.content);

      await db.query(
        `INSERT INTO document_chunks (
          id,
          document_id,
          chunk_index,
          content,
          token_count,
          metadata,
          embedding_model,
          chunk_method,
          embedding
        ) VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9::vector)`,
        [
          randomUUID(),
          documentId,
          chunk.chunk_index,
          chunk.content,
          null,
          JSON.stringify({
            title: chunk.title,
            source: relativeSourcePath,
            basename
          }),
          EMBEDDING_MODEL,
          CHUNK_MODEL,
          normalizeVector(embedding)
        ]
      );
    }

    await db.query('COMMIT');
    console.log(`Saved ${chunks.length} chunks for ${relativeSourcePath}`);
  } catch (error) {
    await db.query('ROLLBACK');
    throw error;
  }
}

async function main() {
  await db.connect();

  const firstArg = process.argv[2];
  const deleteTarget = parseDeleteCommand(firstArg);

  if (deleteTarget) {
    await deleteByFilename(deleteTarget);
    await db.end();
    return;
  }

  const markdownPattern = path.join(projectDir, 'data/**/*.md').replaceAll('\\', '/');
  const markdownFiles = await glob(markdownPattern);
  if (markdownFiles.length === 0) {
    console.log('No markdown files found in donchong-top-db/data. Nothing to ingest.');
    await db.end();
    return;
  }

  for (const filePath of markdownFiles.sort()) {
    await ingestOneFile(filePath);
  }

  await db.end();
  console.log('Ingestion complete.');
}

main().catch(async (error) => {
  console.error(error);
  try {
    await db.end();
  } catch {
    // ignore
  }
  process.exit(1);
});
