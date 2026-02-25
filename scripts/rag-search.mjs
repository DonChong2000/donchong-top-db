import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

import dotenv from 'dotenv';
import pg from 'pg';
import { createGateway, embed } from 'ai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectDir = path.resolve(__dirname, '..');

dotenv.config({ path: path.join(projectDir, '.env') });

const API_KEY = process.env.AI_GATEWAY_API_KEY;
const DATABASE_URL = process.env.DATABASE_URL;
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

function normalizeVector(values) {
  return `[${values.join(',')}]`;
}

function parseQuery(argv) {
  const parts = argv.slice(2).map((value) => value.trim()).filter(Boolean);
  if (parts.length === 0) return null;
  return parts.join(' ');
}

function formatSnippet(text) {
  const trimmed = String(text ?? '').replace(/\s+/g, ' ').trim();
  if (!trimmed) return '';
  return trimmed.length > 240 ? `${trimmed.slice(0, 237)}...` : trimmed;
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

async function search(query, limit = 5) {
  const embedding = await embedText(query);
  const vectorLiteral = normalizeVector(embedding);

  const result = await db.query(
    `SELECT
       content,
       metadata->>'source' AS source,
       metadata->>'basename' AS basename,
       metadata->>'title' AS title,
       1 - (embedding <=> $1::vector) AS similarity
     FROM document_chunks
     ORDER BY embedding <=> $1::vector
     LIMIT $2`,
    [vectorLiteral, limit]
  );

  return result.rows;
}

async function main() {
  const query = parseQuery(process.argv);
  if (!query) {
    console.error('Provide a query string. Example: npm run search -- "string"');
    process.exit(1);
  }

  await db.connect();

  try {
    const rows = await search(query);
    if (rows.length === 0) {
      console.log('No results found.');
      return;
    }

    rows.forEach((row, index) => {
      const rank = index + 1;
      const similarity = Number(row.similarity ?? 0).toFixed(3);
      const source = row.source ?? row.basename ?? 'unknown';
      const title = row.title ? ` | ${row.title}` : '';
      const snippet = formatSnippet(row.content);

      console.log(`${rank}. ${source}${title} (score: ${similarity})`);
      if (snippet) {
        console.log(`   ${snippet}`);
      }
    });
  } finally {
    await db.end();
  }
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
