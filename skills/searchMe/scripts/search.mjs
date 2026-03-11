#!/usr/bin/env node

import { fileURLToPath } from 'node:url';
import path from 'node:path';
import dotenv from 'dotenv';
import pg from 'pg';
import { createGateway, embed } from 'ai';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.join(__dirname, '.env') });

const EMBEDDING_MODEL = 'google/gemini-embedding-001';

const API_KEY = process.env.AI_GATEWAY_API_KEY;
const DATABASE_URL = process.env.DATABASE_URL;

if (!API_KEY || !DATABASE_URL) {
  console.error('Error: AI_GATEWAY_API_KEY and DATABASE_URL must be set in .env');
  process.exit(1);
}

const { Client } = pg;
const gateway = createGateway({ apiKey: API_KEY });
const db = new Client({ connectionString: DATABASE_URL });

function normalizeVector(values) {
  return `[${values.join(',')}]`;
}

function formatSnippet(text) {
  const trimmed = String(text ?? '').replace(/\s+/g, ' ').trim();
  if (!trimmed) return '';
  return trimmed.length > 240 ? `${trimmed.slice(0, 237)}...` : trimmed;
}

async function embedText(text) {
  const { embedding } = await embed({
    model: gateway.embeddingModel(EMBEDDING_MODEL),
    value: text,
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
    [vectorLiteral, limit],
  );

  return result.rows;
}

async function main() {
  const query = process.argv.slice(2).map((v) => v.trim()).filter(Boolean).join(' ');
  if (!query) {
    console.error('Usage: node search.mjs <query>');
    process.exit(1);
  }

  await db.connect();

  try {
    const rows = await search(query);
    if (rows.length === 0) {
      console.log('No results found.');
      return;
    }

    console.log(JSON.stringify(rows.map((row, i) => ({
      rank: i + 1,
      source: row.source ?? row.basename ?? 'unknown',
      title: row.title ?? null,
      similarity: Number(row.similarity ?? 0).toFixed(3),
      snippet: formatSnippet(row.content),
    })), null, 2));
  } finally {
    await db.end();
  }
}

main().catch(async (error) => {
  console.error(error.message ?? error);
  try { await db.end(); } catch { /* ignore */ }
  process.exit(1);
});
