---
name: searchMe
description: RAG similarity search against the document_chunks database. Use when the user wants to search documents, find related content, or query the knowledge base.
disable-model-invocation: true
allowed-tools: Bash
argument-hint: [search query]
---

Run a RAG similarity search using the bundled script:

```bash
node "${CLAUDE_SKILL_DIR}/scripts/search.mjs" $ARGUMENTS
```

Present the results in a clear, readable format. If no results are found, let the user know.
If the script fails due to missing dependencies, run `npm install` inside `${CLAUDE_SKILL_DIR}/scripts/` first, then retry.
