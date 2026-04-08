# Legal Contract RAG — MCP Server

A retrieval-augmented generation (RAG) system for analyzing legal contracts, exposed as an [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server. It lets an LLM-powered assistant answer natural-language questions about a corpus of contracts, compare provisions across agreements, and locate specific clause types — all grounded in the actual contract text rather than parametric knowledge alone.

The project uses the [CUAD (Contract Understanding Atticus Dataset)](https://www.atticusprojectai.org/cuad) as its contract corpus, ChromaDB for vector storage and retrieval, and Ollama with the Qwen3 32B model for generation.

## How It Works

Contracts are loaded from the CUAD dataset, split into overlapping token-level chunks, and indexed into a ChromaDB collection along with metadata (contract title, party names). At query time, the system retrieves the most relevant chunks for a given question, assembles them into a prompt, and passes them to an LLM to generate a grounded answer.

The RAG pipeline supports multi-turn conversations through a sliding-window history mechanism: follow-up questions are rewritten into self-contained queries using the conversation context, and previously retrieved clauses are carried forward so the model can reference them across turns.

The whole pipeline is wrapped in an MCP server so that any MCP-compatible client (such as Claude Desktop) can call it as a set of tools.

## MCP Tools

The server exposes five tools:

**`ask_contracts`** — Ask a question across all contracts in the database. Returns an answer synthesized from the most relevant clauses found anywhere in the corpus. Supports multi-turn conversation via a `session_id`.

**`ask_contract`** — Ask a question about a specific contract by title. Restricts retrieval to that single agreement. Also supports multi-turn conversation.

**`compare_contracts`** — Compare two or more contracts on a given topic. Each contract is queried independently so results are balanced, and the LLM produces a side-by-side comparative analysis.

**`find_contract_clauses`** — Search for a specific clause type (e.g., "termination", "governing law", "non-compete") across all contracts or within a single contract. Returns the raw excerpts and their source contract titles.

**`list_contracts`** — List all contracts in the database, optionally filtered by party name. Useful for discovering available contract titles before calling the other tools.

## Project Structure

```
rag/
├── chunker.py          # Loads CUAD contracts, chunks them, indexes into ChromaDB
├── rag_core.py         # Core RAG logic: query, prompt rewriting, answer generation
├── server.py           # MCP server exposing the five tools above
└── chroma_data/        # ChromaDB persistent storage (generated)

testing/
├── query_testing.py        # Measures retrieval accuracy (conceptual vs. factual)
├── generation_testing.py   # Scores generated answers against reference answers
├── rag_smoke_test.py       # Smoke tests for rag_core functions
└── integration_testing.py  # Integration tests for the MCP server tools
```

## Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.com/)** with the `qwen3:32b` model pulled (`ollama pull qwen3:32b`)
- **[uv](https://docs.astral.sh/uv/)** (used to run the MCP server)
- The **CUAD dataset** — download `CUADv1.json` and `train_separate_questions.json` and place them in `cuad/data/`

## Setup

1. **Install dependencies:**

   ```bash
   pip install chromadb ollama langchain-text-splitters tiktoken mcp
   ```

2. **Pull the model:**

   ```bash
   ollama pull qwen3:32b
   ```

3. **Build the vector database:**

   ```bash
   python rag/chunker.py
   ```

   This loads all contracts from CUAD, chunks them into ~256-token segments with 80-token overlap, and indexes them into a persistent ChromaDB collection. Only needs to be run once.

4. **Run the MCP server:**

   ```bash
   uv run python3 rag/server.py
   ```

   The server communicates over stdio and is designed to be connected to an MCP client such as Claude Desktop.

## Connecting to Claude Desktop

Add the server to your Claude Desktop MCP configuration:

```json
{
  "mcpServers": {
    "contracts": {
      "command": "uv",
      "args": ["run", "python3", "path/to/rag/server.py"]
    }
  }
}
```

Once connected, you can ask Claude questions like:

- *"What contracts involve Acme Corp?"*
- *"What is the governing law in the Birch Communications contract?"*
- *"How do the termination clauses differ between these two agreements?"*
- *"Find all non-compete clauses across the database."*

## Testing

The project includes four levels of testing:

**Retrieval accuracy** (`query_testing.py`) — Samples question/answer pairs from the CUAD training set and measures how often the correct answer appears in the top-k retrieved chunks. Reports accuracy separately for conceptual questions (e.g., "Is there a non-compete clause?") and factual questions (e.g., "What is the effective date?").

```bash
python testing/query_testing.py
```

**Generation quality** (`generation_testing.py`) — Generates answers for sampled questions using the full RAG pipeline, then uses an LLM judge to score each answer against the reference answer on a 1–5 scale for correctness and completeness. Results are saved as timestamped JSON files for comparison across runs.

```bash
python testing/generation_testing.py
```

**Smoke tests** (`rag_smoke_test.py`) — Quick functional tests for the core RAG functions (querying, listing, answer generation, comparison). Can be run in `--db-only` mode to skip tests that require Ollama.

```bash
python testing/rag_smoke_test.py           # all tests
python testing/rag_smoke_test.py --db-only  # ChromaDB tests only
```

**Integration tests** (`integration_testing.py`) — Spins up the MCP server and exercises each tool through the MCP client protocol, including happy-path and error-handling scenarios. Also supports `--db-only`.

```bash
python testing/integration_testing.py           # all tests
python testing/integration_testing.py --db-only  # DB tests only
```

## Design Decisions

- **Chunk size of 256 tokens with 80-token overlap** — chosen to balance retrieval precision (smaller chunks are easier to match) against having enough context for the LLM to produce a useful answer. The overlap helps avoid splitting important clauses across chunk boundaries.
- **Sliding window of 10 conversation turns** — keeps multi-turn conversations functional without exhausting the model's context window. Older turns are dropped as new ones come in.
- **Prompt rewriting for follow-ups** — follow-up questions like "What about the duration?" are rewritten into self-contained queries using the conversation history, which significantly improves retrieval quality on subsequent turns.
- **Separate querying per contract for comparisons** — ensures balanced representation across contracts rather than letting one contract dominate the retrieved results.
- **ChromaDB for vector storage** — provides persistence and semantic search without the operational overhead of a full client-server database.