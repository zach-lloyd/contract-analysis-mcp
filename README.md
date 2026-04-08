# Legal Contract RAG — MCP Server

A retrieval-augmented generation (RAG) system for analyzing legal contracts, exposed as an [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server. It lets an LLM-powered assistant answer natural-language questions about a corpus of contracts, compare provisions across agreements, and locate specific clause types.

The project uses the [Contract Understanding Atticus Dataset (CUAD)](https://www.atticusprojectai.org/cuad) as its contract corpus, ChromaDB for vector storage and retrieval, and Ollama with the Qwen3 32B model for generation.

## How It Works

Contracts are loaded from the CUAD dataset, split into overlapping token-level chunks, and indexed into a ChromaDB collection. The title of each contract and the names of the parties are also stored in the collection as metadata. At query time, the system retrieves the most relevant chunks for a given question, assembles them into a prompt, and passes them to an LLM to generate a grounded answer.

The RAG pipeline supports multi-turn conversations through a sliding-window history mechanism: follow-up questions are rewritten into self-contained queries using the conversation context, and previously retrieved clauses are carried forward so the model can reference them across turns.

The whole pipeline is wrapped in an MCP server so that any MCP-compatible client (such as Claude Desktop) can call it as a set of tools.

## Brief Demo

https://github.com/user-attachments/assets/ffd40fdd-331d-45e2-a98b-8bfb44ce2972

## MCP Tools

The server exposes five tools:

**`ask_contracts`** — Ask a question across all contracts in the database. Returns an answer synthesized from the most relevant clauses found anywhere in the corpus. Supports multi-turn conversation via a `session_id`.

**`ask_contract`** — Ask a question about a specific contract by title. Restricts retrieval to that single agreement. Also supports multi-turn conversation via a `session_id`.

**`compare_contracts`** — Compare two or more contracts on a given topic. Each contract is queried independently so results are balanced, and the LLM produces a side-by-side comparative analysis.

**`find_contract_clauses`** — Search for a specific clause type (e.g., "termination", "governing law", "non-compete") across all contracts or within a single contract. Returns the raw excerpts and their source contract titles.

**`list_contracts`** — List all contracts in the database, optionally filtered by party name. Useful for discovering available contract titles before calling the other tools.

## Project Structure

```
.
├── src/
│  ├── rag/
│  │   ├── chunker.py          # Loads CUAD contracts, chunks them, indexes into ChromaDB
│  │   ├── rag_core.py         # Core RAG logic: query, prompt rewriting, answer generation
│  │   ├── server.py           # MCP server exposing the five tools above
│  │   └── chroma_data/        # ChromaDB persistent storage (generated)
│  └── testing/
│      ├── query_testing.py        # Measures retrieval accuracy (conceptual vs. factual)
│      ├── generation_testing.py   # Scores generated answers against reference answers
│      ├── rag_smoke_test.py       # Smoke tests for rag_core functions
│      └── integration_testing.py  # Integration tests for the MCP server tools
└── README.md
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
   python3 rag/chunker.py
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
      "command": "uv", # You may need to substitute your absolute uv path here
      "args": ["run", "--directory", "python3", "path/to/rag/server.py"]
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

The project includes five levels of testing:

**RAG Smoke tests** (`rag_smoke_test.py`) — Quick functional tests for the core RAG functions (querying, listing, answer generation, comparison). Can be run in `--db-only` mode to skip tests that require Ollama.

```bash
python testing/rag_smoke_test.py           # all tests
python testing/rag_smoke_test.py --db-only  # ChromaDB tests only
```

**Retrieval accuracy** (`query_testing.py`) — Samples question/answer pairs from the CUAD training set and measures how often the correct answer appears in the top-k retrieved chunks. Reports accuracy separately for conceptual questions (e.g., "Is there a non-compete clause?") and factual questions (e.g., "What is the effective date?").

```bash
python testing/query_testing.py
```

**Generation quality** (`generation_testing.py`) — Generates answers for sampled questions using the full RAG pipeline, then uses an LLM judge to score each answer against the reference answer on a 1–5 scale for correctness and completeness, with the LLM judge also providing a one-sentence justification for its answer. Results are saved as timestamped JSON files for comparison across runs.

```bash
python testing/generation_testing.py
```

**Integration tests** (`integration_testing.py`) — Spins up the MCP server and exercises each tool through the MCP client protocol, including happy-path and error-handling scenarios. Also supports `--db-only`.

```bash
python testing/integration_testing.py           # all tests
python testing/integration_testing.py --db-only  # DB tests only
```

**Manual end-to-end tests** - In addition to the above automated tests, I also performed extensive manual testing of the MCP after connecting it to Claude. After activating the MCP, I asked Claude the following question sequences:

*Sequence 1: Deep Follow-Up Chain*
1. "What are the termination provisions in the Suntron Corp Maintenance Agreement?"
2. "Is there a notice period required before termination?"
3. "What about liability — is there a cap on damages?"
4. "How does the indemnification relate to that liability cap?"

*Sequence 2: Cross-Contract Comparison*
1. "What are the parties to the Azul Sa Maintenance Agreement?"
2. "What are the key obligations of each party under that agreement?"
3. "How does the Bloom Energy Maintenance Agreement differ in terms of party obligations?"
4. "Between the Azul and Bloom Energy agreements, which one has broader termination rights?"

*Sequence 3: Mid-Conversation Topic Switch*
1. "What are the non-compete or exclusivity provisions in the Zogenix Distributor Agreement?"
2. "Are there any territorial restrictions in that agreement?"
3. "What is the scope of intellectual property assigned under the Know Labs IP Agreement?"
4. "Does that IP agreement include any license-back provisions?"

*Sequence 4: Ambiguous Pronoun Resolution*
1. "Who are the parties to the Range Resources Transportation Agreement?"
2. "Can they assign the agreement to a third party?"
3. "What's the governing law?"
4. "Does it allow termination for convenience?"

*Sequence 5: Hallucination Resistance*
1. "What are the revenue-sharing provisions in the Emmis Communications Marketing Agreement?"
2. "What about most-favored-nation clauses in that agreement?"
3. "Does the Coral Gold Consulting Agreement contain any non-compete restrictions?"
4. "How does it compare to the merger agreement between Ford and Tesla?"

## Design Decisions

- **Chunk size of 256 tokens with 80-token overlap** — chosen to balance retrieval precision (smaller chunks are easier to match) against having enough context for the LLM to produce a useful answer. The overlap helps avoid splitting important clauses across chunk boundaries.
- **Sliding window of 10 conversation turns** — keeps multi-turn conversations functional without exhausting the model's context window. Older turns are dropped as new ones come in.
- **Prompt rewriting for follow-ups** — follow-up questions like "What about the duration?" are rewritten into self-contained queries using the conversation history, which significantly improves retrieval quality on subsequent turns.
- **Separate querying per contract for comparisons** — ensures balanced representation across contracts rather than letting one contract dominate the retrieved results.
- **ChromaDB for vector storage** — provides persistence and semantic search without the operational overhead of a full client-server database.

## Future Improvements

- Add support for other models in the querying and generation phases. Right now, the MCP uses Qwen3:32b running locally for these phases. This is not ideal for several reasons: 1) many users will not be able to run a model of that size locally, 2) even those who can run it locally will often find that it slows down Claude's response time, and 3) it presents potential security issues. The latter point is not a significant issue as long as the MCP is used with publicly available sample contract databases like the CUAD dataset. But the real value of this MCP lies in applying it to actual real-world contract databases where security will be imperative. Adding the ability to swap out Qwen for another model like Claude or Gemini via API calls is at the top of my list for future improvements.
- Add support for custom contract databases. As noted in the prior bullet, the real value of this MCP is realized when users can use it with their own contract databases. A valuable future project would be to create a simple way for users to input and format their own database of contracts in a way that is compatible with this MCP. This would require putting the contracts into the same format used by the CUAD dataset.
- The list_contracts tool is workable for a database with ~500 contracts, which is the size of the CUAD dataset. However, for much larger datasets, it probably wouldn't be workable. Limiting it to listing the first ~50-100 matching contracts would probably be a more robust solution.
