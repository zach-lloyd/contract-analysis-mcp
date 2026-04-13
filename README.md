# Legal Contract RAG — MCP Server

A retrieval-augmented generation (RAG) system for analyzing legal contracts, exposed as an [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server. It lets an LLM-powered assistant answer natural-language questions about a corpus of contracts, compare provisions across agreements, and locate specific clause types.

The project uses the [Contract Understanding Atticus Dataset (CUAD)](https://www.atticusprojectai.org/cuad) as its contract corpus and ChromaDB for vector storage and retrieval, and can be connected to any MCP-compatible client (including frontier LLMs like Claude and ChatGPT, as well as open-source models).

## How It Works

Contracts are loaded from the CUAD dataset, split into overlapping token-level chunks, and indexed into a ChromaDB collection. The title of each contract and the names of the parties are also stored in the collection as metadata. At query time, the system retrieves the most relevant chunks for a given question and passes them to the client LLM to generate a grounded answer.

The MCP helps preserve conversation history to facilitate smooth multi-turn workflows by caching up to 30 previously-retrieved contract clauses at a time so the model can reference them across turns.

## Brief Demo

https://github.com/user-attachments/assets/6e4fb023-72c6-4f74-b13b-30104ffb7d8f

## MCP Tools

The server exposes five tools:

**`ask_contracts`** — Ask a question across all contracts in the database. Returns the most relevant clauses for the client LLM to use when generating an answer. Supports clause caching via a `session_id`.

**`ask_contract`** — Ask a question about a specific contract by title. Restricts retrieval to that single agreement. Also supports clause caching via a `session_id`.

**`compare_contracts`** — Compare two or more contracts on a given topic. Returns the most relevant clauses for each contract so the client LLM can produce a side-by-side analysis. Each contract is queried independently so results are balanced, and the LLM produces a side-by-side comparative analysis.

**`find_contract_clauses`** — Search for a specific clause type (e.g., "termination", "governing law", "non-compete") across all contracts or within a single contract. Returns the raw excerpts and their source contract titles.

**`list_contracts`** — List all contracts in the database, optionally filtered by party name. Useful for discovering available contract titles before calling the other tools.

## Project Structure

```
.
├── src/
│  ├── rag/
│  │   ├── chunker.py          # Loads CUAD contracts, chunks them, indexes into ChromaDB
│  │   ├── rag_core.py         # Core clause retrieval logic; retrieves contract clauses based on user's question
│  │   ├── server.py           # MCP server exposing the five tools above
│  │   └── chroma_data/        # ChromaDB persistent storage (generated)
│  └── testing/
│      ├── query_testing.py        # Measures retrieval accuracy (conceptual vs. factual)
│      ├── rag_smoke_test.py       # Smoke tests for rag_core functions
│      └── integration_testing.py  # Integration tests for the MCP server tools
└── README.md
```

## Prerequisites

- **Python 3.10+**
- **Client LLM (either Claude Desktop or some other MCP-compatible LLM)**
- **[uv](https://docs.astral.sh/uv/)** (used to run the MCP server)
- The **CUAD dataset** — download `CUADv1.json` and `train_separate_questions.json` and place them in `cuad/data/`

## Setup

1. **Install dependencies:**

   ```bash
   pip install chromadb langchain-text-splitters tiktoken mcp
   ```

2. **Build the vector database:**

   ```bash
   python3 rag/chunker.py
   ```

   This loads all contracts from CUAD, chunks them into ~256-token segments with 80-token overlap, and indexes them into a persistent ChromaDB collection. Only needs to be run once.

3. Connect the MCP server to an MCP client. For Claude Desktop, you would add the following to claude_desktop_config.json:

```json
{
  "mcpServers": {
    "contracts": {
      "command": "/Users/zachlloyd/.local/bin/uv",
      "args": [
        "run",
        "--directory",
        "[LOCATION WHERE THIS MCP IS SAVED]", # Replace this with the path for the project's root folder (e.g., "/Users/my-name/contract-analysis-mcp")
        "python3",
        "src/rag/server.py"
      ]
    }
  }
}
```

## Testing

The project includes four levels of testing:

**RAG Smoke tests** (`rag_smoke_test.py`) — Quick functional tests for the clause retrieval functions (querying and listing). To run these, from the src folder run:

```bash
uv run python3 -m testing.rag_smoke_test          
```

**Retrieval accuracy** (`query_testing.py`) — Samples question/answer pairs from the CUAD training set and measures how often the correct answer appears in the top-k retrieved chunks. Reports accuracy separately for conceptual questions (e.g., "Is there a non-compete clause?") and factual questions (e.g., "What is the effective date?"). To determine how much accuracy improves when additional results are retrieved, these tests run repeatedly with NUM_RESULTS set to each value between 1 and 10. This can be altered by tweaking the NUM_RESULTS parameter at the beginning of the file. To run these, run the below command in the src folder. **Note that these can take awhile to run, particularly when testing multiple values for NUM_RESULTS.**

```bash
uv run python3 -m testing.query_testing
```

**Integration tests** (`integration_testing.py`) — Spins up the MCP server and exercises each tool through the MCP client protocol, including happy-path and error-handling scenarios. Run the below in the src folder.

```bash
uv run python3 -m testing.integration_testing
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
- **Caching up to 30 contract clauses at a time** — maintains previously retrieved contract clauses to facilitate handling of ambiguous follow-up questions and help multi-turn conversations flow more smoothly.
- **Separate querying per contract for comparisons** — ensures balanced representation across contracts rather than letting one contract dominate the retrieved results.
- **ChromaDB for vector storage** — provides persistence and semantic search without the operational overhead of a full client-server database.

## Future Improvements

- Add support for custom contract databases. As noted in the prior bullet, the real value of this MCP is realized when users can use it with their own contract databases. A valuable future project would be to create a simple way for users to input and format their own database of contracts in a way that is compatible with this MCP. This would require putting the contracts into the same format used by the CUAD dataset.
- The list_contracts tool is workable for a database with ~500 contracts, which is the size of the CUAD dataset. However, for much larger datasets, it probably wouldn't be workable. Limiting it to listing the first ~50-100 matching contracts would probably be a more robust solution.
- Add some automated generation testing in addition to the manual testing I described above. This might be complicated but could be accomplished by giving a separate LLM the relevant contract clauses and the client LLM's answer and asking the separate LLM to grade the client's answer on a scale of 1-5 and provide a one-sentence justification for its answer.
