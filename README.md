# Legal Contract MCP Server

An end-to-end pipeline that allows users to create a database from a folder of contracts and connect it to an LLM via a [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server. It lets an LLM-powered assistant answer natural-language questions about the user's corpus of contracts, compare provisions across agreements, and locate specific clause types.

The project uses ChromaDB for vector storage and retrieval, and can be connected to any MCP-compatible client (including frontier LLMs like Claude and ChatGPT, as well as open-source models).

This project evolved from a legal RAG agent that I previously built, which is located in [this repo](https://github.com/zach-lloyd/legal-contract-rag).

## How It Works

In the command line, run ingest.py while specifying the folder containing your contracts, as described in the Setup section below. The ingestion and extraction code will take your contracts as input and split each of them into overlapping token-level chunks, indexed into a ChromaDB collection. You have the option to include a metadata json file that includes the title of each contract and the names of its parties. If no such file is included, the program will extract the title of each contract from its filename and store it as metadata (no party names will be stored as metadata unless manually specified in a metadata.json file). 

At query time, the system retrieves the most relevant chunks for a given question and passes them to the client LLM to generate a grounded answer. The MCP helps preserve conversation history to facilitate smooth multi-turn workflows by caching up to 30 previously-retrieved contract clauses at a time so the model can reference them across turns.

Support for multiple collections of contracts is included, so you can divide your contracts based on client, contract type, etc.

**NOTE: At this time, the system only supports .pdf and .docx files. If you want to upload contracts that are .doc files or more exotic filetypes, you will first need to convert them to one of the supported file types**

## Brief Demo

https://github.com/user-attachments/assets/6e4fb023-72c6-4f74-b13b-30104ffb7d8f

## MCP Tools

The server exposes five tools:

**`ask_contracts`** — Ask a question across all contracts in the database. Returns the most relevant clauses for the client LLM to use when generating an answer. Supports clause caching via a `session_id`.

**`ask_contract`** — Ask a question about a specific contract by title. Restricts retrieval to that single agreement. Also supports clause caching via a `session_id`.

**`compare_contracts`** — Compare two or more contracts on a given topic. Returns the most relevant clauses for each contract so the client LLM can produce a side-by-side analysis. Each contract is queried independently so results are balanced.

**`find_contract_clauses`** — Search for a specific clause type (e.g., "termination", "governing law", "non-compete") across all contracts or within a single contract. Returns the raw excerpts and their source contract titles.

**`list_contracts`** — List all contracts in the database, optionally filtered by party name. Useful for discovering available contract titles before calling the other tools.

## Project Structure

```
.
├── src/
│  ├── ingestion/
│  │   ├── extract.py                             # Extracts text from contracts and puts it in the format the chunker expects
│  │   └── ingest.py                              # CLI entrypoint for ingesting user contracts
│  ├── rag/
│  │   ├── chunker.py                             # Loads CUAD contracts, chunks them, indexes into ChromaDB
│  │   ├── rag_core.py                            # Core clause retrieval logic; retrieves contract clauses based on user's question
│  │   ├── server.py                              # MCP server exposing the five tools above
│  │   └── chroma_data/                           # ChromaDB persistent storage (generated)
│  └── testing/
│      ├── extraction_manual_inspection.py        # Prints excerpts of the first few contracts for manual confirmation that text was extracted correctly
│      ├── extraction_smoke_test.py               # Smoke tests for extraction functions
│      ├── extraction_testing.py                  # Validation tests for extraction module using CUAD dataset
│      ├── query_testing.py                       # Measures retrieval accuracy (conceptual vs. factual) using CUAD dataset
│      ├── rag_smoke_test.py                      # Smoke tests for rag_core functions
│      ├── integration_testing.py                 # Integration tests for the MCP server tools
│      └── test-contracts/                        # Sample library of publicly-available contracts for use in extraction_testing
└── README.md
```

## Prerequisites

- **Python 3.10+**
- **Client LLM (either Claude Desktop or some other MCP-compatible LLM)**
- **[uv](https://docs.astral.sh/uv/)** (used to run the MCP server)
- If you want to run query_testing.py, the **[Contract Understanding Atticus Dataset (CUAD)](https://www.atticusprojectai.org/cuad) dataset** — download `CUADv1.json` and `train_separate_questions.json` and place them in `cuad/data/`

## Setup

1. **Install dependencies:**

   ```bash
   uv pip install chromadb langchain-text-splitters tiktoken mcp
   ```

2. **Run ingest.py pointed at your contract folder to build the collection:**

From the src folder, run:

   ```bash
   uv run python3 -m ingestion.ingest path/to/contract-folder \
    --metadata path/to/metadata.json \ 
    --collection collection-name 
   ```

   This loads all contracts in the specified folder, chunks them into ~256-token segments with 80-token overlap, and indexes them into a persistent ChromaDB collection.

   Providing a metadata.json file is optional but strongly recommended in order to ensure the system has accurate titles and party names.

   Providing a collection name is optional but also strongly recommended. If you do not provide a collection name, it will give your collection the default name legal_contracts.

   If you want to update a collection that has already been created, you must also include the --rebuild flag.

   **IMPORTANT: When you include the --rebuild flag, the current version of the specified collection will be overwritten.**
   
3. **Connect the MCP server to an MCP client.**

For example, if you wanted to connect this server to Claude Desktop, you would add the below to claude_desktop_config.json. Replace "[PATH TO UV]" with the path to the folder where uv is saved (e.g., "/Users/your-name/.local/bin/uv") and replace "[LOCATION WHERE THIS MCP IS SAVED]" with the path to where you cloned the repo on your system (e.g., "/Users/your-name/contract-analysis-mcp").

```json
{
  "mcpServers": {
    "contracts": {
      "command": "[PATH TO UV]", 
      "args": [
        "run",
        "--directory",
        "[LOCATION WHERE THIS MCP IS SAVED]", 
        "python3",
        "src/rag/server.py"
      ]
    }
  }
}
```

## Testing

The project includes five levels of testing:

**Extraction tests** (`extraction_smoke_test.py` and `extraction_testing.py`) - Tests to ensure the system correctly ingests and extracts text from contracts. To run these, from the src folder run:

```bash
uv run python3 -m testing.extraction_smoke_test
uv run python3 -m testing.extraction_testing          
```

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

**Manual end-to-end tests** - In addition to the above automated tests, I also performed extensive manual testing of the MCP after connecting it to Claude. After activating the MCP, I asked Claude the below question sequences, most of which pertain to a sample collection I created using the CUAD dataset. For the last sequence, I created a separate collection of a few publicly available Silicon Valley Bank contracts, to ensure the server correctly handles multiple collections.

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

*Sequence 6: Multiple Collections*
1. "What are the termination provisions in the Suntron Corp Maintenance Agreement?"
2. "What about the loan and security agreement between Silicon Valley Bank and Invision?"
3. "Are there any restrictions on the parties' assignment rights?"
4. "Has SVB ever agreed to a dollar threshold for cross defaults? If so, what amount is typical?"

## Design Decisions

- **Chunk size of 256 tokens with 80-token overlap** — chosen to balance retrieval precision (smaller chunks are easier to match) against having enough context for the LLM to produce a useful answer. The overlap helps avoid splitting important clauses across chunk boundaries.
- **Caching up to 30 contract clauses at a time** — maintains previously retrieved contract clauses to facilitate handling of ambiguous follow-up questions and help multi-turn conversations flow more smoothly.
- **Separate querying per contract for comparisons** — ensures balanced representation across contracts rather than letting one contract dominate the retrieved results.
- **ChromaDB for vector storage** — provides persistence and semantic search without the operational overhead of a full client-server database.
- **Support metadata storage via a metadata.json file** - provides the ability for users to add additional information about their contracts that will help the LLM locate the correct contracts and accurately answer questions about them. This is a lightweight and straightforward solution that avoids requiring a separate UI or additional LLM calls.
- **Support ingestion and extraction from .pdf and .docx files** - by far the most common contract types. I would have liked to also include support for older .doc files, but those are a bit trickier to deal with and converting .doc files to .docx files is fairly easy, so for now I think .docx support is sufficient.

## Future Improvements

- The list_contracts tool is workable for a database with ~500 contracts, which is the size of the CUAD dataset. However, for much larger datasets, it probably wouldn't be workable. Limiting it to listing the first ~50-100 matching contracts would probably be a more robust solution.
- Add some automated generation testing in addition to the manual testing I described above. This might be complicated but could be accomplished by giving a separate LLM the relevant contract clauses and the client LLM's answer and asking the separate LLM to grade the client's answer on a scale of 1-5 and provide a one-sentence justification for its answer.
- Add the ability for users to use an LLM to extract contract title and party data. Implementing this would be a bit complex, but would make the process of gathering and uploading metadata much less cumbersome.
- Add support for other file types (primarily .doc). A bit of a lower priority since the vast majority of contracts will be .pdf and .docx files, which are already supported. But it is still common to find contracts in .doc format, so being able to support those without them needing to be converted would be a nice addition.
