from pathlib import Path
from ingestion.extract import load_contracts_from_directory

_TEST_CONTRACTS = str(Path(__file__).parent / "test-contracts")
NUM_TO_CHECK = 5
NUM_CHARS_TO_PRINT = 500

contracts = load_contracts_from_directory(_TEST_CONTRACTS)

# Print out excerpts of the first few contracts for manual inspection
for c in contracts[:NUM_TO_CHECK]:
    print(f"\n=== {c['title']} ===")
    print(c['text'][:NUM_CHARS_TO_PRINT])
