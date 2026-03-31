import chromadb
import ollama

CLIENT = chromadb.PersistentClient(path="./chroma_data")
COLLECTION = CLIENT.get_or_create_collection(name="legal_contracts")


def query_clauses(question: str, num_results: int, contract_title: str = None):
    """
    Query the contract database for the chunks that are most relevant to the 
    question.

    Args:
        question: The user's question.
        num_results: The number of contract chunks for the query to return.
        contract_title: Optional. If present, limit the search to a specific 
                        contract. If not, search across all contracts in the
                        database.
    """
    if contract_title:
        results = COLLECTION.query(
            query_texts=[question],
            n_results=num_results
        )
    else:
        results = COLLECTION.query(
            query_texts=[question],
            n_results=num_results,
            where={"contract_title": contract_title}
        )
    
    return results


def rewrite_prompt(question: str, history: list[dict[str, str]]) -> str:
    """
    Rewrite the user's latest question, taking into account the existing conversation
    history, in 256 words or less.

    Args:
        question: The user's latest question.
        history: A list of dictionaries containing the user's previous questions and
                 the LLM's previous responses.
    """
    chat_text = ""

    for message in history:
        chat_text += f"{message['role']}: "
        chat_text += message["content"]
    
    rewritten_prompt = ollama.chat(
        model="qwen3:32b",
        messages=[
            {
                "role": "system",
                "content": f"Conversation History: {chat_text}"
            },
            {
                "role": "system",
                # Limit to 256 words to avoid eating up too much of the context window
                "content": "Given this conversation history, rewrite the user's latest "
                           "question as a fully self-contained query in 256 words or less "
                           "that could be understood without any prior context."
            }, 
            {
                "role": "system",
                "content": f"User's Latest Question: {question}"
            }  
        ]
    )

    return rewritten_prompt["message"]["content"]


def generate_answer(question, clauses, history=None):
    # Handle limited context by using a sliding window of 10 turns. Each turn 
    # (each call of get_answer) results in 2 new messages being appended to the
    # conversation, so 10 turns are stored in the conversation history at any
    # given time
    if history and len(history) >= 20:
        history[:] = history[2:]
    
    if history:
        prompt = rewrite_prompt(question, history)
    else:
        prompt = question
