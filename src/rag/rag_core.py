import chromadb
import ollama

CLIENT = chromadb.PersistentClient(path="./chroma_data")
COLLECTION = CLIENT.get_or_create_collection(name="legal_contracts")
NUM_RESULTS = 10
SYS_PROMPT = """
             You are a helpful legal assistant. Provide a concise but thorough 
             answer to the user's latest question, using the relevant clauses 
             from existing legal contracts provided below. The prior history of this
             conversation, if any, is also provided below. You may use it as additional
             context when providing your answer, but your answer should be primarily
             based on the relevant contract clauses that have been provided. If
             the prior answer references a specific contract, you should bias
             towards focusing on that contract in referencing the user's current
             question, unless their question makes it clear that they want you to 
             refer to other contracts. If the answer is not available. You should 
             be honest about that. Do not hallucinate a false answer. Rather, you 
             should respond 'I don't have information about that.'
             """


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
            n_results=num_results,
            where={"contract_title": contract_title}
        )
    else:
        results = COLLECTION.query(
            query_texts=[question],
            n_results=num_results
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


def list_matching_contracts(party_name: str = None) -> list[dict[str, str]]:
    """
    List all contracts in the database, or only those involving a specific party.
    Returns deduplicated contract titles and their associated parties.
 
    Args:
        party_name: Optional. If provided, only return contracts where this party
                    appears in the parties metadata. If omitted, return all contracts.
    """
    # Retrieve all metadata from the collection. ChromaDB requires fetching 
    # documents alongside metadata, but we only need the metadata here
    all_data = COLLECTION.get(include=["metadatas"])
    metadatas = all_data["metadatas"]
 
    # Deduplicate by contract title, keeping the parties metadata
    seen = {}
    for meta in metadatas:
        title = meta["contract_title"]
        if title not in seen:
            seen[title] = meta.get("parties", "")
 
    contracts = [
        {"contract_title": title, "parties": parties}
        for title, parties in seen.items()
    ]
 
    # If a party name was provided, filter to contracts involving that party
    if party_name:
        contracts = [
            c for c in contracts
            if party_name.lower() in c["parties"].lower()
        ]
    
    return contracts


def generate_answer(
        question: str, 
        contract_title: str = None, 
        clauses: list[str] = None, 
        history: list[dict[str, str]] = None
        ) -> tuple[str, list[dict[str, str]], list[str]]:
    """
    Queries the database for relevant clauses based on the user's question, then 
    feeds those clauses, previously retrieved clauses, and the user's question 
    into an LLM, rewritten to capture previous conversation history if necessary.
    Returns the LLM's answer to the question, as well as updated history and clauses.

    Args:
        question: The user's question.
        contract_title: Optional. If specified, the query is restricted to this contract.
        clauses: Optional. List of clauses that have been retrieved in previous turns of 
                 the conversation.
        history: Optional. Prior question and answers from the last 10 turns of the 
                 conversation.
    """
    if clauses is None:
        clauses = []
    if history is None:
        history = []

    # Handle limited context by using a sliding window of 10 turns. Each turn 
    # (each call of get_answer) results in 2 new messages being appended to the
    # conversation, so 10 turns are stored in the conversation history at any
    # given time
    if len(history) >= 20:
        history = history[2:]
    
    if len(history) > 0:
        prompt = rewrite_prompt(question, history)
    else:
        prompt = question
    
    results = query_clauses(prompt, NUM_RESULTS, contract_title)

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]

    for meta, chunk in zip(metadatas, chunks):
       title_and_excerpt = f"Contract Title: {meta['contract_title']}\nContract Excerpt: {chunk}\n\n"

       if title_and_excerpt not in clauses:
           clauses.append(title_and_excerpt)
    
    if len(clauses) > 30:
        clauses[:] = clauses[-30:]
    
    context = " ".join(clauses)

    # Add the retrieved clauses to the system prompt
    full_sys_prompt = f"""
                  {SYS_PROMPT} The relevant contract clauses are as follows: {context}.
                  The prior history of this conversation is below, ending with the
                  user's current question. If there is no text provided other than a
                  question from the user, then this is the first question and there 
                  is no conversation history to reference.
                  """
    
    # The system prompt is not appended to the conversation history to avoid quickly
    # eating up the context window
    sys_message = {"role": "system", "content": full_sys_prompt}

    history.append({"role": "user", "content": question})

    response = ollama.chat(
        model="qwen3:32b",
        messages=[sys_message] + history
    )

    answer = response["message"]["content"]

    history.append({"role": "assistant", "content": answer})

    return answer, history, clauses
