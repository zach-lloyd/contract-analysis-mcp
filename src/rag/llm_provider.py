"""
LLM provider abstraction layer.

Supports three backends:
  - "ollama": local Ollama server (default, uses qwen3:32b)
  - "claude": Anthropic API (uses claude-sonnet-4-20250514)
  - "gemini": Google Gemini API (uses gemini-2.5-flash)

Selection is done once at startup via `set_provider()`. After that, the rest
of the codebase just calls `chat(messages)` without caring which backend is
active.

API keys are read from environment variables:
  - ANTHROPIC_API_KEY  (for claude)
  - GEMINI_API_KEY     (for gemini)
"""

import os

# ---------------------------------------------------------------------------
# Provider state
# ---------------------------------------------------------------------------
_provider: str = "ollama"

# Default models per provider; override with set_provider(model=...)
_DEFAULTS = {
    "ollama": "qwen3:32b",
    "claude": "claude-opus-4-6",
    "gemini": "gemini-3.1-pro",
}

_model: str = _DEFAULTS["ollama"]
_anthropic_client = None
_gemini_client = None

VALID_PROVIDERS = list(_DEFAULTS.keys())


def set_provider(provider: str, model: str = None) -> None:
    """
    Set the active LLM provider (and optionally override the default model).

    Args:
        provider: One of "ollama", "claude", or "gemini".
        model: Optional. The model string to use. If omitted, uses the
               default for the chosen provider.
    """
    global _provider, _model

    provider = provider.lower()
    if provider not in VALID_PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Valid options: {', '.join(VALID_PROVIDERS)}"
        )

    _provider = provider
    _model = model or _DEFAULTS[provider]


def get_provider() -> str:
    """Return the name of the currently active provider."""
    return _provider


def get_model() -> str:
    """Return the model string currently in use."""
    return _model


# ---------------------------------------------------------------------------
# Unified chat interface
# ---------------------------------------------------------------------------
def chat(messages: list[dict[str, str]]) -> str:
    """
    Send a list of messages to the active LLM and return the assistant's
    response as a plain string.

    Messages follow the OpenAI/Ollama convention:
        [{"role": "system"|"user"|"assistant", "content": "..."}]

    Each backend adapter handles any format differences internally.

    Args:
        messages: The conversation messages.

    Returns:
        The assistant's response text.
    """
    if _provider == "ollama":
        return _chat_ollama(messages)
    elif _provider == "claude":
        return _chat_claude(messages)
    elif _provider == "gemini":
        return _chat_gemini(messages)


# ---------------------------------------------------------------------------
# Backend: Ollama (local)
# ---------------------------------------------------------------------------
def _chat_ollama(messages: list[dict[str, str]]) -> str:
    import ollama

    response = ollama.chat(model=_model, messages=messages)
    return response["message"]["content"]


# ---------------------------------------------------------------------------
# Backend: Anthropic Claude
# ---------------------------------------------------------------------------
def _chat_claude(messages: list[dict[str, str]]) -> str:
    """
    Anthropic's API differs from the Ollama/OpenAI convention in two ways:
      1. The system prompt is a top-level parameter, not a message with
         role="system".
      2. Messages must strictly alternate user/assistant, starting with user.
    """
    from anthropic import Anthropic

    # Cache the client to avoid creating a new client each time the function runs
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Set it before using the 'claude' provider."
            )
        _anthropic_client = Anthropic(api_key=api_key)

    # Separate system messages from conversation messages
    system_parts = []
    conversation = []

    for msg in messages:
        if msg["role"] == "system":
            system_parts.append(msg["content"])
        else:
            conversation.append({"role": msg["role"], "content": msg["content"]})

    system_text = "\n\n".join(system_parts) if system_parts else None

    # Ensure the conversation starts with a user message. If the first
    # non-system message is from the assistant (unlikely but defensive),
    # prepend an empty user turn.
    if conversation and conversation[0]["role"] != "user":
        conversation.insert(0, {"role": "user", "content": "Begin."})

    kwargs = {
        "model": _model,
        "max_tokens": 4096, # Max length of the model's response; required by Anthropic API
        "messages": conversation,
    }
    if system_text:
        kwargs["system"] = system_text

    response = _anthropic_client.messages.create(**kwargs)
    return response.content[0].text


# ---------------------------------------------------------------------------
# Backend: Google Gemini
# ---------------------------------------------------------------------------
def _chat_gemini(messages: list[dict[str, str]]) -> str:
    """
    The google-genai SDK uses a different message format:
      - System instructions are passed separately.
      - Role names: "user" and "model" (not "assistant").
    """
    from google import genai
    from google.genai import types

    # Cache the client to avoid creating a new client each time the function runs
    global _gemini_client
    if _gemini_client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY environment variable is not set. "
                "Set it before using the 'gemini' provider."
            )
        _gemini_client = genai.Client(api_key=api_key)

    # Separate system messages from conversation messages
    system_parts = []
    conversation = []

    for msg in messages:
        if msg["role"] == "system":
            system_parts.append(msg["content"])
        else:
            # Gemini uses "model" instead of "assistant"
            role = "model" if msg["role"] == "assistant" else msg["role"]
            conversation.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg["content"])],
                )
            )

    system_text = "\n\n".join(system_parts) if system_parts else None

    config = types.GenerateContentConfig(
        system_instruction=system_text,
    )

    response = _gemini_client.models.generate_content(
        model=_model,
        contents=conversation,
        config=config,
    )

    return response.text
