"""Central configuration loader for Career Co-pilot.

Reads settings from a `.env` file (or real environment variables) and exposes
a shared OpenAI-compatible client factory.  Works with:
  - OpenAI       (leave OPENAI_BASE_URL blank)
  - Qwen         OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
  - Any other OpenAI-compatible API (e.g. Together, Groq, local Ollama)

.env keys
---------
OPENAI_API_KEY   — required
OPENAI_BASE_URL  — optional; set to the Qwen endpoint to use Qwen models
OPENAI_MODEL     — optional; default gpt-4o-mini / qwen-plus
SERPER_API_KEY   — optional; needed for company culture lookup
"""
import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

# Load .env from the project root (two levels up from this file)
_ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=_ENV_PATH, override=False)

# Qwen-compatible endpoint
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Return a cached OpenAI-compatible client.

    Automatically uses OPENAI_BASE_URL if set (e.g. for Qwen).
    Raises EnvironmentError if OPENAI_API_KEY is missing.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Add it to your .env file or export it in your shell."
        )
    base_url = os.getenv("OPENAI_BASE_URL") or None  # None → use OpenAI default
    return OpenAI(api_key=api_key, base_url=base_url)


def get_model(default: str = "gpt-4o-mini") -> str:
    """Return the model name from OPENAI_MODEL env var, or the given default."""
    return os.getenv("OPENAI_MODEL") or default
