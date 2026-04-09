"""Central configuration loader for Career Co-pilot.

Reads settings from a `.env` file (or real environment variables) and exposes
a shared OpenAI-compatible client factory.  Works with:
  - OpenAI       (leave OPENAI_BASE_URL blank)
  - Qwen         OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
  - Any other OpenAI-compatible API (e.g. Together, Groq, local Ollama)

Configuration priority (highest → lowest):
  1. _runtime_config  — set by app.py from sidebar inputs (for public deployment)
  2. os.getenv()      — from .env or real environment (for local dev)

.env keys
---------
OPENAI_API_KEY   — required
OPENAI_BASE_URL  — optional; set to the Qwen endpoint to use Qwen models
OPENAI_MODEL     — optional; default gpt-4o-mini / qwen-plus
OPENAI_EMBED_MODEL — optional; default text-embedding-v4
SERPER_API_KEY   — optional; needed for company culture lookup
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

# Load .env from the project root (two levels up from this file)
_ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=_ENV_PATH, override=True)  # override=True so .env always wins

# Qwen-compatible endpoint
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ---------------------------------------------------------------------------
# Runtime config — populated by app.py from st.session_state each rerun.
# Keys: OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, OPENAI_EMBED_MODEL,
#        SERPER_API_KEY
# ---------------------------------------------------------------------------
_runtime_config: dict[str, str] = {}


def set_runtime_config(cfg: dict[str, str]) -> None:
    """Replace the runtime config dict. Called by app.py on every Streamlit rerun."""
    _runtime_config.clear()
    _runtime_config.update({k: v for k, v in cfg.items() if v})


def _get(key: str, default: str = "") -> str:
    """Fetch a config value: runtime config → env var → default."""
    return _runtime_config.get(key) or os.getenv(key) or default


def get_client() -> OpenAI:
    """Return an OpenAI-compatible client.

    Checks runtime config first (sidebar input), then env vars (.env / shell).
    Raises EnvironmentError if OPENAI_API_KEY is missing from both sources.
    """
    api_key = _get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Enter it in the sidebar or add it to your .env file."
        )
    base_url = _get("OPENAI_BASE_URL") or None  # None → use OpenAI default
    return OpenAI(api_key=api_key, base_url=base_url)


def get_model(default: str = "gpt-4o-mini") -> str:
    """Return the model name from runtime config or OPENAI_MODEL env var."""
    return _get("OPENAI_MODEL", default)


def get_embed_model(default: str = "text-embedding-v4") -> str:
    """Return the embedding model name from runtime config or env var."""
    return _get("OPENAI_EMBED_MODEL", default)
