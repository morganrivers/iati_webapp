"""Single source of truth for constructing google-genai clients.

Reads the Gemini API key from the environment and returns a client already
wrapped for LangSmith tracing.
"""

import os

from google import genai

from llm_tracing import wrap_genai_client


def make_genai_client():
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY_GEMINI")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY (or GOOGLE_API_KEY_GEMINI).")
    return wrap_genai_client(genai.Client(api_key=api_key))
