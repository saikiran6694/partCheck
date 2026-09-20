"""Optional LLM step: explain a Report's findings in plain language with fixes.

Uses Groq's OpenAI-compatible chat completions API. Requires GROQ_API_KEY in
the environment; enabled via the CLI's `--explain` flag, never on by default.
"""

from __future__ import annotations

import os

import requests
from dotenv import load_dotenv

from partcheck.models import Report

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-20b"
REQUEST_TIMEOUT_S = 30

SYSTEM_PROMPT = (
    "You are a manufacturing engineer reviewing an automated manufacturability "
    "report for a 3D part (STL). Explain each finding in plain language for "
    "someone who isn't a geometry expert -- what the issue is and why it "
    "matters for manufacturing -- and suggest one concrete fix per finding. "
    "Be concise: a short paragraph or a few bullet points per finding, no "
    "preamble."
)


class ExplainError(RuntimeError):
    """Raised when the LLM explanation step can't run or fails."""


def explain(report: Report, model: str = DEFAULT_MODEL) -> str:
    """Return a plain-language explanation of `report`'s findings.

    Raises ExplainError if GROQ_API_KEY is unset or the API call fails.
    """
    if not report.findings:
        return "No manufacturability issues were found."

    load_dotenv()  # picks up GROQ_API_KEY from a .env file, if present; never overrides an already-set var
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ExplainError("GROQ_API_KEY is not set; cannot call the LLM explainer.")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": report.model_dump_json(indent=2)},
        ],
        "temperature": 0.2,
        "max_tokens": 800,
    }

    try:
        response = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=REQUEST_TIMEOUT_S,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        raise ExplainError(f"Groq API request failed: {e}") from e

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise ExplainError(f"Unexpected Groq API response shape: {data}") from e
