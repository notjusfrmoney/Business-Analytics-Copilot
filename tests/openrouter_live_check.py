from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm.client import LLMClient, LLMError
from src.utils.config import get_openrouter_settings


def main() -> int:
    try:
        get_openrouter_settings()
    except ValueError as exc:
        print(f"OpenRouter live check not run: {exc}")
        return 1

    client = LLMClient()
    try:
        content = client.generate_text("Return exactly this DuckDB query: SELECT 1;")
        if not content.strip():
            raise LLMError("OpenRouter returned an empty response.")
    except LLMError as exc:
        print(f"OpenRouter live API request failed: {exc}")
        return 1

    print("OpenRouter live API request succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
