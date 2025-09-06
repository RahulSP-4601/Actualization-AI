# llm_post.py
# Post-processes rahul_panchal.py JSON with an LLM:
# - validates the schema
# - lightly normalizes titles/labels
# - fixes obvious parsing glitches (without inventing text)
# - guarantees the output matches the expected JSON schema
#
# Usage:
#   python llm_post.py <input.json> <output.json>

# Create a .env file with your OPENAI_API_KEY and optional OPENAI_MODEL. 
# Use OPENAI_API_KEY=REDACTED
#OPENAI_MODEL=gpt-4o-mini

import os
import sys
import json
from typing import Any, Dict

from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError(
        "openai package not installed. Run: pip install openai python-dotenv"
    ) from e

load_dotenv() 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found. Add it to your .env file.")

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=OPENAI_API_KEY)

SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "title": {"type": "string"},
        "contract_type": {"type": "string"},
        "effective_date": {
            "type": ["string", "null"],
            "description": "ISO date (YYYY-MM-DD) or null",
            "pattern": r"^\d{4}-\d{2}-\d{2}$",
        },
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "number": {"type": ["string", "null"]},
                    "clauses": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "text": {"type": "string"},
                                "label": {"type": "string"},
                                "index": {"type": "integer", "minimum": 0},
                            },
                            "required": ["text", "label", "index"],
                        },
                    },
                },
                "required": ["title", "number", "clauses"],
            },
        },
    },
    "required": ["title", "contract_type", "effective_date", "sections"],
}


SYSTEM_PROMPT = """You are a careful, literal post-processor.
You receive a JSON object produced by a regex-based contract parser.
Your job is to:
- keep ALL original clause text exactly as-is (do not paraphrase or invent)
- minimally fix obvious structural glitches (e.g., wrong section titles cased oddly, duplicated heading lines, stray leading section numbers inside clause text)
- normalize minor label styles (e.g., '1)' vs '1.' consistency inside a section) only if safe
- ensure the final output COMPLIES 100% with the provided JSON schema
- NEVER invent new clauses, sections, dates, or words that didn’t exist
- If unsure, prefer leaving fields unchanged.

Rules:
- Preserve the meaning and exact wording of “text” fields.
- You may re-title a section (e.g., titlecase) but do not invent new titles.
- You may switch a null-like empty string number to null (and vice versa) only if consistent.
- The output MUST be valid JSON and match the schema exactly.
"""

USER_INSTRUCTIONS = """Input is a parsed contract JSON. 
Please return a corrected JSON that:
- conforms to the JSON schema
- fixes trivial, mechanical issues only (no hallucinations)
- keeps clause text verbatim.

If you detect content outside the schema, discard it.
"""


def _call_llm(input_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Call the LLM with JSON schema enforced output."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_INSTRUCTIONS},
        {
            "role": "user",
            "content": json.dumps(input_obj, ensure_ascii=False, separators=(",", ":")),
        },
    ]

    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "contract_post", "strict": True, "schema": SCHEMA},
        },
    )

    content = resp.choices[0].message.content
    if not content:
        raise RuntimeError("LLM returned empty content.")

    try:
        obj = json.loads(content)
    except Exception as e:
        raise RuntimeError(f"Failed to parse LLM JSON: {e}\nRaw: {content[:4000]}")

    return obj


def enrich_contract(input_json_path: str, output_json_path: str) -> None:
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fixed = _call_llm(data)

    for s in fixed.get("sections", []):
        prev = -1
        for c in s.get("clauses", []):
            if c["index"] <= prev:
                
                for i, clause in enumerate(s["clauses"]):
                    clause["index"] = i
                break
            prev = c["index"]

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(fixed, f, ensure_ascii=False, indent=2)


def _cli():
    if len(sys.argv) != 3:
        print("Usage: python llm_post.py <input.json> <output.json>", file=sys.stderr)
        sys.exit(2)
    enrich_contract(sys.argv[1], sys.argv[2])
    print(f"LLM post-processed JSON written to: {sys.argv[2]}")


if __name__ == "__main__":
    _cli()
