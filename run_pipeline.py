# run_pipeline.py
# Orchestrates: PDF -> (regex parser) -> JSON -> (LLM post) -> final JSON
#
# Usage:
#   python run_pipeline.py <input.pdf> <final_output.json>
#
# Create a .env file with your OPENAI_API_KEY and optional OPENAI_MODEL. 

import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path
import llm_post


def run_regex_parser(input_pdf: str, tmp_json: str) -> None:
    """
    Runs your existing rahul_panchal.py as a separate process
    to keep your current script intact.
    """
    script_path = Path(__file__).parent / "rahul_panchal.py"
    if not script_path.exists():
        raise RuntimeError(f"rahul_panchal.py not found at {script_path}")

    cmd = [sys.executable, str(script_path), input_pdf, tmp_json]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"rahul_panchal.py failed: {e}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python run_pipeline.py <input.pdf> <final_output.json>", file=sys.stderr)
        sys.exit(2)

    input_pdf = sys.argv[1]
    final_out = sys.argv[2]

    if not os.path.exists(input_pdf):
        raise FileNotFoundError(f"Input PDF not found: {input_pdf}")

    os.makedirs(os.path.dirname(os.path.abspath(final_out)), exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_json = os.path.join(tmpdir, "parsed.json")

        print("Step 1/2: parsing PDF with rahul_panchal.py ...")
        run_regex_parser(input_pdf, tmp_json)

        try:
            with open(tmp_json, "r", encoding="utf-8") as f:
                parsed = json.load(f)
            title = parsed.get("title", "")[:120]
            sec_count = len(parsed.get("sections", []))
            print(f"  Parsed title: {title}")
            print(f"  Sections detected: {sec_count}")
        except Exception:
            print("  (Could not preview parsed JSON)")

        print("Step 2/2: post-processing with LLM ...")
        llm_post.enrich_contract(tmp_json, final_out)

    print(f"✅ Done. Final JSON: {final_out}")


if __name__ == "__main__":
    main()
