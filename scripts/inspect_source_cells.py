"""Check an existing v4/v5 guarded JSON against the source table; no Ollama."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tech_specs.source_cell_check import check_source_cell_answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not args.report.is_file():
        parser.error("Report not found")
    record = json.loads(args.report.read_text(encoding="utf-8"))
    original = record.get("original_result") or record
    check = check_source_cell_answer(original)
    print(f"Source-cell check: accepted={check['accepted']} | applicable={check['applicable']}")
    print(f"Kind: {check['kind']} | Source: {check['source_id']} | Cells: {len(check['cells'])}")
    print(f"Reason: {check['reason']}")
    if check["accepted"]:
        print("\n" + original["draft"])
        print("\nSource text SHA256: " + check["source_sha256"])
        print("\nNote: deterministic source-cell reconstruction, not an independent PDF audit.")
    else:
        print("\nNo deterministic override. Keep the normal LLM review and its recorded status.")


if __name__ == "__main__":
    main()
