"""Run a local, second-pass claim-to-source audit of ask_local.py output.

Example:
  python scripts/check_answer.py --report data/processed/tim2_structured.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tech_specs.faithfulness import OllamaEvidenceJudge, audit_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True,
                        help="JSON saved by scripts/ask_local.py --output")
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--num-predict", type=int, default=450)
    parser.add_argument("--output", type=Path,
                        help="Where to save audit JSON; default is beside input file")
    args = parser.parse_args()
    if not args.report.is_file():
        parser.error(f"Report not found: {args.report}")
    output = args.output or args.report.with_name(args.report.stem + "_faithfulness.json")
    if output.resolve() == args.report.resolve():
        parser.error("Audit output cannot overwrite the input report")
    if output.exists():
        parser.error(f"Output already exists: {output}")

    report = json.loads(args.report.read_text(encoding="utf-8"))
    judge = OllamaEvidenceJudge(model=args.model, num_ctx=args.num_ctx,
                                num_predict=args.num_predict)
    try:
        result = audit_report(report, judge)
    except (ValueError, RuntimeError) as exc:
        parser.exit(2, f"Audit could not finish: {exc}\n")

    print(f"Question: {result['question']}")
    for item in result["claims"]:
        print(f"\nClaim {item['number']}: {item['verdict'].upper()}")
        print("  " + item["claim"])
        print("  Sources: " + ", ".join(item["source_ids"]))
        print("  Judge: " + item["reason"])
        if item["issue"]:
            print("  Check: " + item["issue"])
        for qc in item["quote_checks"]:
            print("  Verbatim quote match: " + ("YES" if qc["verbatim_found"] else "NO")
                  + f" ({qc['source_id']})")
    print(f"\nAudit status: {result['audit_status']}")
    print("Caution: a local LLM audit is NOT independent proof of correctness.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report saved to: {output}")


if __name__ == "__main__":
    main()
