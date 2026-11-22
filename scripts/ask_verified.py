"""Ask the guarded local LangGraph with one optional answer-repair attempt.

Keeps the previous ask_local.py and indexes unchanged. No benchmark gold data
enters the model. Exact table-cell answers use a deterministic check; other answers retain local LLM audits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tech_specs.answer_graph import build_answer_graph
from tech_specs.faithfulness import OllamaEvidenceJudge
from tech_specs.local_answer import LocalOllama
from tech_specs.retrieval import BM25Retriever, load_chunks
from tech_specs.review_models import OllamaCoverageJudge, OllamaRepair
from tech_specs.table_answers import TableFirstGenerator
from tech_specs.verified_answer_graph import build_verified_answer_graph

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--method", choices=("bm25", "vector", "hybrid"), default="bm25")
    parser.add_argument("--document", choices=("attiny85", "stm32l432kc"))
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--candidates-per-facet", type=int, default=10)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--num-predict", type=int, default=800)
    parser.add_argument("--no-repair", action="store_true", help="Audit only, no revision")
    parser.add_argument("--no-table-first", action="store_true", help="Use the original local LLM even when a structured table is recognized")
    parser.add_argument("--corpus", type=Path, default=PROCESSED / "chunks.jsonl")
    parser.add_argument("--db-path", type=Path, default=PROCESSED / "qdrant_local")
    parser.add_argument("--manifest", type=Path, default=PROCESSED / "vector_index_manifest.json")
    parser.add_argument("--output", type=Path, help="Save complete state JSON (must not exist)")
    args = parser.parse_args()
    if args.top_k < 1 or args.candidates_per_facet < 1:
        parser.error("--top-k and --candidates-per-facet must be positive")
    if args.output and args.output.exists():
        parser.error("Output exists. Choose a new --output path to preserve prior runs")

    chunks = load_chunks(args.corpus)
    bm25 = BM25Retriever(chunks) if args.method != "vector" else None
    client = None
    try:
        if args.method == "bm25":
            retriever = bm25
        else:
            from fastembed import TextEmbedding
            from qdrant_client import QdrantClient
            from tech_specs.vector_retrieval import (
                MODEL_NAME, VectorRetriever, load_manifest, verify_index,
            )
            manifest = load_manifest(args.manifest)
            if manifest.get("partial_index"):
                raise ValueError("Cannot use a partial vector index")
            client = QdrantClient(path=str(args.db_path))
            verify_index(client, manifest, corpus_path=args.corpus)
            vectors = VectorRetriever(client, TextEmbedding(model_name=MODEL_NAME))
            if args.method == "vector":
                retriever = vectors
            else:
                from tech_specs.hybrid_retrieval import HybridRetriever
                retriever = HybridRetriever(bm25, vectors)

        generator = TableFirstGenerator(
            LocalOllama(model=args.model, num_ctx=args.num_ctx, num_predict=args.num_predict),
            enabled=not args.no_table_first,
        )
        base_graph = build_answer_graph(
            retriever, generator,
            top_k=args.top_k, candidates_per_facet=args.candidates_per_facet,
        )
        graph = build_verified_answer_graph(
            base_graph,
            judge=OllamaEvidenceJudge(model=args.model, num_ctx=args.num_ctx),
            coverage_judge=OllamaCoverageJudge(model=args.model, num_ctx=args.num_ctx),
            repairer=OllamaRepair(model=args.model, num_ctx=args.num_ctx,
                                 num_predict=args.num_predict),
            max_repairs=0 if args.no_repair else 1,
            source_cell_route=not args.no_table_first,
        )
        result = graph.invoke({"question": args.question, "document_hint": args.document})
        result["initial_answer_strategy"] = generator.last_strategy
        result["final_answer_strategy"] = (
            "local_llm_repair" if result.get("repairs", 0) and
            result.get("working_result", {}).get("status") == "citations_checked"
            else ("repair_failed_or_abstained" if result.get("repairs", 0)
                  else generator.last_strategy)
        )
        if generator.last_extraction is not None:
            extracted = generator.last_extraction
            result["table_extraction"] = {
                "source_id": extracted.source_id, "kind": extracted.kind,
                "cells": list(extracted.cells),
            }
        if result.get("status") == "source_table_checked":
            result["final_answer_strategy"] = "source_table_extraction"
        print(f"Answer strategy: initial={result['initial_answer_strategy']} | final={result['final_answer_strategy']}")
        check = result.get("source_cell_check") or {}
        if check.get("applicable"):
            print(f"Source-cell check: accepted={check['accepted']} | "
                  f"kind={check.get('kind', '')} | cells={len(check.get('cells', []))} "
                  f"| reason={check.get('reason', '')}")
        print(f"Method: {args.method} | model: {args.model} | status: {result['status']}")
        print(f"Repair attempts: {result['repairs']}")
        for index, audit in enumerate(result.get("audits", []), start=1):
            print(f"  Audit {index}: {audit['audit_status']} | "
                  f"counts={audit['counts']}")
        for index, table in enumerate(result.get("table_checks", []), start=1):
            print(f"  Table check {index}: checked_cells={table['checked_cells']} | "
                  f"conflicts={len(table['issues'])}")
            for issue in table['issues']:
                print("    - " + issue['message'])
        for index, coverage in enumerate(result.get("coverage_checks", []), start=1):
            print(f"  Coverage {index}: {coverage['verdict']} "
                  f"| missing={coverage['missing_aspects']}")
        print("\n" + result["report"])
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation protects prior experiment outputs.
            with args.output.open("x", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)
            print(f"\nReport saved to: {args.output}")
    except (RuntimeError, ValueError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
