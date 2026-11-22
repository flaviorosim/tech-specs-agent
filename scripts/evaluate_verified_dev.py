"""Evaluate the v5 guarded agent on v3 DEV questions only (never holdout).

Run from the repository root. Requires the project's existing Ollama/Qdrant setup.
The answer key is used only to construct post-run manual-review columns.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from tech_specs.answer_benchmark import ensure_run_config, load_dev_set
from tech_specs.answer_graph import build_answer_graph
from tech_specs.faithfulness import OllamaEvidenceJudge
from tech_specs.local_answer import LocalOllama
from tech_specs.retrieval import BM25Retriever, load_chunks
from tech_specs.review_models import OllamaCoverageJudge, OllamaRepair
from tech_specs.table_answers import TableFirstGenerator
from tech_specs.verified_answer_graph import build_verified_answer_graph
from tech_specs.verified_benchmark import run_verified_batch, summarize

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=ROOT / "evaluation/questions_v3_dev.json")
    parser.add_argument("--answer-key", type=Path, default=ROOT / "evaluation/benchmark_v3_answer_key.json")
    parser.add_argument("--corpus", type=Path, default=PROCESSED / "chunks.jsonl")
    parser.add_argument("--db-path", type=Path, default=PROCESSED / "qdrant_local")
    parser.add_argument("--manifest", type=Path, default=PROCESSED / "vector_index_manifest.json")
    parser.add_argument("--method", choices=("bm25", "vector", "hybrid"), default="bm25")
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--candidates-per-facet", type=int, default=10)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--num-predict", type=int, default=1200)
    parser.add_argument("--no-repair", action="store_true")
    parser.add_argument("--no-table-first", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.top_k < 1 or args.candidates_per_facet < 1 or args.num_ctx < 1 or args.num_predict < 1:
        parser.error("Retrieval depths and model token limits must be positive")

    # Strictly restrict the experiment to the already reviewed dev split.
    questions = load_dev_set(args.questions, args.answer_key)
    if len(questions) != 18:
        parser.error(f"Expected 18 DEV questions; found {len(questions)}. Check split paths.")

    default_name = f"agent_v5_dev_{args.method}_{args.model.replace(':', '-').replace('/', '-')}"
    if args.no_table_first:
        default_name += "_no_table"
    if args.no_repair:
        default_name += "_no_repair"
    out = args.output_dir or PROCESSED / default_name

    # Changing relevant source code or an experimental setting creates a new
    # experimental condition. Fail loudly instead of mixing resumed versions.
    tracked = (
        "auto_planner.py", "facet_retrieval.py", "local_answer.py",
        "answer_graph.py", "review_models.py", "faithfulness.py",
        "table_answers.py", "table_guard.py", "source_cell_check.py",
        "verified_answer_graph.py", "verified_benchmark.py",
    )
    tracked_paths = [ROOT / "src" / "tech_specs" / name for name in tracked]
    tracked_paths += [Path(__file__).resolve(), ROOT / "src" / "tech_specs" / "answer_benchmark.py"]
    missing = [str(p) for p in tracked_paths if not p.is_file()]
    if missing:
        parser.error("Missing project files: " + ", ".join(missing))
    config = {
        "protocol_version": "verified-dev-v5-1",
        "method": args.method,
        "model": args.model,
        "top_k": args.top_k,
        "candidates_per_facet": args.candidates_per_facet,
        "num_ctx": args.num_ctx,
        "num_predict": args.num_predict,
        "max_repairs": 0 if args.no_repair else 1,
        "table_first": not args.no_table_first,
        "corpus_sha256": file_hash(args.corpus),
        "questions_sha256": file_hash(args.questions),
        "answer_key_sha256": file_hash(args.answer_key),
        "code_sha256": {p.name: file_hash(p) for p in tracked_paths},
    }
    if args.method != "bm25":
        config["vector_manifest_sha256"] = file_hash(args.manifest)
    ensure_run_config(out, config)

    print(f"DEV questions: {len(questions)} | method={args.method} | model={args.model}")
    print(f"Output: {out} | max_repairs={config['max_repairs']} | source_tables={config['table_first']}")
    chunks = load_chunks(args.corpus)
    bm25 = BM25Retriever(chunks) if args.method != "vector" else None
    client = None
    try:
        if args.method == "bm25":
            retriever = bm25
        else:
            from fastembed import TextEmbedding
            from qdrant_client import QdrantClient
            from tech_specs.vector_retrieval import MODEL_NAME, VectorRetriever, load_manifest, verify_index
            manifest = load_manifest(args.manifest)
            if manifest.get("partial_index"):
                raise ValueError("Refusing partial vector index")
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
        base = build_answer_graph(retriever, generator, top_k=args.top_k,
                                  candidates_per_facet=args.candidates_per_facet)
        guarded = build_verified_answer_graph(
            base,
            judge=OllamaEvidenceJudge(model=args.model, num_ctx=args.num_ctx),
            coverage_judge=OllamaCoverageJudge(model=args.model, num_ctx=args.num_ctx),
            repairer=OllamaRepair(model=args.model, num_ctx=args.num_ctx,
                                 num_predict=args.num_predict),
            max_repairs=config["max_repairs"], source_cell_route=config["table_first"],
        )

        def invoke(question: str, document: str) -> dict:
            state = guarded.invoke({"question": question, "document_hint": document})
            state["initial_answer_strategy"] = generator.last_strategy
            state["final_answer_strategy"] = (
                "source_table_extraction" if state.get("status") == "source_table_checked"
                else ("local_llm_repair" if state.get("repairs", 0)
                      and (state.get("working_result") or {}).get("status") == "citations_checked"
                      else ("repair_failed_or_abstained" if state.get("repairs", 0)
                            else generator.last_strategy))
            )
            return state

        rows = run_verified_batch(questions, invoke, out, limit=args.limit)
        results = summarize(rows)
        print("\nSummary:")
        for key in ("source_table_checked", "local_checks_passed", "coverage_review_recommended",
                    "withheld", "execution_errors", "repairs_total"):
            print(f"  {key}: {results[key]}")
        print(f"Manual review: {out / 'review.csv'}")
        print("No accuracy/completeness claims: review answer against the reference_answer manually.")
        print("Holdout split not accessed by the graph; do NOT tune against the 12 test questions.")
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
