"""Run the 18 v3 development questions through the existing local LangGraph.

No reference answer, gold IDs or holdout questions are supplied to the graph.
Results are resumable. Each question uses local Ollama; no paid API.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from tech_specs.answer_benchmark import ensure_run_config, load_dev_set, run_batch
from tech_specs.answer_graph import build_answer_graph
from tech_specs.local_answer import LocalOllama
from tech_specs.retrieval import BM25Retriever, load_chunks

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=ROOT / "evaluation/questions_v3_dev.json")
    parser.add_argument("--answer-key", type=Path, default=ROOT / "evaluation/benchmark_v3_answer_key.json")
    parser.add_argument("--corpus", type=Path, default=PROCESSED / "chunks.jsonl")
    parser.add_argument("--method", choices=("bm25", "vector", "hybrid"), default="bm25")
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--limit", type=int, help="Run first N questions for a smoke test; rerun to resume")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--candidates-per-facet", type=int, default=10)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--num-predict", type=int, default=800)
    parser.add_argument("--no-audit", action="store_true", help="Skip local second-pass judge")
    parser.add_argument("--output-dir", type=Path, help="Default: data/processed/agent_v3_dev_<method>_<model>")
    parser.add_argument("--db-path", type=Path, default=PROCESSED / "qdrant_local")
    parser.add_argument("--manifest", type=Path, default=PROCESSED / "vector_index_manifest.json")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.top_k < 1 or args.candidates_per_facet < 1:
        parser.error("--top-k and --candidates-per-facet must be positive")

    questions = load_dev_set(args.questions, args.answer_key)
    dirname = "agent_v3_dev_" + args.method + "_" + args.model.replace(":", "-").replace("/", "-")
    out = args.output_dir or PROCESSED / dirname
    print(f"Development questions: {len(questions)} | method={args.method} | model={args.model}")
    print(f"Output: {out} | judge={'disabled' if args.no_audit else 'enabled'}")

    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    ensure_run_config(out, {
        "protocol_version": 1,
        "method": args.method,
        "model": args.model,
        "top_k": args.top_k,
        "candidates_per_facet": args.candidates_per_facet,
        "num_ctx": args.num_ctx,
        "num_predict": args.num_predict,
        "audit_enabled": not args.no_audit,
        "corpus_sha256": sha256_file(args.corpus),
        "questions_sha256": sha256_file(args.questions),
        "answer_key_sha256": sha256_file(args.answer_key),
    })

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
                raise ValueError("Cannot search partial vector index")
            client = QdrantClient(path=str(args.db_path))
            verify_index(client, manifest, corpus_path=args.corpus)
            vectors = VectorRetriever(client, TextEmbedding(model_name=MODEL_NAME))
            if args.method == "vector":
                retriever = vectors
            else:
                from tech_specs.hybrid_retrieval import HybridRetriever
                retriever = HybridRetriever(bm25, vectors)

        llm = LocalOllama(model=args.model, num_ctx=args.num_ctx, num_predict=args.num_predict)
        graph = build_answer_graph(retriever, llm, top_k=args.top_k,
                                   candidates_per_facet=args.candidates_per_facet)
        judge = None
        audit_fn = None
        if not args.no_audit:
            from tech_specs.faithfulness import OllamaEvidenceJudge, audit_report
            judge = OllamaEvidenceJudge(model=args.model, num_ctx=args.num_ctx)
            audit_fn = audit_report

        def invoke(question: str, document: str) -> dict:
            return graph.invoke({"question": question, "document_hint": document})

        rows = run_batch(questions, invoke, out, judge=judge, audit_fn=audit_fn, limit=args.limit)
        valid = sum(r["status"] == "citations_checked" for r in rows)
        judged = sum(r["audit_status"] == "all_supported_by_local_judge" for r in rows)
        errors = sum(bool(r["error"]) for r in rows)
        print(f"\nProcessed: {len(rows)}/{len(questions)} | citations_checked: {valid} | "
              f"all_supported_by_local_judge: {judged} | errors: {errors}")
        print(f"Manual review file: {out / 'review.csv'}")
        print("Note: source IDs/anchors and a local judge do not establish factual accuracy or completeness.")
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
