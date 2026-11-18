"""Search technical PDF chunks with a BM25 lexical baseline."""

import argparse
from pathlib import Path

from tech_specs.retrieval import BM25Retriever, load_chunks

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT_DIR / "data" / "processed" / "chunks.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help='Search question, e.g. "TIM2 counter resolution"')
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--document", choices=("stm32l432kc", "attiny85"))
    parser.add_argument("--full", action="store_true", help="Show the full chunk")
    args = parser.parse_args()

    chunks = load_chunks(args.corpus)
    retriever = BM25Retriever(chunks)
    results = retriever.search(
        args.query, top_k=args.top_k, document=args.document
    )

    print(f"Indexed chunks: {len(chunks)}")
    print(f"Query: {args.query!r}")
    print(f"Results: {len(results)}")

    if not results:
        print("No positive-score BM25 matches found.")
        return

    for hit in results:
        chunk = hit["chunk"]
        content = chunk["content"]
        preview = content if args.full else content[:550]
        if not args.full and len(content) > 550:
            preview += "..."
        print("\n" + "=" * 65)
        print(f"#{hit['rank']} | BM25 score: {hit['score']:.4f}")
        print(f"Chunk: {chunk['chunk_id']} | Type: {chunk['content_type']}")
        print(f"Document: {chunk['document']} | PDF page: {chunk['page']}")
        print(f"Section: {chunk.get('section') or '(unassigned)'}")
        print(f"Review required: {chunk.get('metadata', {}).get('review_required', False)}")
        print("-" * 65)
        print(preview)


if __name__ == "__main__":
    main()
