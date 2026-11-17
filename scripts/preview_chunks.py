
import argparse
import json
from pathlib import Path

from tech_specs.ingestion import extract_page
from tech_specs.chunking import chunk_page_elements


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "chunk_previews"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("document")
    parser.add_argument("--page", type=int, required=True)
    parser.add_argument("--max-chars", type=int, default=1200)

    args = parser.parse_args()

    pdf_path = RAW_DIR / f"{args.document}.pdf"

    result = extract_page(
        pdf_path,
        page_number=args.page,
    )

    chunks = chunk_page_elements(
        result["elements"],
        max_chars=args.max_chars,
    )

    print(f"\nDocument: {args.document}")
    print(f"Page: {args.page}")
    print(f"Elements: {len(result['elements'])}")
    print(f"Chunks: {len(chunks)}")

    for chunk in chunks:
        print("\n" + "=" * 65)
        print(f"ID: {chunk['chunk_id']}")
        print(f"Type: {chunk['content_type']}")
        print(f"Section: {chunk['section']}")
        print(f"Length: {len(chunk['content'])}")
        print(f"Oversized: {chunk['metadata']['oversized']}")
        print(f"Source elements: {chunk['element_ids']}")
        print("-" * 65)
        print(chunk["content"][:500])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        OUTPUT_DIR /
        f"{args.document}_p{args.page:03d}_chunks.json"
    )

    output_path.write_text(
        json.dumps(
            chunks,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
