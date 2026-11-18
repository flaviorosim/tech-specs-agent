"""Build a JSONL retrieval corpus from the downloaded technical PDFs."""

import argparse
import hashlib
import json
from pathlib import Path

import pymupdf

from tech_specs.chunking import chunk_page_elements
from tech_specs.corpus import prepare_chunk
from tech_specs.ingestion import extract_page


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
DOCUMENTS = ("stm32l432kc", "attiny85")
COUNT_KEYS = (
    "chunks_written",
    "text_chunks",
    "table_chunks",
    "without_section",
    "oversized_chunks",
    "review_tables_excluded",
    "empty_chunks_excluded",
    "possible_duplicates",
    "header_chunks_cleaned",
)


def new_counts() -> dict:
    return {key: 0 for key in COUNT_KEYS}


def build_corpus(
    document_names: list[str],
    output_path: Path,
    stats_path: Path,
    max_chars: int = 1200,
    max_pages: int | None = None,
    include_review_tables: bool = False,
) -> dict:
    """Stream chunks to JSONL and write a quality summary."""

    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")
    if max_pages is not None and max_pages < 1:
        raise ValueError("max_pages must be positive")

    # Fail early if any requested input is missing.
    paths = {name: RAW_DIR / f"{name}.pdf" for name in document_names}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing PDF for {name}: {path}. "
                "Run: python scripts/download_docs.py"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_path.with_suffix(output_path.suffix + ".tmp")

    report = {
        "settings": {
            "max_chars": max_chars,
            "max_pages_per_document": max_pages,
            "include_review_tables": include_review_tables,
        },
        "documents": {},
    }

    try:
        with temp_output.open("w", encoding="utf-8", newline="\n") as writer:
            for name, pdf_path in paths.items():
                # Only get page_count here. extract_page() currently opens
                # the PDF itself for each requested page.
                with pymupdf.open(pdf_path) as pdf:
                    total_pages = pdf.page_count

                pages_to_process = min(total_pages, max_pages or total_pages)
                counts = new_counts()
                sections = set()
                seen_fingerprints = set()
                active_section = None  # Reset when switching documents.

                print(f"\n{name}: processing {pages_to_process}/{total_pages} pages")

                for page_number in range(1, pages_to_process + 1):
                    try:
                        result = extract_page(
                            pdf_path,
                            page_number=page_number,
                            active_section=active_section,
                        )
                        # Carry the last section to the next PDF page.
                        active_section = result["last_section"]
                        chunks = chunk_page_elements(
                            result["elements"],
                            max_chars=max_chars,
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            f"Failed on {name}, PDF page {page_number}"
                        ) from exc

                    for raw_chunk in chunks:
                        if (
                            raw_chunk["content_type"] == "table"
                            and raw_chunk.get("metadata", {}).get(
                                "review_required", False
                            )
                            and not include_review_tables
                        ):
                            counts["review_tables_excluded"] += 1
                            continue

                        chunk = prepare_chunk(
                            raw_chunk,
                            include_review_tables=include_review_tables,
                        )
                        if chunk is None:
                            counts["empty_chunks_excluded"] += 1
                            continue

                        if chunk["metadata"].get("header_removed"):
                            counts["header_chunks_cleaned"] += 1

                        # Report possible duplicates, but don't delete them:
                        # identical technical data can legitimately recur.
                        fingerprint = hashlib.sha256(
                            json.dumps(
                                [chunk.get("section"), chunk["content"]],
                                ensure_ascii=False,
                            ).encode("utf-8")
                        ).hexdigest()
                        if fingerprint in seen_fingerprints:
                            counts["possible_duplicates"] += 1
                        seen_fingerprints.add(fingerprint)

                        writer.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                        counts["chunks_written"] += 1
                        counts[f"{chunk['content_type']}_chunks"] += 1
                        counts["without_section"] += chunk.get("section") is None
                        counts["oversized_chunks"] += bool(
                            chunk.get("metadata", {}).get("oversized", False)
                        )
                        if chunk.get("section"):
                            sections.add(chunk["section"])

                    if page_number % 25 == 0 or page_number == pages_to_process:
                        print(
                            f"  Page {page_number}/{pages_to_process}: "
                            f"{counts['chunks_written']} chunks written"
                        )

                report["documents"][name] = {
                    "pdf_pages": total_pages,
                    "pages_processed": pages_to_process,
                    "distinct_sections": len(sections),
                    **counts,
                }

        # Replace the corpus only after all documents were processed.
        temp_output.replace(output_path)
    except Exception:
        temp_output.unlink(missing_ok=True)
        raise

    report["totals"] = {
        "pages_processed": sum(
            item["pages_processed"] for item in report["documents"].values()
        ),
        **{
            key: sum(item[key] for item in report["documents"].values())
            for key in COUNT_KEYS
        },
    }
    stats_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nCorpus: {output_path}")
    print(f"Statistics: {stats_path}")
    print(f"Total chunks: {report['totals']['chunks_written']}")
    print(f"Chunks without section: {report['totals']['without_section']}")
    print(f"Review tables excluded: {report['totals']['review_tables_excluded']}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--documents", nargs="+", choices=DOCUMENTS, default=list(DOCUMENTS)
    )
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--include-review-tables", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=PROCESSED_DIR / "chunks.jsonl"
    )
    parser.add_argument(
        "--stats-output", type=Path,
        default=PROCESSED_DIR / "corpus_stats.json",
    )
    args = parser.parse_args()

    build_corpus(
        document_names=args.documents,
        output_path=args.output,
        stats_path=args.stats_output,
        max_chars=args.max_chars,
        max_pages=args.max_pages,
        include_review_tables=args.include_review_tables,
    )


if __name__ == "__main__":
    main()