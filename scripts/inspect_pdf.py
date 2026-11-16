
import argparse
import json
from pathlib import Path

from tech_specs.ingestion import extract_page


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "inspections"


def main():

    parser = argparse.ArgumentParser(
        description="Inspect PDF parsing results"
    )

    parser.add_argument(
        "document",
        choices=["stm32l432kc", "attiny85"],
        help="Document name without .pdf",
    )

    parser.add_argument(
        "--page",
        type=int,
        default=1,
        help="Page number (1-based)",
    )

    args = parser.parse_args()

    pdf_path = RAW_DIR / f"{args.document}.pdf"

    result = extract_page(pdf_path, args.page)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"{args.document}_p{args.page:03d}"

    # Save complete structured extraction
    json_path = OUTPUT_DIR / f"{filename}.json"

    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Save text versions for easy comparison
    raw_path = OUTPUT_DIR / f"{filename}_raw.txt"
    sorted_path = OUTPUT_DIR / f"{filename}_sorted.txt"

    raw_path.write_text(
        result["raw_text"],
        encoding="utf-8",
    )

    sorted_path.write_text(
        result["sorted_text"],
        encoding="utf-8",
    )

    # Display extraction statistics
    print(f"\nDocument: {result['document']}")
    print(f"Page: {result['page']}")
    print(f"Text blocks: {len(result['text_blocks'])}")
    print(f"Detected tables: {len(result['tables'])}")

    for table in result["tables"]:
        print(
            f"\nTable {table['table_id']}: "
            f"{table['rows']} rows, "
            f"{table['columns']} columns"
        )
        print(table["markdown"][:600])

    print(f"\nSaved inspection to: {json_path}")


if __name__ == "__main__":
    main()
