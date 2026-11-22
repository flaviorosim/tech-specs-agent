"""Import an additional PDF datasheet without touching original corpus/indexes."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tech_specs.custom_docs import import_pdf_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Path to a locally downloaded PDF datasheet")
    parser.add_argument("--name", help="Display name in the web app")
    args = parser.parse_args()
    info = import_pdf_bytes(ROOT, args.pdf.name, args.pdf.read_bytes(), display_name=args.name)
    print(f"Imported: {info['title']}\nDocument ID: {info['id']}")
    print(f"PDF pages: {info['pages']}; text pages: {info['text_pages']}; "
          f"chunks: {info['chunks']}; table chunks: {info['table_chunks']}")
    for warning in info.get('warnings', []):
        print('WARNING:', warning)
    print('BM25 can be used immediately. Select this document in the Streamlit sidebar.')
    print(f'To build vectors: python scripts/index_custom_vectors.py {info["id"]}')

if __name__ == '__main__':
    main()
