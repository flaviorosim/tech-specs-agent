"""Build the optional Qdrant/FastEmbed index for an imported datasheet."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tech_specs.custom_docs import index_custom_vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('document_id', help='ID printed by import_datasheet.py')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild a broken existing index while the app is closed')
    args = parser.parse_args()
    print(index_custom_vectors(ROOT, args.document_id, rebuild=args.rebuild))

if __name__ == '__main__':
    main()
