"""Local, per-document PDF ingestion; does not touch the original benchmark corpus.

Text PDFs become immediately BM25-searchable.  Semantic indexes are optional and
live in each document folder.  No OCR and no datasheet-specific parsing rules.
"""
from __future__ import annotations

import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import pymupdf

DOC_DIR = Path("data/user_docs")
MAX_BYTES = 30 * 1024 * 1024
MAX_PAGES = 800
SCHEMA_VERSION = 1


def library_dir(root: str | Path) -> Path:
    return Path(root).resolve() / DOC_DIR


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", Path(name).stem.lower()).strip("-")[:34]
    return slug or "datasheet"


def _split_text(text: str, *, max_chars: int = 1100, overlap: int = 120) -> list[str]:
    """Create capped overlapping spans; never silently discard long paragraphs."""
    if max_chars < 250 or not 0 <= overlap < max_chars // 2:
        raise ValueError("Invalid chunk size or overlap")
    text = text.replace("\x00", "").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    result: list[str] = []
    offset = 0
    while offset < len(text):
        end = min(offset + max_chars, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n", offset + max_chars // 2, end),
                           text.rfind(". ", offset + max_chars // 2, end))
            if boundary > offset:
                end = boundary + 1
        chunk = text[offset:end].strip()
        if chunk:
            result.append(chunk)
        if end >= len(text):
            break
        offset = max(offset + 1, end - overlap)
    return result


def _table_markdown(table: Any) -> str:
    """Extract best-effort Markdown while page object remains alive."""
    if hasattr(table, "to_markdown"):
        value = table.to_markdown()
        if isinstance(value, str):
            return value.strip()
    return ""


def extract_chunks(pdf_path: str | Path, document_id: str, *,
                   max_chars: int = 1100, max_pages: int = MAX_PAGES) -> tuple[list[dict], dict]:
    """Extract text and optional table spans. Page numbers are 1-based PDF pages."""
    chunks: list[dict] = []
    text_pages = 0
    tables_count = 0
    warnings: list[str] = []
    with pymupdf.open(str(pdf_path)) as pdf:
        if pdf.needs_pass:
            raise ValueError("Password-protected PDFs are not supported")
        if pdf.page_count < 1 or pdf.page_count > max_pages:
            raise ValueError(f"PDF must have between 1 and {max_pages} pages")
        pages = pdf.page_count
        for index, page in enumerate(pdf):
            page_num = index + 1
            raw = page.get_text("text", sort=True)
            if raw.strip():
                text_pages += 1
            parts = _split_text(raw, max_chars=max_chars)
            for part_index, part in enumerate(parts, 1):
                chunks.append({
                    "chunk_id": f"{document_id}_p{page_num:04d}_t{part_index:03d}",
                    "document": document_id, "page": page_num,
                    "section": f"PDF page {page_num}",
                    "content_type": "text", "content": part,
                    "metadata": {"extractor": "pymupdf_sorted_text", "generic": True},
                })
            # Tables are a *secondary*, best-effort index. Extracted text above is
            # intentionally retained: it is a fallback if cell recognition fails.
            try:
                found_tables = page.find_tables()
                for table_index, table in enumerate(found_tables.tables, 1):
                    markdown = _table_markdown(table)
                    if not markdown or "|" not in markdown:
                        continue
                    for part_index, part in enumerate(_split_text(markdown, max_chars=max_chars, overlap=0), 1):
                        chunks.append({
                            "chunk_id": f"{document_id}_p{page_num:04d}_tb{table_index:03d}_{part_index:03d}",
                            "document": document_id, "page": page_num,
                            "section": f"PDF page {page_num} · auto-detected table",
                            "content_type": "table", "content": part,
                            "metadata": {"extractor": "pymupdf_find_tables", "generic": True,
                                         "unverified_table": True},
                        })
                        tables_count += 1
            except (AttributeError, ValueError, RuntimeError) as exc:
                warnings.append(f"Table extraction on PDF page {page_num}: {type(exc).__name__}")
    if not chunks:
        raise ValueError("No searchable text found. Scanned PDFs require OCR, which this importer does not perform.")
    if text_pages < pages * 0.5:
        warnings.append("Many pages have no extractable text; scanned pages may be missing.")
    if tables_count == 0:
        warnings.append("No structured tables extracted; tables remain searchable as PDF text where present.")
    stats = {"pages": pages, "text_pages": text_pages, "chunks": len(chunks),
             "table_chunks": tables_count, "warnings": sorted(set(warnings))[:30]}
    return chunks, stats


def list_custom_documents(root: str | Path) -> list[dict]:
    """Read only valid, existing manifests from dedicated per-document folders."""
    base = library_dir(root)
    documents = []
    if not base.is_dir():
        return documents
    for folder in sorted(base.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        manifest = folder / "manifest.json"
        if not manifest.is_file():
            continue
        try:
            info = json.loads(manifest.read_text(encoding="utf-8"))
            if info.get("schema_version") != SCHEMA_VERSION or info.get("id") != folder.name:
                continue
            if not (folder / "chunks.jsonl").is_file() or not (folder / "source.pdf").is_file():
                continue
            info["vector_indexed"] = (folder / "vector_index_manifest.json").is_file()
            documents.append(info)
        except (ValueError, OSError, TypeError):
            continue
    return documents


def get_custom_document(root: str | Path, document_id: str) -> dict:
    for item in list_custom_documents(root):
        if item["id"] == document_id:
            return item
    raise ValueError(f"Unknown imported datasheet: {document_id}")


def import_pdf_bytes(root: str | Path, filename: str, data: bytes, *,
                     display_name: str | None = None) -> dict:
    """Import PDF atomically under an immutable content-derived document ID."""
    if not isinstance(filename, str) or not filename.casefold().endswith(".pdf"):
        raise ValueError("Only .pdf files are supported")
    if not isinstance(data, bytes) or not 20 <= len(data) <= MAX_BYTES:
        raise ValueError("Invalid PDF or file exceeds the 30 MB upload limit")
    if not data.lstrip().startswith(b"%PDF-"):
        raise ValueError("The uploaded file does not have a PDF header")
    digest = hashlib.sha256(data).hexdigest()
    doc_id = f"user-{_slug(filename)}-{digest[:10]}"
    base = library_dir(root)
    base.mkdir(parents=True, exist_ok=True)
    dest = base / doc_id
    if dest.exists():
        result = get_custom_document(root, doc_id)
        if result["sha256"] != digest:
            raise ValueError("Document ID collision")
        return result
    # Temp folder is always inside destination filesystem for atomic rename.
    with tempfile.TemporaryDirectory(prefix=".import-", dir=base) as tmp:
        stage = Path(tmp)
        (stage / "source.pdf").write_bytes(data)
        chunks, stats = extract_chunks(stage / "source.pdf", doc_id)
        with (stage / "chunks.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for chunk in chunks:
                stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        title = (display_name or Path(filename).stem).strip()[:100]
        info = {"schema_version": SCHEMA_VERSION, "id": doc_id,
                "title": title or doc_id, "filename": Path(filename).name[:150],
                "sha256": digest, **stats}
        (stage / "manifest.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        # atomic on same volume; duplicate upload may race, handled explicitly
        try:
            stage.rename(dest)
        except FileExistsError:
            return get_custom_document(root, doc_id)
    info["vector_indexed"] = False
    return info


def index_custom_vectors(root: str | Path, document_id: str, *, rebuild: bool = False,
                         batch_size: int = 32) -> dict:
    """Build a separate Qdrant Local collection; never touch original Qdrant."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    info = get_custom_document(root, document_id)
    folder = library_dir(root) / document_id
    corpus = folder / "chunks.jsonl"
    manifest_path = folder / "vector_index_manifest.json"
    from fastembed import TextEmbedding
    from qdrant_client import QdrantClient, models
    from tech_specs.retrieval import load_chunks
    from tech_specs.vector_retrieval import (
        COLLECTION_NAME, MODEL_NAME, VECTOR_SIZE, corpus_sha256,
        embedding_text, point_id, verify_index,
    )
    chunks = load_chunks(corpus)
    db_path = folder / "qdrant_local"
    client = QdrantClient(path=str(db_path))
    try:
        if manifest_path.is_file() and not rebuild:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            verify_index(client, manifest, corpus_path=corpus)
            return {"indexed": True, "chunks": len(chunks), "skipped": True}
        if client.collection_exists(COLLECTION_NAME):
            if not rebuild:
                raise ValueError("Vector collection already exists without a valid manifest. Use rebuild=True when closed.")
            client.delete_collection(COLLECTION_NAME)
        client.create_collection(collection_name=COLLECTION_NAME,
                                 vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE))
        model = TextEmbedding(model_name=MODEL_NAME)
        for offset in range(0, len(chunks), batch_size):
            batch = chunks[offset:offset + batch_size]
            vectors = list(model.passage_embed([embedding_text(chunk) for chunk in batch]))
            if len(vectors) != len(batch):
                raise ValueError("Embedding count mismatch")
            client.upsert(collection_name=COLLECTION_NAME,
                points=[models.PointStruct(id=point_id(c["chunk_id"]), vector=v.tolist(),
                                           payload={"document": document_id, "chunk": c})
                        for c, v in zip(batch, vectors)], wait=True)
        if client.count(COLLECTION_NAME, exact=True).count != len(chunks):
            raise ValueError("Vector index count mismatch")
        manifest = {"model_name": MODEL_NAME, "collection_name": COLLECTION_NAME,
                    "vector_size": VECTOR_SIZE, "corpus_sha256": corpus_sha256(corpus),
                    "chunks_indexed": len(chunks), "partial_index": False}
        manifest_path.with_suffix(".json.tmp").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest_path.with_suffix(".json.tmp").replace(manifest_path)
        return {"indexed": True, "chunks": len(chunks), "skipped": False}
    finally:
        client.close()
