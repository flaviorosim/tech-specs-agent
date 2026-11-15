
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pymupdf
import requests

# Project root, independent of the current working directory
ROOT_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT_DIR / "data" / "raw"
MANIFEST_PATH = ROOT_DIR / "data" / "manifest.json"

PDF_SOURCES = {
    "stm32l432kc": (
        "https://www.st.com/resource/en/datasheet/stm32l432kc.pdf"
    ),
    "attiny85": (
        "https://ww1.microchip.com/downloads/en/DeviceDoc/"
        "atmel-2586-avr-8-bit-microcontroller-attiny25-"
        "attiny45-attiny85_datasheet.pdf"
    ),
}

def inspect_pdf(path: Path) -> dict:
    """Validate a PDF and extract basic file metadata."""

    with path.open("rb") as file:
        if file.read(5) != b"%PDF-":
            raise ValueError("File does not have a PDF signature")

    # Verify the file can be opened and read
    try:
        with pymupdf.open(path) as document:
            pages = document.page_count

        if pages == 0:
            raise ValueError("PDF contains no pages")

    except Exception as exc:
        raise ValueError(f"Invalid PDF: {exc}") from exc

    # Calculate SHA-256 incrementally
    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(65536), b""):
            sha256.update(block)

    return {
        "pages": pages,
        "size_bytes": path.stat().st_size,
        "sha256": sha256.hexdigest(),
    }

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Preserve metadata from previous executions
    if MANIFEST_PATH.exists():
        manifest = json.loads(
            MANIFEST_PATH.read_text(encoding="utf-8")
        )
    else:
        manifest = {}

    for name, url in PDF_SOURCES.items():
        destination = RAW_DIR / f"{name}.pdf"
        downloaded = False

        if destination.exists():
            try:
                metadata = inspect_pdf(destination)
                print(f"[SKIP] {name}: valid PDF already exists")

            except ValueError:
                print(f"[WARN] {name}: invalid local file")
                destination.unlink()

        if not destination.exists():
            print(f"[FETCH] Downloading {name}...")

            temporary = destination.with_suffix(".pdf.part")

            try:
                with requests.get(
                    url,
                    stream=True,
                    timeout=(10, 90),
                ) as response:
                    response.raise_for_status()

                    with temporary.open("wb") as file:
                        for chunk in response.iter_content(65536):
                            if chunk:
                                file.write(chunk)

                metadata = inspect_pdf(temporary)
                temporary.replace(destination)
                downloaded = True

                print(f"[OK] {name}: {metadata['pages']} pages")

            except (requests.RequestException, OSError, ValueError) as exc:
                temporary.unlink(missing_ok=True)
                print(f"[ERROR] {name}: {exc}")
                continue

        entry = {
            "filename": destination.name,
            "source_url": url,
            **metadata,
        }

        if downloaded:
            entry["downloaded_at_utc"] = (
                datetime.now(timezone.utc).isoformat()
            )
        elif name in manifest:
            if "downloaded_at_utc" in manifest[name]:
                entry["downloaded_at_utc"] = (
                    manifest[name]["downloaded_at_utc"]
                )

        manifest[name] = entry

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("[DONE] Document collection finished")

if __name__ == "__main__":
    main()