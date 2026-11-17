
from pathlib import Path
from tech_specs.structure import build_page_elements
from tech_specs.sections import annotate_sections

import pymupdf
import re

def is_footer_table(table, page) -> bool:
    """
    Detect likely footer elements incorrectly classified
    as tables.
    """

    markdown = table.to_markdown()

    # Footer-like table must be near the bottom
    near_bottom = (
        table.bbox[1] > page.rect.height * 0.85
    )

    # Suspiciously small structure
    single_row = table.row_count == 1

    # Typical page number or ST document revision
    footer_pattern = re.search(
        r"\b\d+\s*/\s*\d+\b|DS\d+\s+Rev\s*\d+",
        markdown,
        re.IGNORECASE,
    )

    return (
        near_bottom
        and single_row
        and footer_pattern is not None
    )
    
def extract_page(
    pdf_path: Path,
    page_number: int,
    active_section: dict | None = None,
) -> dict:
    

    """
    Extract text, blocks, tables and typography metadata
    from a PDF page.

    """

    pdf_path = Path(pdf_path)

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    with pymupdf.open(pdf_path) as document:

        if not 1 <= page_number <= document.page_count:
            raise ValueError(
                f"Page must be between 1 and {document.page_count}"
            )

        # PyMuPDF indexes pages from zero
        page = document[page_number - 1]

        # 1. Plain text: original and position-sorted
        raw_text = page.get_text("text")
        sorted_text = page.get_text("text", sort=True)

        # 2. Extract text blocks and their coordinates
        text_blocks = []

        for block in page.get_text("blocks", sort=True):
            x0, y0, x1, y1, text, block_no, block_type = block

            # Ignore non-text blocks
            if block_type != 0 or not text.strip():
                continue

            text_blocks.append({
                "block_id": block_no,
                "bbox": [
                    round(x0, 2),
                    round(y0, 2),
                    round(x1, 2),
                    round(y1, 2),
                ],
                "text": text.strip(),
            })

        # 3. Detect and extract tables
        tables = []
        
        detected = page.find_tables()

        for index, table in enumerate(detected.tables):

            if is_footer_table(table, page):
                continue

            tables.append({
                "table_id": len(tables) + 1,
                "bbox": [
                    round(value, 2)
                    for value in table.bbox
                ],
                "rows": table.row_count,
                "columns": table.col_count,
                "markdown": table.to_markdown(),
            })

        # 4. Inspect typography for future heading detection
        typography = []

        page_dict = page.get_text("dict")

        for block in page_dict["blocks"]:
            if block["type"] != 0:
                continue

            for line in block["lines"]:
                for span in line["spans"]:

                    text = span["text"].strip()

                    if not text:
                        continue

                    typography.append({
                        "text": text[:120],
                        "font": span["font"],
                        "size": round(span["size"], 2),
                    })
                    
        # 5. Build structured elements            
        elements = build_page_elements(
            page=page,
            document_name=pdf_path.stem,
            page_number=page_number,
            tables=tables,
        )
        
        
        # 6. Detect and annotate sections
        elements, last_section = annotate_sections(
             elements=elements,
            active_section=active_section,
        )
        
        # 7. Verify that extracted tables reached the elements list
        structured_table_count = sum(
            element["content_type"] == "table"
            for element in elements
        )

        if structured_table_count != len(tables):
            raise RuntimeError(
                f"Table count mismatch: extracted={len(tables)}, "
                f"structured={structured_table_count}"
            )
        
        return {
            "document": pdf_path.stem,
            "page": page_number,
            "raw_text": raw_text,
            "sorted_text": sorted_text,
            "text_blocks": text_blocks,
            "tables": tables,
            "typography_sample": typography[:40],
            "elements": elements,
            "last_section": last_section,
        }
