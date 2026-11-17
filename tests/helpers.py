"""Test helpers: run async DB code from sync tests, and build sample files in memory."""

import asyncio
import io
from collections.abc import Awaitable, Callable

from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import get_settings


def run_db[T](fn: Callable[[AsyncConnection], Awaitable[T]]) -> T:
    """Run `fn(conn)` on a fresh connection to the test database."""

    async def runner() -> T:
        engine = create_async_engine(get_settings().database_url, poolclass=NullPool)
        try:
            async with engine.connect() as conn:
                return await fn(conn)
        finally:
            await engine.dispose()

    return asyncio.run(runner())


def make_pdf(pages: list[str], title: str | None = None) -> bytes:
    """A minimal valid PDF with one text line per input line (Helvetica, 12 pt)."""
    page_ids = [5 + 2 * i for i in range(len(pages))]
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: (
            f"<< /Type /Pages /Kids [{' '.join(f'{p} 0 R' for p in page_ids)}] "
            f"/Count {len(pages)} >>"
        ).encode(),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        4: f"<< /Title ({title or ''}) >>".encode(),
    }
    for page_id, page_text in zip(page_ids, pages, strict=True):
        ops = ["BT", "/F1 12 Tf", "14 TL", "72 740 Td"]
        for line in page_text.split("\n"):
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.append(f"({escaped}) Tj T*")
        ops.append("ET")
        stream = "\n".join(ops).encode()
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_id + 1} 0 R >>"
        ).encode()
        objects[page_id + 1] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"
        )

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for obj_id in sorted(objects):
        offsets[obj_id] = len(out)
        out += f"{obj_id} 0 obj\n".encode() + objects[obj_id] + b"\nendobj\n"
    xref_at = len(out)
    size = max(objects) + 1
    out += f"xref\n0 {size}\n0000000000 65535 f \n".encode()
    for obj_id in range(1, size):
        out += f"{offsets[obj_id]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {size} /Root 1 0 R /Info 4 0 R >>\n".encode()
    out += f"startxref\n{xref_at}\n%%EOF\n".encode()
    return bytes(out)


def make_docx() -> bytes:
    from docx import Document

    doc = Document()
    doc.core_properties.title = "Travel Policy"
    doc.add_heading("Travel Policy", level=1)
    doc.add_paragraph("This policy applies to all employees.")
    doc.add_heading("Hotels", level=2)
    doc.add_paragraph("Hotel stays are capped at 150 EUR per night.")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "City"
    table.cell(0, 1).text = "Cap"
    table.cell(1, 0).text = "London"
    table.cell(1, 1).text = "200 EUR"
    doc.add_heading("Flights", level=2)
    doc.add_paragraph("Economy class for flights under 6 hours.")
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


SAMPLE_MARKDOWN = b"""# Employee Handbook

Welcome to the company.

## Leave

### Annual leave

Employees get 25 days of annual leave per year.

### Parental leave

Parents get 16 weeks of paid leave.

```python
# Not a heading: inside a code block
days = 25
```
"""
