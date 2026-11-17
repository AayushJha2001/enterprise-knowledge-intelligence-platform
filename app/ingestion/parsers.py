"""Turn uploaded files into an ordered list of text blocks that know where they came from.

Every block carries its page number (PDF) and its heading path (DOCX, Markdown, HTML), so
chunks can be cited as "Employee Handbook, p. 12" or "Handbook > Leave > Parental leave".
"""

import io
import re
from dataclasses import dataclass, field
from pathlib import PurePath

from bs4 import BeautifulSoup, Tag
from docx import Document as load_docx
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader
from pypdf.errors import PdfReadError


class ParseError(Exception):
    """The file can't be turned into text. Retrying won't help."""


@dataclass(frozen=True)
class Block:
    text: str
    page: int | None = None
    section_path: tuple[str, ...] = ()


@dataclass
class ParsedDocument:
    blocks: list[Block]
    title: str | None = None
    page_count: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)


# Supported upload types, keyed by file extension.
MIME_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".html": "text/html",
    ".htm": "text/html",
    ".txt": "text/plain",
}

# Magic bytes for binary formats: rejects e.g. an .exe renamed to .pdf at upload time.
_SIGNATURES: dict[str, bytes] = {
    "application/pdf": b"%PDF",
    MIME_TYPES[".docx"]: b"PK",  # DOCX is a zip archive
}

_BLANK_LINES = re.compile(r"\n\s*\n")
_WHITESPACE = re.compile(r"[ \t\r\f\v]+")


def detect_mime_type(filename: str, head: bytes) -> str | None:
    """Return the MIME type for a supported upload, or None if unsupported or mismatched."""
    mime = MIME_TYPES.get(PurePath(filename).suffix.lower())
    if mime is None:
        return None
    signature = _SIGNATURES.get(mime)
    if signature is not None and not head.startswith(signature):
        return None
    return mime


def parse_document(data: bytes, mime_type: str) -> ParsedDocument:
    parser = _PARSERS.get(mime_type)
    if parser is None:
        raise ParseError(f"Unsupported type: {mime_type}")
    return parser(data)


# ---------------------------------------------------------------- helpers


def _clean(text: str) -> str:
    lines = (_WHITESPACE.sub(" ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _paragraphs(text: str) -> list[str]:
    return [p for p in (_clean(part) for part in _BLANK_LINES.split(text)) if p]


class _HeadingStack:
    """Tracks the current heading path, e.g. ("Handbook", "Leave", "Parental leave")."""

    def __init__(self) -> None:
        self._stack: list[tuple[int, str]] = []

    def push(self, level: int, title: str) -> None:
        while self._stack and self._stack[-1][0] >= level:
            self._stack.pop()
        self._stack.append((level, title))

    @property
    def path(self) -> tuple[str, ...]:
        return tuple(title for _, title in self._stack)


# ---------------------------------------------------------------- PDF


def parse_pdf(data: bytes) -> ParsedDocument:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ParseError("PDF is password-protected")
        blocks = [
            Block(text=paragraph, page=page_number)
            for page_number, page in enumerate(reader.pages, start=1)
            for paragraph in _paragraphs(page.extract_text() or "")
        ]
        meta = reader.metadata
        title = (meta.title or "").strip() if meta else ""
        page_count = len(reader.pages)
    except ParseError:
        raise
    except (PdfReadError, ValueError, KeyError) as exc:
        raise ParseError(f"Could not read PDF: {exc}") from exc

    if not blocks:
        raise ParseError("No text found in PDF (it may be scanned images; OCR is not supported)")
    return ParsedDocument(blocks=blocks, title=title or None, page_count=page_count)


# ---------------------------------------------------------------- DOCX

_HEADING_STYLE = re.compile(r"^Heading (\d)$")


def _docx_body_items(document) -> list[Paragraph | Table]:
    """Paragraphs and tables in reading order (python-docx lists them separately)."""
    items: list[Paragraph | Table] = []
    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            items.append(Paragraph(child, document))
        elif tag == "tbl":
            items.append(Table(child, document))
    return items


def _table_text(table: Table) -> str:
    rows = []
    for row in table.rows:
        cells: list[str] = []
        for cell in row.cells:
            text = _clean(cell.text).replace("\n", " ")
            if not cells or cells[-1] != text:  # merged cells repeat their text
                cells.append(text)
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def parse_docx(data: bytes) -> ParsedDocument:
    try:
        document = load_docx(io.BytesIO(data))
    except Exception as exc:  # python-docx raises several unrelated types for bad files
        raise ParseError(f"Could not read DOCX: {exc}") from exc

    headings = _HeadingStack()
    blocks: list[Block] = []
    title = (document.core_properties.title or "").strip() or None

    for item in _docx_body_items(document):
        if isinstance(item, Table):
            text = _table_text(item)
            if text:
                blocks.append(Block(text=text, section_path=headings.path))
            continue

        text = _clean(item.text)
        if not text:
            continue
        style = item.style.name if item.style is not None else ""
        if style == "Title":
            title = title or text
            continue
        match = _HEADING_STYLE.match(style)
        if match:
            headings.push(int(match.group(1)), text)
            continue
        blocks.append(Block(text=text, section_path=headings.path))

    if not blocks:
        raise ParseError("No text found in DOCX")
    return ParsedDocument(blocks=blocks, title=title)


# ---------------------------------------------------------------- Markdown

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_MD_FENCE = re.compile(r"^\s*(```|~~~)")


def parse_markdown(data: bytes) -> ParsedDocument:
    headings = _HeadingStack()
    blocks: list[Block] = []
    title: str | None = None
    paragraph: list[str] = []
    code: list[str] | None = None  # lines of the open code fence, if any

    def flush() -> None:
        if paragraph:
            text = _clean("\n".join(paragraph))
            if text:
                blocks.append(Block(text=text, section_path=headings.path))
            paragraph.clear()

    for line in _decode(data).splitlines():
        if _MD_FENCE.match(line):
            if code is None:
                flush()
                code = [line]
            else:
                code.append(line)
                # Keep code verbatim: indentation matters.
                blocks.append(Block(text="\n".join(code), section_path=headings.path))
                code = None
            continue
        if code is not None:
            code.append(line)
            continue

        heading = _MD_HEADING.match(line)
        if heading:
            flush()
            level, text = len(heading.group(1)), heading.group(2).strip()
            if level == 1 and title is None:
                title = text
            headings.push(level, text)
        elif line.strip():
            paragraph.append(line)
        else:
            flush()

    flush()
    if code:  # unterminated fence
        blocks.append(Block(text="\n".join(code), section_path=headings.path))
    if not blocks:
        raise ParseError("No text found in Markdown file")
    return ParsedDocument(blocks=blocks, title=title)


# ---------------------------------------------------------------- HTML

_HTML_HEADINGS = {f"h{level}": level for level in range(1, 7)}
_HTML_CONTENT = ("p", "li", "pre", "table", "blockquote", "dd", "dt")
_HTML_NOISE = ("script", "style", "noscript", "nav", "header", "footer", "aside", "form")


_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?)\]])")


def _inline_text(element: Tag) -> str:
    """Text of an element; inline tags (<b>, <a>) are joined without stray spaces."""
    return _SPACE_BEFORE_PUNCT.sub(r"\1", _clean(element.get_text(" ")))


def _html_table_text(table: Tag) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = [_inline_text(c) for c in tr.find_all(["th", "td"])]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def parse_html(data: bytes) -> ParsedDocument:
    soup = BeautifulSoup(_decode(data), "html.parser")
    for noise in soup.find_all(_HTML_NOISE):
        noise.decompose()

    title_tag = soup.find("title")
    title = _clean(title_tag.get_text()) if title_tag else None
    root = soup.find("main") or soup.find("article") or soup.body or soup

    headings = _HeadingStack()
    blocks: list[Block] = []
    wanted = (*_HTML_HEADINGS, *_HTML_CONTENT)
    for element in root.find_all(wanted):
        # Skip elements nested in another wanted element (e.g. <p> inside <li>);
        # the outer element already contributed their text.
        if element.find_parent(wanted) is not None:
            continue
        if element.name in _HTML_HEADINGS:
            text = _inline_text(element)
            if text:
                title = title or text
                headings.push(_HTML_HEADINGS[element.name], text)
            continue
        if element.name == "table":
            text = _html_table_text(element)
        elif element.name == "pre":
            text = element.get_text().strip("\n")
        else:
            text = _inline_text(element)
        if text:
            blocks.append(Block(text=text, section_path=headings.path))

    if not blocks:  # unstructured page: fall back to its visible text
        blocks = [Block(text=p) for p in _paragraphs(root.get_text("\n\n"))]
    if not blocks:
        raise ParseError("No text found in HTML file")
    return ParsedDocument(blocks=blocks, title=title or None)


# ---------------------------------------------------------------- plain text


def parse_text(data: bytes) -> ParsedDocument:
    blocks = [Block(text=p) for p in _paragraphs(_decode(data))]
    if not blocks:
        raise ParseError("File is empty")
    return ParsedDocument(blocks=blocks)


_PARSERS = {
    "application/pdf": parse_pdf,
    MIME_TYPES[".docx"]: parse_docx,
    "text/markdown": parse_markdown,
    "text/html": parse_html,
    "text/plain": parse_text,
}
