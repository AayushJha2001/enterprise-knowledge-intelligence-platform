import pytest

from app.ingestion.parsers import (
    MIME_TYPES,
    ParseError,
    detect_mime_type,
    parse_document,
)
from tests.helpers import SAMPLE_MARKDOWN, make_docx, make_pdf

DOCX = MIME_TYPES[".docx"]


class TestDetectMimeType:
    def test_known_extensions(self) -> None:
        assert detect_mime_type("a.PDF", b"%PDF-1.7") == "application/pdf"
        assert detect_mime_type("notes.md", b"# hi") == "text/markdown"
        assert detect_mime_type("page.htm", b"<html>") == "text/html"
        assert detect_mime_type("report.docx", b"PK\x03\x04") == DOCX

    def test_rejects_unknown_extension(self) -> None:
        assert detect_mime_type("virus.exe", b"MZ") is None

    def test_rejects_binary_with_wrong_signature(self) -> None:
        assert detect_mime_type("fake.pdf", b"MZ\x90\x00") is None
        assert detect_mime_type("fake.docx", b"%PDF") is None


class TestPdf:
    def test_keeps_page_numbers_and_title(self) -> None:
        data = make_pdf(["Refund policy\nRefunds within 30 days.", "Page two text."], "Policies")
        parsed = parse_document(data, "application/pdf")
        assert parsed.title == "Policies"
        assert parsed.page_count == 2
        assert {b.page for b in parsed.blocks} == {1, 2}
        all_text = " ".join(b.text for b in parsed.blocks)
        assert "Refunds within 30 days." in all_text
        assert "Page two text." in all_text

    def test_corrupt_pdf_raises_parse_error(self) -> None:
        with pytest.raises(ParseError):
            parse_document(b"%PDF-1.4 this is not really a pdf", "application/pdf")

    def test_pdf_without_text_raises_parse_error(self) -> None:
        with pytest.raises(ParseError, match="No text"):
            parse_document(make_pdf([""]), "application/pdf")


class TestDocx:
    def test_headings_become_section_paths(self) -> None:
        parsed = parse_document(make_docx(), DOCX)
        assert parsed.title == "Travel Policy"
        by_text = {b.text: b.section_path for b in parsed.blocks}
        assert by_text["This policy applies to all employees."] == ("Travel Policy",)
        assert by_text["Hotel stays are capped at 150 EUR per night."] == (
            "Travel Policy",
            "Hotels",
        )
        assert by_text["Economy class for flights under 6 hours."] == ("Travel Policy", "Flights")

    def test_tables_are_kept_in_order(self) -> None:
        parsed = parse_document(make_docx(), DOCX)
        table = next(b for b in parsed.blocks if "London" in b.text)
        assert table.text == "City | Cap\nLondon | 200 EUR"
        assert table.section_path == ("Travel Policy", "Hotels")

    def test_garbage_raises_parse_error(self) -> None:
        with pytest.raises(ParseError):
            parse_document(b"PK not a zip", DOCX)


class TestMarkdown:
    def test_heading_hierarchy(self) -> None:
        parsed = parse_document(SAMPLE_MARKDOWN, "text/markdown")
        assert parsed.title == "Employee Handbook"
        by_text = {b.text: b.section_path for b in parsed.blocks}
        assert by_text["Welcome to the company."] == ("Employee Handbook",)
        assert by_text["Employees get 25 days of annual leave per year."] == (
            "Employee Handbook",
            "Leave",
            "Annual leave",
        )
        # A sibling heading replaces the previous one at the same level.
        assert by_text["Parents get 16 weeks of paid leave."] == (
            "Employee Handbook",
            "Leave",
            "Parental leave",
        )

    def test_code_blocks_are_verbatim_and_not_headings(self) -> None:
        parsed = parse_document(SAMPLE_MARKDOWN, "text/markdown")
        code = next(b for b in parsed.blocks if b.text.startswith("```"))
        assert "# Not a heading: inside a code block" in code.text
        assert code.section_path[-1] == "Parental leave"

    def test_windows_1252_text_is_decoded(self) -> None:
        parsed = parse_document("Café menu".encode("cp1252"), "text/markdown")
        assert parsed.blocks[0].text == "Café menu"


class TestHtml:
    HTML = b"""<html><head><title>IT Help</title><script>var x = 1;</script></head>
    <body><nav>Home | About</nav><main>
      <h1>VPN</h1><p>Use the corporate VPN when <b>remote</b>.</p>
      <h2>Setup</h2><ul><li><p>Install the client.</p></li><li>Sign in.</li></ul>
      <table><tr><th>OS</th><th>Client</th></tr><tr><td>Windows</td><td>v5</td></tr></table>
    </main><footer>Copyright</footer></body></html>"""

    def test_structure_and_noise_removal(self) -> None:
        parsed = parse_document(self.HTML, "text/html")
        assert parsed.title == "IT Help"
        texts = [b.text for b in parsed.blocks]
        assert texts == [
            "Use the corporate VPN when remote.",
            "Install the client.",
            "Sign in.",
            "OS | Client\nWindows | v5",
        ]
        assert parsed.blocks[1].section_path == ("VPN", "Setup")
        assert not any("var x" in t or "Home" in t or "Copyright" in t for t in texts)


def test_plain_text_paragraphs() -> None:
    parsed = parse_document(b"First para\nstill first.\n\n\nSecond para.", "text/plain")
    assert [b.text for b in parsed.blocks] == ["First para\nstill first.", "Second para."]


def test_empty_text_raises_parse_error() -> None:
    with pytest.raises(ParseError):
        parse_document(b"  \n\n ", "text/plain")
