import io

import pytest

from app.cli import password_from_stream


@pytest.mark.parametrize(
    "raw",
    [
        b"s3cret-password\n",
        b"s3cret-password\r\n",  # Windows line ending
        b"\xef\xbb\xbfs3cret-password\r\n",  # Windows PowerShell pipes a UTF-8 BOM first
        b"s3cret-password",  # no trailing newline
        b"s3cret-password\nignored second line\n",
    ],
)
def test_password_from_stream(raw: bytes) -> None:
    assert password_from_stream(io.BytesIO(raw)) == "s3cret-password"


def test_password_keeps_meaningful_whitespace_and_unicode() -> None:
    assert password_from_stream(io.BytesIO(" pässwörd mit space \n".encode())) == (
        " pässwörd mit space "
    )
