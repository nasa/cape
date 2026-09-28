# Standard library
from io import BytesIO

# Third-party
import pytest

# Local imports
from cape.fileutils import readline_reverse


@pytest.mark.parametrize("contents", [
    b"", b"x", b"\n", b"x\n", b"record", b"record\n",
    b"\nrecord", b"\nrecord\n", b"\n\nrecord\n", b"\n\n",
    b"first\n\nlast\n", b"first\r\nlast\r\n",
    "\ncafé\n".encode("utf-8"),
])
@pytest.mark.parametrize("in_memory", [False, True])
def test_reverse_line_preserves_all_lines(tmp_path, contents, in_memory):
    path = tmp_path / "output.log"
    path.write_bytes(contents)
    stream = BytesIO(contents) if in_memory else path.open("rb")
    with stream:
        stream.seek(0, 2)
        for expected in reversed(contents.splitlines(keepends=True)):
            previous = stream.tell()
            assert readline_reverse(stream) == expected
            assert stream.tell() < previous
        assert stream.tell() == 0
        assert readline_reverse(stream) == b""
        assert readline_reverse(stream) == b""
        assert stream.tell() == 0


@pytest.mark.parametrize("contents", [b"xremaining\n", b"\nremaining\n"])
@pytest.mark.parametrize("in_memory", [False, True])
def test_reverse_line_at_first_byte_stays_before_cursor(
        tmp_path, contents, in_memory):
    path = tmp_path / "output.log"
    path.write_bytes(contents)
    stream = BytesIO(contents) if in_memory else path.open("rb")
    with stream:
        stream.seek(1)
        assert readline_reverse(stream) == contents[:1]
        assert stream.tell() == 0
        assert readline_reverse(stream) == b""
