"""Tests for the ``cape data-path`` command."""

# Standard library
from pathlib import Path
import sys

# Third-party imports
import testutils

# Local imports
import cape


def test_data_paths():
    """Test lookup of the packaged agent data files."""
    for name in ("AGENTS.md", "ANALYSIS.md", "project-agents.md"):
        cmd = [sys.executable, "-m", "cape", "data-path", name]
        stdout, _, ierr = testutils.call_o(cmd)
        expected = Path(cape.__file__).parent / "agent" / name
        assert ierr == 0
        assert stdout.strip() == str(expected.resolve())


def test_data_path_unknown_name():
    """Test rejection of names outside the public allowlist."""
    cmd = [sys.executable, "-m", "cape", "data-path", "UNKNOWN"]
    _, _, ierr = testutils.call_o(cmd)
    assert ierr != 0
