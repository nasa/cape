"""Tests for the ``cape data-path`` command."""

# Standard library
from pathlib import Path
import sys

# Third-party imports
import testutils

# Local imports
import cape


def test_data_path_agents():
    """Test lookup of the packaged agent instructions."""
    cmd = [sys.executable, "-m", "cape", "data-path", "AGENTS.md"]
    stdout, _, ierr = testutils.call_o(cmd)
    expected = Path(cape.__file__).parent / "agent" / "AGENTS.md"
    assert ierr == 0
    assert stdout.strip() == str(expected.resolve())


def test_data_path_unknown_name():
    """Test rejection of names outside the public allowlist."""
    cmd = [sys.executable, "-m", "cape", "data-path", "UNKNOWN"]
    _, _, ierr = testutils.call_o(cmd)
    assert ierr != 0
