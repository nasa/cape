"""Tests for report-listing CLI commands and front-desk help."""

# Standard library
from pathlib import Path
import sys

# Third-party imports
import testutils


THIS_DIR = Path(__file__).parent
CAPE_JSON = str(THIS_DIR / "cape.json")


def test_list_reports():
    """Test listing configured reports."""
    cmd = [sys.executable, "-m", "cape", "list-reports", "-f", CAPE_JSON]
    stdout, _, ierr = testutils.call_o(cmd)
    assert ierr == 0
    assert stdout.splitlines() == ["case", "sweep"]


def test_list_report_subfigs():
    """Test listing subfigures from the default and a named report."""
    cmd = [
        sys.executable, "-m", "cape", "list-report-subfigs", "-f", CAPE_JSON]
    stdout, _, ierr = testutils.call_o(cmd)
    assert ierr == 0
    assert stdout.splitlines() == ["conditions", "history"]

    cmd.extend(["--report", "sweep"])
    stdout, _, ierr = testutils.call_o(cmd)
    assert ierr == 0
    assert stdout.splitlines() == ["conditions", "history", "sweep-table"]


def test_frontdesk_help_verbosity():
    """Test concise default help and verbose command descriptions."""
    cmd = [sys.executable, "-m", "cape", "-h"]
    stdout, _, ierr = testutils.call_o(cmd)
    assert ierr == 0
    assert "Use cape CMD -h for command help" in stdout
    assert "List configured reports" not in stdout

    cmd.append("-v")
    stdout, _, ierr = testutils.call_o(cmd)
    assert ierr == 0
    assert "List configured reports" in stdout
