"""Tests for the ``cape init-agent`` command."""

# Standard library
from pathlib import Path

# Local imports
from cape import agent
from cape.cfdx import cli


AGENT_DIR = Path(agent.__file__).parent


def test_init_agent_creates_files(tmp_path, monkeypatch):
    """Test creation of both agent instruction files."""
    monkeypatch.chdir(tmp_path)
    assert cli.main(["cape", "init-agent"]) == 0
    assert (tmp_path / "AGENTS.md").read_text() == \
        (AGENT_DIR / "project-agents.md").read_text()
    assert (tmp_path / "ANALYSIS.md").read_text() == \
        (AGENT_DIR / "ANALYSIS.md").read_text()


def test_init_agent_appends_cape_section(tmp_path, monkeypatch):
    """Test preserving local content while adding CAPE instructions."""
    agents = tmp_path / "AGENTS.md"
    analysis = tmp_path / "ANALYSIS.md"
    agents.write_text("# Local instructions\n\nKeep this text.\n")
    analysis.write_text("# Existing analysis\n")
    monkeypatch.chdir(tmp_path)
    assert cli.main(["cape", "init-agent"]) == 0
    text = agents.read_text()
    assert text.startswith("# Local instructions\n\nKeep this text.\n")
    assert text.endswith((AGENT_DIR / "project-agents.md").read_text())
    assert analysis.read_text() == "# Existing analysis\n"


def test_init_agent_is_idempotent(tmp_path, monkeypatch):
    """Test an existing CAPE section and analysis file remain unchanged."""
    agents = tmp_path / "AGENTS.md"
    analysis = tmp_path / "ANALYSIS.md"
    agents.write_text("# Local\n\n## CAPE\n\nExisting guidance.\n")
    analysis.write_text("# Existing analysis\n")
    monkeypatch.chdir(tmp_path)
    assert cli.main(["cape", "init-agent"]) == 0
    assert agents.read_text() == "# Local\n\n## CAPE\n\nExisting guidance.\n"
    assert analysis.read_text() == "# Existing analysis\n"
