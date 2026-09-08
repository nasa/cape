"""Tests for lazy loading of skill-provided tool schemas."""

# Third-party imports
import pytest

# Local imports
from cape.agent import agentcntl
from cape.agent.skills import checkdbtools, fileedit, skillbase


class DummyOpts:
    """Minimal agent options for testing skill assembly."""

    def __init__(self, skillset):
        self.skillset = skillset

    def get_ModelOpt(self, model, name, vdef=None):
        assert name == "SkillSet"
        return self.skillset

    def get_opt(self, name, vdef=None):
        return vdef


@pytest.fixture(autouse=True)
def active_skills():
    """Restore the process-wide available-skill registry after tests."""
    saved = dict(skillbase.ACTIVE_SKILLS)
    yield
    skillbase.ACTIVE_SKILLS.clear()
    skillbase.ACTIVE_SKILLS.update(saved)


def make_cntl(skillset, monkeypatch):
    """Create a controller shell and assemble only its skills."""
    cntl = agentcntl.AgentCntl.__new__(agentcntl.AgentCntl)
    cntl.RootDir = "."
    cntl.model = "test-model"
    cntl.opts = DummyOpts(skillset)
    cntl.tools = {}
    cntl.tool_schemas = []
    monkeypatch.setattr(agentcntl, "discover_user_skills", lambda root: {})
    cntl.assemble_skills()
    return cntl


def schema_names(cntl):
    """Return tool names from a controller's active schemas."""
    return [schema["function"]["name"] for schema in cntl.tool_schemas]


def test_01_skill_tools_start_hidden(monkeypatch):
    """Only ``use_skill`` is exposed before a skill is loaded."""
    cntl = make_cntl("full", monkeypatch)
    assert schema_names(cntl) == ["use_skill"]
    assert set(cntl.tools) == {"use_skill"}
    assert cntl.tools["use_skill"] == cntl.use_skill
    assert cntl.loaded_skills == set()


def test_02_loading_activates_declared_tools(monkeypatch):
    """Loading a skill adds exactly its declared tools and schemas."""
    cntl = make_cntl("full", monkeypatch)
    result = cntl.tools["use_skill"]("check-db")
    assert result["success"] is True
    assert result["tools_added"] == ["cape_check_db"]
    assert result["tools_active"] == ["cape_check_db"]
    assert cntl.tools["cape_check_db"] is checkdbtools.cape_check_db
    assert schema_names(cntl) == ["use_skill", "cape_check_db"]
    assert cntl.loaded_skills == {"check-db"}


def test_03_loading_is_idempotent(monkeypatch):
    """Loading the same skill twice does not duplicate its schemas."""
    cntl = make_cntl("full", monkeypatch)
    cntl.use_skill("check-db")
    result = cntl.use_skill("check-db")
    assert result["success"] is True
    assert result["tools_added"] == []
    assert result["tools_active"] == ["cape_check_db"]
    assert schema_names(cntl).count("cape_check_db") == 1


def test_04_no_skill_overhead_when_disabled(monkeypatch):
    """A disabled skill set does not expose even ``use_skill``."""
    cntl = make_cntl("none", monkeypatch)
    assert cntl.skills == {}
    assert cntl.tools == {}
    assert cntl.tool_schemas == []
    assert cntl.system_prompt == agentcntl.SYSTEM_PROMPT


def test_05_unknown_skill_does_not_activate(monkeypatch):
    """An unavailable skill returns an error without changing tools."""
    cntl = make_cntl("full", monkeypatch)
    result = cntl.use_skill("nope")
    assert result["success"] is False
    assert set(cntl.tools) == {"use_skill"}
    assert schema_names(cntl) == ["use_skill"]


def test_06_file_editor_activation(monkeypatch):
    """Loading ``file-editor`` activates its three file tools."""
    cntl = make_cntl("full", monkeypatch)
    # Configured with an empty allow-list by default
    assert fileedit.ALLOW_PATTERNS == []
    result = cntl.use_skill("file-editor")
    assert result["success"] is True
    ftools = ["list_editable_files", "read_file", "edit_file"]
    assert result["tools_added"] == ftools
    assert result["tools_active"] == ftools
    assert cntl.tools["edit_file"] is fileedit.edit_file
    assert schema_names(cntl) == ["use_skill"] + ftools
    assert cntl.loaded_skills == {"file-editor"}
