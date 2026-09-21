"""Tests for the built-in ``check-cases`` skill."""

# Standard library
from unittest import mock

# Local imports
from cape.agent import skills
from cape.agent.skills import checkcases
from cape.agent.tools import cfdxtools
from cape.cfdx import cli


def test_01_builtin_skill():
    """Check skill registration and tier membership."""
    skill = skills.BUILTIN_SKILLS["check-cases"]
    assert skill.tools == ["cape_check"]
    assert skills.SKILL_TOOL_MODULES["check-cases"] is checkcases
    assert "check-cases" in skills.SKILL_SETS["full"]
    assert "check-cases" not in skills.SKILL_SETS["medium"]
    assert "check-cases" not in skills.SKILL_SETS["low"]
    assert "check-cases" not in skills.SKILL_SETS["none"]


def test_02_cape_c_removed_from_full():
    """Simple cape_c tool is for low/medium; full uses the skill."""
    assert "cape_c" not in cfdxtools.TOOL_SETS["full"]
    assert "cape_c" in cfdxtools.TOOL_SETS["medium"]
    assert "cape_c" in cfdxtools.TOOL_SETS["low"]
    assert "cape_check" not in cfdxtools.TOOL_DICT


def test_03_pruned_tools():
    """Unreachable tool schemas are not registered."""
    for name in (
            "cape_dispatch",
            "cape_review",
            "cape_open_pdf",
            "cape_open_subfig",
            "cape_skeleton"):
        assert name not in cfdxtools.TOOL_DICT
    # Every remaining tool is in at least one set
    all_names = set()
    for names in cfdxtools.TOOL_SETS.values():
        all_names.update(names)
    for name in cfdxtools.TOOL_DICT:
        assert name in all_names or name in ("cape_c", "cape_inspect_json")


def test_04_tool_schema():
    """Check that the rich check tool schema was registered."""
    assert checkcases.TOOLS["cape_check"] is checkcases.cape_check
    assert len(checkcases.TOOL_SCHEMAS) == 1
    schema = checkcases.TOOL_SCHEMAS[0]["function"]
    assert schema["name"] == "cape_check"
    props = schema["parameters"]["properties"]
    for opt in ("cols", "counters", "hide_cols", "status", "nproc"):
        assert opt in props


def test_05_column_translation():
    """Comma strings become lists; hide_cols becomes hide-cols."""
    with mock.patch.object(
            checkcases.toolutils, "wrap_cli") as wrap_cli:
        wrap_cli.return_value = {"success": True}
        result = checkcases.cape_check(
            I="100:500", cols="i,frun,user", counters="")
    assert result["success"] is True
    wrap_cli.assert_called_once_with(
        cli.cape_c,
        I="100:500",
        cols=["i", "frun", "user"],
        counters=[],
        __long_stdout=True)


def test_06_hide_cols_translation():
    """Comma-separated hide_cols strings are split into a list."""
    with mock.patch.object(
            checkcases.toolutils, "wrap_cli") as wrap_cli:
        wrap_cli.return_value = {"success": True}
        result = checkcases.cape_check(hide_cols="progress,cpu-hours")
    assert result["success"] is True
    wrap_cli.assert_called_once_with(
        cli.cape_c,
        hide_cols=["progress", "cpu-hours"],
        __long_stdout=True)


def test_07_hide_cols_parser_alias():
    """The CLI parser accepts the Pythonic hide_cols kwarg."""
    parser = cli.CfdxCheckArgs(hide_cols="progress,cpu-hours")
    assert parser.get_opt("hide-cols") == "progress,cpu-hours"


def test_07_cli_alias():
    """Tool name maps to the ``check`` command for CLI display."""
    assert cli.CMD_FUNCS.get("cape_check") == "check"
