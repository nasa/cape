"""Tests for the built-in ``check-db`` skill."""

# Standard library
from unittest import mock

# Local imports
from cape.agent import skills
from cape.agent.skills import checkdbtools
from cape.agent.tools import cfdxtools


def test_01_builtin_skill():
    """Check skill registration and removal of the old fixed tools."""
    skill = skills.BUILTIN_SKILLS["check-db"]
    assert skill.tools == ["cape_check_db"]
    assert "check-db" in skills.SKILL_SETS["medium"]
    assert "check-db" in skills.SKILL_SETS["full"]
    assert "check-db" not in skills.SKILL_SETS["low"]
    assert not any(
        name in cfdxtools.TOOL_DICT
        for name in (
            "cape_check_db",
            "cape_check_fm",
            "cape_check_ll",
            "cape_check_triqfm",
        )
    )


def test_02_tool_schema():
    """Check that the four commands use one compact tool schema."""
    assert checkdbtools.TOOLS["cape_check_db"] is checkdbtools.cape_check_db
    assert len(checkdbtools.TOOL_SCHEMAS) == 1
    schema = checkdbtools.TOOL_SCHEMAS[0]["function"]
    assert schema["name"] == "cape_check_db"
    component = schema["parameters"]["properties"]["component"]
    assert component["enum"] == ["all", "fm", "ll", "triqfm"]


def test_03_dispatch():
    """Check dispatch of each compact component selector."""
    for component, func in checkdbtools.CHECK_FUNCS.items():
        with mock.patch.object(
                checkdbtools.toolutils, "wrap_cli") as wrap_cli:
            wrap_cli.return_value = {"success": True}
            result = checkdbtools.cape_check_db(
                component, f="cape.json", I="2:4")
        assert result["success"] is True
        wrap_cli.assert_called_once_with(func, f="cape.json", I="2:4")


def test_04_bad_component():
    """Check validation of an invalid component selector."""
    result = checkdbtools.cape_check_db("nope", I="0")
    assert result["success"] is False
    assert result["allowed_components"] == ["all", "fm", "ll", "triqfm"]
