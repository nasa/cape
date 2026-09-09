
# Local
from cape.cfdx.options import Options
from cape.cfdx.options.actionopts import (
    ActionOpts, ActionsOpts, DEFAULT_ACTIONS)


def test_action_shell_str():
    # Single action from a shell string
    opts = ActionOpts("mycmd {I}")
    assert opts["function"] == "mycmd {I}"
    # Default action type
    assert opts.get_opt("type") == "shell"


def test_action_types():
    # Full action definition
    opts = ActionOpts({
        "type": "cli",
        "function": "cape_extract_dex",
        "index": 2,
        "kwargs": {"dex": "[A-L]*"},
    })
    assert opts["type"] == "cli"
    assert opts["index"] == 2
    assert opts["kwargs"]["dex"] == "[A-L]*"
    # Bad action type is rejected w/ a warning
    assert ActionOpts(type="bogus").get("type") is None


def test_action_addfilename():
    # Option to add JSON file to "shell" | "cli" actions
    opts = ActionOpts({"function": "mycmd", "AddFileName": True})
    assert opts["AddFileName"] is True
    # Default value
    assert ActionOpts("mycmd").get_opt("AddFileName") is False
    # Non-bool value rejected w/ a warning
    assert ActionOpts(
        function="mycmd", AddFileName="yes").get("AddFileName") is None


def test_actions_actions():
    # Section with a mix of action definition formats
    opts = ActionsOpts({
        "approve": [
            {"type": "cntl", "function": "MarkPASS"},
            "./tool.py",
        ],
        "UserTools": ["writecp"],
        "writecp": "./tools/writesurfcp.py {I}",
    })
    # Check normalization of action lists
    actlist = opts.get_Action("approve")
    assert len(actlist) == 2
    assert all(isinstance(actj, ActionOpts) for actj in actlist)
    assert actlist[0]["type"] == "cntl"
    assert actlist[1]["function"] == "./tool.py"
    # UserTools names and definitions
    assert opts.get_UserToolsNames() == ["writecp"]
    assert (
        opts.get_Action("writecp")[0]["function"] ==
        "./tools/writesurfcp.py {I}")


def test_actions_defaults():
    # Empty section still provides default actions
    opts = ActionsOpts()
    # Check each default
    for name, actlist in DEFAULT_ACTIONS.items():
        assert [
            dict(actj) for actj in opts.get_Action(name)
        ] == [dict(vj) for vj in actlist]
    # Unknown action name
    assert opts.get_Action("bogus") is None


def test_actions_in_options():
    # Read from full CAPE options
    opts = Options(**{
        "Actions": {
            "UserTools": ["t1"],
            "t1": "echo {I}",
        },
    })
    # Section converted to correct class
    assert isinstance(opts["Actions"], ActionsOpts)
    # Promoted methods work
    assert opts.get_UserToolsNames() == ["t1"]
    assert opts.get_Action("t1")[0]["function"] == "echo {I}"
    # Defaults still available
    assert opts.get_Action("approve")[0]["function"] == "MarkPASS"
