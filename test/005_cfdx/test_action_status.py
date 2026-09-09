"""Test status text for ``cape-perform`` actions."""

# Local imports
from cape.cfdx.cntl import Cntl, _action_status_line, _format_action_call
from cape.cfdx.options.actionopts import ActionOpts


def test_format_action_call():
    """Include explicit positional and keyword args in an action call."""
    # Short calls are shown in full
    assert _format_action_call(
        "Cntl", "update_dex", {"kwargs": {"dex": "[A-C]*"}}
    ) == 'Cntl.update_dex(dex="[A-C]*")'
    # Cap the argument payload at 20 characters, including the ellipsis
    act = {
        "args": [3, "wing"],
        "kwargs": {"dex": "[A-C]*", "force": True},
    }
    assert _format_action_call("Cntl", "update_dex", act) == (
        'Cntl.update_dex(3, "wing", dex="[...)')


def test_action_title(capsys, monkeypatch):
    """Show explicit args and kwargs in the action banner."""
    act = ActionOpts({
        "type": "cntl",
        "function": "update_dex",
        "kwargs": {"dex": "[A-C]*"},
    })
    cntl = object.__new__(Cntl)
    monkeypatch.setattr(Cntl, "_run_action", lambda *args: None)
    cntl._perform_action(act, [], 0, 1)
    assert capsys.readouterr().out == (
        '-- (1/1) Cntl.update_dex(dex="[A-C]*") --\n')


def test_action_status_line_notty():
    """Let ``compile_rst()`` suppress colors for redirected output."""
    line = _action_status_line("Cntl.update_dex()", 0, 1, 2)
    assert line == "  ✔ Cntl.update_dex()  log/cape-perform.2.3"
