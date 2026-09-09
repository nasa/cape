"""Test the ``AddFileName`` option for ``cape-perform`` actions."""

# Third-party
import numpy as np

# Local imports
from cape.cfdx import cli
from cape.cfdx.cntl import Cntl
from cape.cfdx.options.actionopts import ActionOpts


# Create a bare Cntl instance w/ a known JSON file name
def _bare_cntl(fdir: str = "") -> Cntl:
    cntl = object.__new__(Cntl)
    cntl.RootDir = "/root/dir"
    cntl.fdir = fdir
    cntl.fname = "cape.json"
    return cntl


def test_shell_action_addfilename(monkeypatch):
    """Append ``-f`` to shell commands when *AddFileName* is true."""
    cmds = []
    monkeypatch.setattr(
        Cntl, "_perform_shell_action",
        lambda self, cmd, I: cmds.append(cmd))
    # Shell action w/ AddFileName
    act = ActionOpts("./tools/mytool.py {I}", AddFileName=True)
    _bare_cntl()._perform_action(act, [0], 0, 1)
    assert cmds == ["./tools/mytool.py {I} -f cape.json"]
    # W/ JSON file in a subfolder of the root dir
    cmds.clear()
    _bare_cntl("cases")._perform_action(act, [0], 0, 1)
    assert cmds == ["./tools/mytool.py {I} -f cases/cape.json"]
    # W/o AddFileName, command is unmodified
    cmds.clear()
    act = ActionOpts("./tools/mytool.py", AddFileName=False)
    _bare_cntl()._perform_action(act, [0], 0, 1)
    assert cmds == ["./tools/mytool.py"]


def test_cli_action_addfilename(monkeypatch):
    """Add ``f=`` kwarg to cli functions when *AddFileName* is true."""
    calls = []
    monkeypatch.setattr(
        cli, "cape_fake_action",
        lambda *a, **kw: calls.append((a, kw)),
        raising=False)
    # cli action w/ AddFileName
    act = ActionOpts({
        "type": "cli",
        "function": "cape_fake_action",
        "AddFileName": True,
    })
    _bare_cntl()._perform_action(act, np.array([0, 1]), 0, 1)
    assert calls == [((), {"I": "0:2", "f": "cape.json"})]
    # Explicit "f" in the action's kwargs takes precedence
    calls.clear()
    act = ActionOpts({
        "type": "cli",
        "function": "cape_fake_action",
        "AddFileName": True,
        "kwargs": {"f": "other.json"},
    })
    _bare_cntl("cases")._perform_action(act, np.array([0]), 0, 1)
    assert calls == [((), {"I": "0", "f": "other.json"})]
    # W/o AddFileName, no "f" kwarg is added
    calls.clear()
    act = ActionOpts({"type": "cli", "function": "cape_fake_action"})
    _bare_cntl()._perform_action(act, np.array([0]), 0, 1)
    assert calls == [((), {"I": "0"})]


def test_cntl_action_addfilename_ignored():
    """*AddFileName* has no effect on ``Cntl``-method actions."""
    calls = []
    cntl = _bare_cntl()
    cntl.fake_method = lambda *a, **kw: calls.append((a, kw))
    # cntl action w/ AddFileName; no file name is passed
    act = ActionOpts({
        "type": "cntl",
        "function": "fake_method",
        "AddFileName": True,
    })
    cntl._perform_action(act, [0], 0, 1)
    assert calls == [((), {"I": [0]})]
