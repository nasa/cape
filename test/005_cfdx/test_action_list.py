"""Test ``cape perform --list`` (:meth:`Cntl.list_actions`)."""

# Standard library
import contextlib
import io

# Local imports
from cape.cfdx import cli
from cape.cfdx.cntl import Cntl
from cape.cfdx.options import Options


# Create a bare Cntl instance w/ given *Actions* section
def _bare_cntl(actions: dict) -> Cntl:
    cntl = object.__new__(Cntl)
    cntl.opts = Options(Actions=actions)
    return cntl


def test_list_actions():
    """List default and user-defined actions with their steps."""
    cntl = _bare_cntl({
        "UserTools": ["bump"],
        "bump": {"function": "./tools/bump.py", "AddFileName": True},
        "approve": [
            {"type": "cntl", "function": "MarkPASS"},
            {"type": "cntl", "function": "update_dex", "index": 2,
             "kwargs": {"dex": "A*"}},
            {"type": "cntl", "function": "update_dex", "index": 2,
             "kwargs": {"dex": "B*"}},
        ],
    })
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        names = cntl.list_actions()
    # Defaults plus user-defined, w/o "UserTools"
    assert names == [
        "approve", "bump", "defail", "dezombie", "extend", "extend2"]
    out = buf.getvalue()
    assert "./tools/bump.py [+f]" in out
    assert 'Cntl.update_dex(dex="A*") [index=2]' in out
    assert "Cntl.MarkPASS() [index" not in out
    # Redefined "approve" is not marked as a default
    lines = out.splitlines()
    assert any("approve" in line and "default" not in line for line in lines)
    assert any("extend2" in line and "(default)" in line for line in lines)


def test_cli_perform_list(monkeypatch):
    """``--list`` and its alias ``--ls`` don't require an action."""
    calls = []
    monkeypatch.setattr(
        cli, "read_cntl",
        lambda cls, *a, **kw: (_FakeCntl(calls), kw))
    for opt in ("--list", "--ls"):
        argv = ["cape", "perform", opt]
        a, kw = cli.CfdxPerformArgs().parse(argv)
        ierr, v = cli.cape_perform(*a, **kw)
        assert ierr == cli.IERR_OK
    assert calls == ["list", "list"]
    # No action and no --list is an option error
    ierr, _ = cli.cape_perform()
    assert ierr == cli.IERR_OPT


class _FakeCntl:
    def __init__(self, calls: list):
        self.calls = calls

    def list_actions(self):
        self.calls.append("list")
        return []
