
# Standard library
import builtins

# Third-party
import pytest

# Local imports
import cape.promptutils as pu


# Reset detection cache and env var around each test
@pytest.fixture(autouse=True)
def clean_detection(monkeypatch):
    monkeypatch.setattr(pu, "_CLICKABLE_OK", None)
    monkeypatch.delenv(pu.ENVVAR_PROMPT_CLICK, raising=False)
    yield
    pu._CLICKABLE_OK = None


# Mock :func:`input` to return a canned reply
def _mock_input(monkeypatch, reply):
    monkeypatch.setattr(builtins, "input", lambda msg='': reply)


# Readline prompt still answers @N, default, and free text identically
def test_01_readline_behavior(monkeypatch):
    vopt = ["next", "extend", "skip"]
    # "@N" selects option N (1-based)
    _mock_input(monkeypatch, "@2")
    v = pu.prompt_color("pick", "skip", vopt, clickable=False, show=False)
    assert v == "extend"
    # Empty input accepts default
    _mock_input(monkeypatch, "")
    v = pu.prompt_color("pick", "skip", vopt, clickable=False, show=False)
    assert v == "skip"
    # Free text passes through
    _mock_input(monkeypatch, "anything")
    v = pu.prompt_color("pick", "skip", vopt, clickable=False, show=False)
    assert v == "anything"
    # Free text without option list passes through
    _mock_input(monkeypatch, "value")
    v = pu.prompt_color("Enter a value", clickable=False, show=False)
    assert v == "value"


# Non-tty (auto-detected) falls back to readline prompt
def test_02_auto_no_tty(monkeypatch, capsys):
    # Textual would be "installed", but no terminal available
    monkeypatch.setattr(pu, "_textual_available", lambda: True)
    _mock_input(monkeypatch, "@1")
    v = pu.prompt_color("pick", "n", ["y", "n"], show=False)
    assert v == "y"
    # Nothing crashed, and no textual app was started
    out, _ = capsys.readouterr()
    assert out == ""


# Forcing clickable without textual falls back to readline prompt
def test_03_forced_without_textual(monkeypatch):
    monkeypatch.setattr(pu, "_textual_available", lambda: False)
    _mock_input(monkeypatch, "@3")
    v = pu.prompt_color(
        "pick", "skip", ["next", "extend", "skip"],
        clickable=True, show=False)
    assert v == "skip"


# Raw reply from clickable backend is parsed like typed input
def test_04_clickable_reply_parsing(monkeypatch):
    # Fake an available clickable backend
    monkeypatch.setattr(pu, "_textual_available", lambda: True)
    monkeypatch.setattr(pu, "clickable_prompt_ok", lambda c=None: True)
    # Click answers "@N", parsed to vopt[N-1]
    monkeypatch.setattr(pu, "prompt_click", lambda *a, **kw: "@2")
    v = pu.prompt_color("pick", "skip", ["next", "extend", "skip"])
    assert v == "extend"
    # Typed free text passes through the clickable backend
    monkeypatch.setattr(pu, "prompt_click", lambda *a, **kw: "custom")
    v = pu.prompt_color("pick", "skip", ["next", "extend", "skip"],
                        show=False)
    assert v == "custom"
    # Empty reply (Escape) accepts default
    monkeypatch.setattr(pu, "prompt_click", lambda *a, **kw: "")
    v = pu.prompt_color("pick", "skip", ["next", "extend", "skip"],
                        show=False)
    assert v == "skip"


# Errors from clickable backend fall back to readline prompt
def test_05_clickable_error_fallback(monkeypatch):
    monkeypatch.setattr(pu, "clickable_prompt_ok", lambda c=None: True)

    def bad_click(*a, **kw):
        raise RuntimeError("terminal blew up")

    monkeypatch.setattr(pu, "prompt_click", bad_click)
    _mock_input(monkeypatch, "@1")
    v = pu.prompt_color("pick", "skip", ["next", "extend", "skip"],
                        clickable=True, show=False)
    assert v == "next"


# KeyboardInterrupt from clickable backend is not swallowed
def test_06_clickable_interrupt(monkeypatch):
    monkeypatch.setattr(pu, "clickable_prompt_ok", lambda c=None: True)

    def cancelled_click(*a, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(pu, "prompt_click", cancelled_click)
    _mock_input(monkeypatch, "next")
    with pytest.raises(KeyboardInterrupt):
        pu.prompt_color("pick", "skip", ["next", "extend", "skip"],
                        clickable=True, show=False)


# prompt_click maps a quit app (None reply) to KeyboardInterrupt
def test_07_prompt_click_quit(monkeypatch):

    class QuitApp:
        def run(self):
            return None

    class ReplyApp:
        def run(self):
            return "@1"

    # App that quits without answering (e.g. Ctrl-C) raises
    monkeypatch.setattr(pu, "_new_click_prompt", lambda *a, **kw: QuitApp())
    with pytest.raises(KeyboardInterrupt):
        pu.prompt_click("pick", "skip", ["next"])
    # Normal reply passes through
    monkeypatch.setattr(pu, "_new_click_prompt", lambda *a, **kw: ReplyApp())
    assert pu.prompt_click("pick", "skip", ["next"]) == "@1"


# Plain prompts without an option list never use clickable backend
def test_08_plain_prompt_no_click(monkeypatch):
    monkeypatch.setattr(pu, "clickable_prompt_ok", lambda c=None: True)

    def click_should_not_run(*a, **kw):
        raise AssertionError("clickable backend used on plain prompt")

    monkeypatch.setattr(pu, "prompt_click", click_should_not_run)
    _mock_input(monkeypatch, "typed")
    v = pu.prompt_color("Enter a value", clickable=True, show=False)
    assert v == "typed"
