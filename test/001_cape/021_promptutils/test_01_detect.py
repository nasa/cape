
# Standard library
import sys

# Third-party
import pytest

# Local imports
import cape.promptutils as pu


# Fake STDIN/STDOUT with controllable isatty()
class FakeTTY:
    def __init__(self, istty):
        self._istty = istty

    def isatty(self):
        return self._istty


# Reset detection cache and env var around each test
@pytest.fixture(autouse=True)
def clean_detection(monkeypatch):
    # Clear cached auto-detection result
    monkeypatch.setattr(pu, "_CLICKABLE_OK", None)
    # Remove env var override
    monkeypatch.delenv(pu.ENVVAR_PROMPT_CLICK, raising=False)
    yield
    pu._CLICKABLE_OK = None


# Explicit ``clickable=False`` always disables clickable prompts
def test_01_explicit_off(monkeypatch):
    monkeypatch.setattr(pu, "_textual_available", lambda: True)
    assert pu.clickable_prompt_ok(False) is False


# Explicit ``clickable=True`` requires textual, but no terminal
def test_02_explicit_on(monkeypatch):
    # With textual: forced on
    monkeypatch.setattr(pu, "_textual_available", lambda: True)
    assert pu.clickable_prompt_ok(True) is True
    # Without textual: forced on still unavailable
    monkeypatch.setattr(pu, "_textual_available", lambda: False)
    assert pu.clickable_prompt_ok(True) is False


# Environment variable overrides
def test_03_envvar(monkeypatch):
    monkeypatch.setattr(pu, "_textual_available", lambda: True)
    # "always" forces on, without terminal
    monkeypatch.setenv(pu.ENVVAR_PROMPT_CLICK, "always")
    assert pu.clickable_prompt_ok() is True
    # "never" forces off
    monkeypatch.setenv(pu.ENVVAR_PROMPT_CLICK, "never")
    assert pu.clickable_prompt_ok() is False
    # "always" still requires textual
    monkeypatch.setattr(pu, "_textual_available", lambda: False)
    monkeypatch.setenv(pu.ENVVAR_PROMPT_CLICK, "ALWAYS")
    assert pu.clickable_prompt_ok() is False
    # Explicit argument beats the environment variable
    monkeypatch.setenv(pu.ENVVAR_PROMPT_CLICK, "never")
    monkeypatch.setattr(pu, "_textual_available", lambda: True)
    assert pu.clickable_prompt_ok(True) is True


# Auto-detection needs textual, ttys, and a real $TERM
def test_04_auto(monkeypatch):
    # Start from a TUI-capable configuration
    monkeypatch.setattr(pu, "_textual_available", lambda: True)
    monkeypatch.setattr(sys, "stdin", FakeTTY(True))
    monkeypatch.setattr(sys, "stdout", FakeTTY(True))
    monkeypatch.setenv("TERM", "xterm-256color")
    assert pu.clickable_prompt_ok() is True
    # Without textual: no clickable prompts
    monkeypatch.setattr(pu, "_CLICKABLE_OK", None)
    monkeypatch.setattr(pu, "_textual_available", lambda: False)
    assert pu.clickable_prompt_ok() is False
    # Without tty on stdin: no clickable prompts
    monkeypatch.setattr(pu, "_CLICKABLE_OK", None)
    monkeypatch.setattr(pu, "_textual_available", lambda: True)
    monkeypatch.setattr(sys, "stdin", FakeTTY(False))
    assert pu.clickable_prompt_ok() is False
    # Without tty on stdout: no clickable prompts
    monkeypatch.setattr(pu, "_CLICKABLE_OK", None)
    monkeypatch.setattr(sys, "stdin", FakeTTY(True))
    monkeypatch.setattr(sys, "stdout", FakeTTY(False))
    assert pu.clickable_prompt_ok() is False
    # Dumb terminal: no clickable prompts
    monkeypatch.setattr(pu, "_CLICKABLE_OK", None)
    monkeypatch.setattr(sys, "stdout", FakeTTY(True))
    monkeypatch.setenv("TERM", "dumb")
    assert pu.clickable_prompt_ok() is False
    # Missing $TERM: no clickable prompts
    monkeypatch.setattr(pu, "_CLICKABLE_OK", None)
    monkeypatch.delenv("TERM", raising=False)
    assert pu.clickable_prompt_ok() is False


# Auto-detection result is cached per session
def test_05_caching(monkeypatch):
    monkeypatch.setattr(pu, "_textual_available", lambda: True)
    monkeypatch.setattr(sys, "stdin", FakeTTY(True))
    monkeypatch.setattr(sys, "stdout", FakeTTY(True))
    monkeypatch.setenv("TERM", "xterm")
    # First call detects and caches
    assert pu.clickable_prompt_ok() is True
    # Change the environment; cached result persists
    monkeypatch.setattr(sys, "stdout", FakeTTY(False))
    assert pu.clickable_prompt_ok() is True


# The env var name is stable and non-empty
def test_06_envvar_name():
    assert isinstance(pu.ENVVAR_PROMPT_CLICK, str)
    assert pu.ENVVAR_PROMPT_CLICK == "CAPE_PROMPT_CLICK"
