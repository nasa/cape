# Standard library
import os

# Local imports
from cape import capeconfig
from cape.agent import agentutils
from cape.tui import tuiutils


def test_default_histfiles(clean_env, monkeypatch):
    # Set home folder to temp location
    monkeypatch.setenv("HOME", str(clean_env))
    # All three history files default to $XDG_STATE_HOME fallback
    statedir = clean_env / ".local" / "state" / "cape"
    assert capeconfig.get_cape_histfile("HistoryFile") == str(
        statedir / ".cape_history")
    assert capeconfig.get_cape_histfile("AgentHistoryFile") == str(
        statedir / ".cape_agent_history")
    assert capeconfig.get_cape_histfile("TUIHistoryFile") == str(
        statedir / ".cape_tui_history")
    # State folder gets created for readline to write to
    assert statedir.is_dir()
    # Solver-specific wrappers agree
    assert agentutils.get_agent_histfile() == str(
        statedir / ".cape_agent_history")
    assert tuiutils.get_tui_histfile() == str(
        statedir / ".cape_tui_history")


def test_histfiles_obey_xdg_state_home(clean_env, monkeypatch):
    # Set $XDG_STATE_HOME to absolute path
    monkeypatch.setenv("XDG_STATE_HOME", str(clean_env / "xdg-state"))
    assert capeconfig.get_cape_histfile("HistoryFile") == str(
        clean_env / "xdg-state" / "cape" / ".cape_history")


def test_relative_histfile_under_statedir(clean_env, monkeypatch):
    # Set relative history name and explicit *StateDir*
    monkeypatch.setenv("CAPE_HISTORY_FILE", "myhist")
    monkeypatch.setenv("CAPE_STATE_DIR", str(clean_env / "state"))
    assert capeconfig.get_cape_histfile("HistoryFile") == str(
        clean_env / "state" / "myhist")


def test_absolute_histfile_unchanged(clean_env, monkeypatch):
    # Absolute paths are used without modification
    histfile = clean_env / "abs" / "hist"
    monkeypatch.setenv("CAPE_TUI_HISTORY_FILE", str(histfile))
    assert capeconfig.get_cape_histfile("TUIHistoryFile") == str(histfile)


def test_histfile_expands_user(clean_env, monkeypatch):
    # A leading '~' expands to user's home folder
    monkeypatch.setenv("HOME", str(clean_env))
    monkeypatch.setenv("CAPE_HISTORY_FILE", os.path.join("~", ".hist"))
    assert capeconfig.get_cape_histfile("HistoryFile") == str(
        clean_env / ".hist")
