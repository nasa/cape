# Standard library
import json
import os

# Local imports
from cape import capeconfig


def test_default_statedir(clean_env, monkeypatch):
    # Set home folder to temp location
    monkeypatch.setenv("HOME", str(clean_env))
    # Check default when $XDG_STATE_HOME not set
    assert capeconfig.get_default_statedir() == os.path.join(
        "~", ".local", "state", "cape")
    # Default should also work through full getter, which creates dir
    statedir = capeconfig.get_cape_statedir()
    assert statedir == str(clean_env / ".local" / "state" / "cape")
    assert os.path.isdir(statedir)


def test_default_statedir_invalid_xdg(clean_env, monkeypatch):
    # Set home folder to temp location
    monkeypatch.setenv("HOME", str(clean_env))
    # Empty $XDG_STATE_HOME is ignored per the XDG spec
    monkeypatch.setenv("XDG_STATE_HOME", "")
    assert capeconfig.get_default_statedir() == os.path.join(
        "~", ".local", "state", "cape")
    # Relative $XDG_STATE_HOME is also invalid per the XDG spec
    monkeypatch.setenv("XDG_STATE_HOME", "relative/state")
    assert capeconfig.get_default_statedir() == os.path.join(
        "~", ".local", "state", "cape")


def test_xdg_statedir(clean_env, monkeypatch):
    # Set $XDG_STATE_HOME to absolute path
    xdg_statedir = clean_env / "xdg-state"
    monkeypatch.setenv("XDG_STATE_HOME", str(xdg_statedir))
    # Check default
    assert capeconfig.get_default_statedir() == str(xdg_statedir / "cape")
    # Full getter should match and create the folder
    statedir = capeconfig.get_cape_statedir()
    assert statedir == str(xdg_statedir / "cape")
    assert os.path.isdir(statedir)
    # Also works through generic option getter
    assert capeconfig.get_cape_opt("StateDir") == statedir


def test_envvar_overrides_xdg(clean_env, monkeypatch):
    # Set both env vars
    statedir_expected = clean_env / "cape-env"
    monkeypatch.setenv("XDG_STATE_HOME", str(clean_env / "xdg-state"))
    monkeypatch.setenv("CAPE_STATE_DIR", str(statedir_expected))
    # CAPE-specific env var takes precedence
    assert capeconfig.get_cape_statedir() == str(statedir_expected)


def test_configfile_overrides_xdg(clean_env, monkeypatch):
    # Set *StateDir* in config file
    statedir_expected = clean_env / "cape-cfg"
    configfile = clean_env / "capeconfig.json"
    configfile.write_text(json.dumps({"StateDir": str(statedir_expected)}))
    # Value from config file takes precedence over $XDG_STATE_HOME
    monkeypatch.setenv("XDG_STATE_HOME", str(clean_env / "xdg-state"))
    assert capeconfig.get_cape_statedir() == str(statedir_expected)
