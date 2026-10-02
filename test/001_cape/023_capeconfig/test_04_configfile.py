# Standard library
import json
import os

# Local imports
from cape import capeconfig


def test_envvar_wins_over_legacy(clean_env, monkeypatch):
    # Create legacy config file in temp home folder
    monkeypatch.setenv("HOME", str(clean_env))
    legacy_configfile = clean_env / ".capeconfig.json"
    legacy_configfile.write_text("{}")
    # $CAPE_CONFIG_FILE takes precedence even over legacy file
    env_configfile = clean_env / "other.json"
    monkeypatch.setenv("CAPE_CONFIG_FILE", str(env_configfile))
    assert capeconfig.get_cape_configfile() == str(env_configfile)


def test_legacy_wins_over_xdg(clean_env, monkeypatch):
    # Create legacy config file in temp home folder
    monkeypatch.delenv("CAPE_CONFIG_FILE")
    monkeypatch.setenv("HOME", str(clean_env))
    legacy_configfile = clean_env / ".capeconfig.json"
    legacy_configfile.write_text("{}")
    # Legacy file takes precedence over $XDG_CONFIG_HOME location
    monkeypatch.setenv("XDG_CONFIG_HOME", str(clean_env / "xdg"))
    assert capeconfig.get_cape_configfile() == str(legacy_configfile)


def test_xdg_configfile(clean_env, monkeypatch):
    # Set $XDG_CONFIG_HOME to absolute path; no legacy file
    monkeypatch.delenv("CAPE_CONFIG_FILE")
    monkeypatch.setenv("HOME", str(clean_env))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(clean_env / "xdg"))
    configfile = capeconfig.get_cape_configfile()
    assert configfile == str(clean_env / "xdg" / "cape" / "config.json")
    # Reading config creates the file and its parent folder
    capeconfig.read_cape_config()
    assert os.path.isfile(configfile)


def test_default_configfile(clean_env, monkeypatch):
    # No $XDG_CONFIG_HOME, no legacy file
    monkeypatch.delenv("CAPE_CONFIG_FILE")
    monkeypatch.setenv("HOME", str(clean_env))
    configfile = capeconfig.get_cape_configfile()
    assert configfile == str(clean_env / ".config" / "cape" / "config.json")
    # Reading config creates the file and its parent folder
    capeconfig.read_cape_config()
    assert os.path.isfile(configfile)


def test_invalid_xdg_config_home(clean_env, monkeypatch):
    # Set home folder to temp location
    monkeypatch.delenv("CAPE_CONFIG_FILE")
    monkeypatch.setenv("HOME", str(clean_env))
    # Empty $XDG_CONFIG_HOME is ignored per the XDG spec
    monkeypatch.setenv("XDG_CONFIG_HOME", "")
    assert capeconfig.get_default_configfile() == os.path.join(
        "~", ".config", "cape", "config.json")
    # Relative $XDG_CONFIG_HOME is also invalid per the XDG spec
    monkeypatch.setenv("XDG_CONFIG_HOME", "relative/xdg")
    assert capeconfig.get_default_configfile() == os.path.join(
        "~", ".config", "cape", "config.json")


def test_set_cape_opt_writes_resolved_file(clean_env, monkeypatch):
    # No legacy file; config should be created at XDG default location
    monkeypatch.delenv("CAPE_CONFIG_FILE")
    monkeypatch.setenv("HOME", str(clean_env))
    capeconfig.set_cape_opt("LocalHost", "myhost")
    configfile = clean_env / ".config" / "cape" / "config.json"
    assert json.loads(configfile.read_text())["LocalHost"] == "myhost"
