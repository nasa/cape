# Standard library
import json
import os

# Local imports
from cape import capeconfig


def test_default_cachedir(clean_env, monkeypatch):
    # Set home folder to temp location
    monkeypatch.setenv("HOME", str(clean_env))
    # Check default when $XDG_CACHE_HOME not set
    assert capeconfig.get_default_cachedir() == os.path.join(
        "~", ".cache", "cape")
    # Default should also work through full getter, which creates dir
    cachedir = capeconfig.get_cape_cachedir()
    assert cachedir == str(clean_env / ".cache" / "cape")
    assert os.path.isdir(cachedir)


def test_default_cachedir_invalid_xdg(clean_env, monkeypatch):
    # Set home folder to temp location
    monkeypatch.setenv("HOME", str(clean_env))
    # Empty $XDG_CACHE_HOME is ignored per the XDG spec
    monkeypatch.setenv("XDG_CACHE_HOME", "")
    assert capeconfig.get_default_cachedir() == os.path.join(
        "~", ".cache", "cape")
    # Relative $XDG_CACHE_HOME is also invalid per the XDG spec
    monkeypatch.setenv("XDG_CACHE_HOME", "relative/cache")
    assert capeconfig.get_default_cachedir() == os.path.join(
        "~", ".cache", "cape")


def test_xdg_cachedir(clean_env, monkeypatch):
    # Set $XDG_CACHE_HOME to absolute path
    xdg_cachedir = clean_env / "xdg-cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg_cachedir))
    # Check default
    assert capeconfig.get_default_cachedir() == str(xdg_cachedir / "cape")
    # Full getter should match and create the folder
    cachedir = capeconfig.get_cape_cachedir()
    assert cachedir == str(xdg_cachedir / "cape")
    assert os.path.isdir(cachedir)
    # Also works through generic option getter
    assert capeconfig.get_cape_opt("CacheDir") == cachedir


def test_envvar_overrides_xdg(clean_env, monkeypatch):
    # Set both env vars
    cachedir_expected = clean_env / "cape-env"
    monkeypatch.setenv("XDG_CACHE_HOME", str(clean_env / "xdg-cache"))
    monkeypatch.setenv("CAPE_CACHE_DIR", str(cachedir_expected))
    # CAPE-specific env var takes precedence
    assert capeconfig.get_cape_cachedir() == str(cachedir_expected)


def test_configfile_overrides_xdg(clean_env, monkeypatch):
    # Set *CacheDir* in config file
    cachedir_expected = clean_env / "cape-cfg"
    configfile = clean_env / "capeconfig.json"
    configfile.write_text(json.dumps({"CacheDir": str(cachedir_expected)}))
    # Value from config file takes precedence over $XDG_CACHE_HOME
    monkeypatch.setenv("XDG_CACHE_HOME", str(clean_env / "xdg-cache"))
    assert capeconfig.get_cape_cachedir() == str(cachedir_expected)


def test_cachedir_expands_user(clean_env, monkeypatch):
    # Users can still use '~' in *CacheDir* or $CAPE_CACHE_DIR
    monkeypatch.setenv("HOME", str(clean_env))
    monkeypatch.setenv("CAPE_CACHE_DIR", os.path.join("~", "mycache"))
    assert capeconfig.get_cape_cachedir() == str(clean_env / "mycache")
