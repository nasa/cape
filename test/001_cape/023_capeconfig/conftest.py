# Standard library
import getpass

# Third-party
import pytest

# Local imports
from cape import capeconfig


# Current user (key for module's config cache)
USER = getpass.getuser()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    # Delete env vars that affect config, cache, and state folders
    monkeypatch.delenv("CAPE_CACHE_DIR", raising=False)
    monkeypatch.delenv("CAPE_STATE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    # Point config file at empty location; clear config cache
    monkeypatch.setenv("CAPE_CONFIG_FILE", str(tmp_path / "capeconfig.json"))
    capeconfig.CONFIG_CACHE.pop(USER, None)
    yield tmp_path
    capeconfig.CONFIG_CACHE.pop(USER, None)
