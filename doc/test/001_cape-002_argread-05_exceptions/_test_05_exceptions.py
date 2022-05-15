

# Third-party
import pytest

# Local
import argread


def test_bad_argv():
    with pytest.raises(TypeError):
        a, kw = argread.readkeys("p -cj")


def test_bad_args():
    with pytest.raises(TypeError):
        a, kw = argread.readkeys(["p", "-cj", True])


def test_empty_argv():
    with pytest.raises(IndexError):
        a, kw = argread.readkeys([])