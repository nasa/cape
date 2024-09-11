

# Third-party
import pytest

# Local
from cape.argread import readkeys


def test_bad_argv():
    with pytest.raises(TypeError):
        a, kw = readkeys("p -cj")


def test_bad_args():
    with pytest.raises(TypeError):
        a, kw = readkeys(["p", "-cj", True])


def test_empty_argv():
    with pytest.raises(TypeError):
        a, kw = readkeys([])
