

# Third-party
import pytest

# Local
from cape import optdict


def test_bad_arg0():
    with pytest.raises(TypeError):
        opts = optdict.OptionsDict(1)

