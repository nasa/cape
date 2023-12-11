

# Third-party
import pytest

# Local
from cape.optdict import OptionsDict, OptdictTypeError


def test_bad_arg0():
    with pytest.raises(OptdictTypeError):
        OptionsDict(1)

