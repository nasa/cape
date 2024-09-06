
# Standard library
from io import StringIO

# Third-party
import numpy as np
import pytest

# Local imports
from cape.dkit.ftypes.capefile import (
    CapeFileTypeError,
    CapeFileValueError,
    assert_isinstance,
    assert_size,
    assert_value,
)


# Basic tests that pass
def test_util01_pass():
    # These are null tests
    assert_isinstance(3, None)
    assert_value(3, None)
    assert_value(3, 3)
    # This is a size test that passes
    assert_size(np.zeros(40), 40, None)


# Errors
def test_util02_err():
    # Test isinstance error
    with pytest.raises(CapeFileTypeError):
        assert_isinstance(3, float, "test")
    # Multiple allowed classes
    with pytest.raises(CapeFileTypeError):
        assert_isinstance(3, (float, str), "test")
    # Test value error
    with pytest.raises(CapeFileValueError):
        assert_value(3, (2, 4), "test")
    with pytest.raises(CapeFileValueError):
        assert_value(3, 2, "test")
    # Test size error
    fp = StringIO()
    with pytest.raises(CapeFileValueError):
        assert_size(np.ones(5), 4, fp)
