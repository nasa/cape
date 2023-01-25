
# Third-party
import pytest

# Local imports
from cape.optdict.optitem import OptdictTypeError, assert_array, check_array


# Test some sample values
def test_arraydepth01():
    assert check_array("a", 0) is True
    assert check_array("a", 1) is False
    assert check_array([[], [1]], 2) is True


# Test some failures w/ assert
def test_arraydepth02():
    assert_array([[[1, 2]]], 3)
    with pytest.raises(OptdictTypeError):
        assert_array("a", 1)
    with pytest.raises(OptdictTypeError):
        assert_array([1, 2], 2, "Some option")

